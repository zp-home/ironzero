"""Concentrate on the LOWEST-HP enemy factory, and keep re-issuing it.

probe_concentrate suppressed the curriculum attack but still used pick_target's
"nearest factory" choice, which changes as the army moves -- so the army
oscillates between targets. This locks ONE target for the whole leg (the enemy
factory with the least HP, i.e. the one closest to dying) and only re-selects it
when that target is destroyed or leaves vision.

Rationale: an enemy landFactory at 25 HP needs one more volley, not a new march.
Focusing the army on a nearly-dead factory converts scattered damage into a kill,
and a kill is what decrements mainBuildings.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def enemy_factories(env) -> list[dict]:
    return [u for u in (env.state.get("visible_enemy_entities") or [])
            if u.get("is_building") and u.get("is_factory")
            and "extractor" not in str(u.get("unit_type", "")).lower()
            and not u.get("is_turret")
            and float(u.get("hp") or 0) > 0]


def run(seed: int, port: int) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=16,
        units_per_team=16,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
        opening_attack=False,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="focus",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    locked_id = None
    kills = 0
    prev_ids: set[int] = set()
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            facs = enemy_factories(env)
            ids = {int(u["entity_id"]) for u in facs if u.get("entity_id") is not None}
            # Count factories that disappeared from vision (destroyed or lost sight).
            for gone in prev_ids - ids:
                kills += 1
            prev_ids = ids
            if locked_id is not None and locked_id not in ids:
                locked_id = None
            if locked_id is None and facs:
                weakest = min(facs, key=lambda u: float(u.get("hp") or 0))
                locked_id = int(weakest["entity_id"])
            if locked_id is not None:
                foe = env.state.get("visible_enemy_entities") or []
                try:
                    slot = next(i for i, u in enumerate(foe)
                                if u.get("entity_id") == locked_id)
                except StopIteration:
                    slot = 0
                slot = min(slot, max(0, int(env.micro.target_slots) - 1))
                act = np.array([ACTION_KINDS.index("attack_target"), 4, 0, 0, slot],
                               dtype=np.int64)
            else:
                act = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            _o, _r, t, tr, info = env.step(act)
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            if t or tr:
                res = info.get("result")
                tick = env.state.get("tick")
                break
        end_facs = [(u.get("unit_type"), round(float(u.get("hp") or 0)))
                    for u in enemy_factories(env)]
    finally:
        env.close()
    return {"res": res, "peak": peak, "tick": tick, "kills": kills, "end_facs": end_facs}


def main() -> int:
    n = 8
    wins = 0
    port = 28750
    rows = []
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port)
        port += 1
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} "
              f"tick={r['tick']} facs_left={r['end_facs']}", flush=True)
    print(f"\n=== focus weakest factory, d1 Lake: {wins}/{n} ===", flush=True)
    Path("runs/focus_weakest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
