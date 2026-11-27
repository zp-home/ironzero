"""Amphibious flanking: route hoverTanks around the enemy main force to factories.

Round 15 found: production-targeting doubles peak army (13-19 -> 24-30), and a
landFactory was damaged to 25 HP, but the army stalls 411-1167 px short because
it fights the enemy main force at y~314 instead of reaching the factories at
y~210.

Round 16 confirmed factory capacity is NOT the bottleneck (3 landFactories,
29+ tanks produced). The remaining obstacle is the interception itself.

hoverTank is HOVER (amphibious) and can_attack_air. This probe routes the WHOLE
army across the lake (waypoint at the water's edge, then the enemy factory),
bypassing the ground choke point where the enemy army waits.
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
import ironzero.rwpp.env as env_mod
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


def run(seed: int, port: int, use_hover: bool) -> dict:
    if use_hover:
        env_mod._UNIT_COMBAT_RANK["hoverTank"] = 6
    else:
        env_mod._UNIT_COMBAT_RANK["hoverTank"] = 3
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
    env = RwppDuelEnv(game, port=port, seed=seed,
                      episode_prefix=f"flank{'h' if use_hover else 'g'}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    best_dist = None
    locked_id = None
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            facs = enemy_factories(env)
            ids = {int(u["entity_id"]) for u in facs if u.get("entity_id") is not None}
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
            if locked_id is not None and com:
                tgt = next((u for u in foe if u.get("entity_id") == locked_id), None)
                if tgt:
                    cx = sum(float(u.get("x") or 0) for u in com) / len(com)
                    cy = sum(float(u.get("y") or 0) for u in com) / len(com)
                    d = math.hypot(float(tgt.get("x") or 0) - cx,
                                   float(tgt.get("y") or 0) - cy)
                    best_dist = d if best_dist is None else min(best_dist, d)
            if t or tr:
                res = info.get("result")
                break
        facs_left = [(u.get("unit_type"), round(float(u.get("hp") or 0)))
                     for u in enemy_factories(env)]
    finally:
        env.close()
    return {"res": res, "peak": peak, "closest": round(best_dist or -1),
            "facs_left": facs_left}


def main() -> int:
    for use_hover in (False, True):
        n = 8
        wins = 0
        port = 28950 if use_hover else 28900
        print(f"\n=== use_hover={use_hover} ===", flush=True)
        for i in range(n):
            seed = 71000 + i
            r = run(seed, port, use_hover)
            port += 1
            if r["res"] == "victory":
                wins += 1
            print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} "
                  f"closest={r['closest']} facs={r['facs_left']}", flush=True)
        print(f"wins={wins}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
