"""Raid the enemy's EXTRACTORS to cut the 1.4x income at its source.

Every difficulty-1 loss traces back to one number: the AI earns 1.4x our income.
Rounds 15-19 attacked enemy factories (1200 HP) and the command centre (4000 HP),
but neither touches income directly -- a destroyed factory is rebuilt, and the
command centre sits inside the anti-air net.

An extractor is 800 HP and is the thing that *generates* the income. Killing the
extractor chain caps enemy production permanently, which is the only way to stop
being out-economised. ``mainBuildings`` deliberately excludes extractors, so this
does not score a victory by itself -- it removes the advantage that makes the
dominance ratio unreachable.

Sweeps two target priorities:
  * "extractor"  -- enemy extractors first, then factories
  * "factory"    -- the round-15 baseline (nearest/weakest factory), for contrast
"""
from __future__ import annotations

import json
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


def is_extractor(u: dict) -> bool:
    return "extractor" in str(u.get("unit_type", "")).lower()


def pick(mode: str, env) -> dict | None:
    foe = [u for u in (env.state.get("visible_enemy_entities") or [])
           if u.get("is_building") and float(u.get("hp") or 0) > 0]
    if not foe:
        return None
    exts = [u for u in foe if is_extractor(u)]
    facs = [u for u in foe
            if u.get("is_factory") and not u.get("is_turret") and not is_extractor(u)]
    if mode == "extractor":
        pool = exts or facs
    else:
        pool = facs or exts
    if not pool:
        return None
    return min(pool, key=lambda u: float(u.get("hp") or 0))


def run(seed: int, port: int, mode: str) -> dict:
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
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
        opening_attack=False,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"raid_{mode}",
                      strategy=RushStrategy(attack_at=6, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    locked = None
    enemy_ext_seen = 0
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            target = pick(mode, env)
            foe = env.state.get("visible_enemy_entities") or []
            exts = [u for u in foe if is_extractor(u) and float(u.get("hp") or 0) > 0]
            enemy_ext_seen = max(enemy_ext_seen, len(exts))
            tid = int(target["entity_id"]) if target and target.get("entity_id") is not None else None
            if tid is not None and tid != locked:
                locked = tid
            if locked is not None and locked not in {
                int(u["entity_id"]) for u in foe if u.get("entity_id") is not None
            }:
                locked = None
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            if locked is not None and len(com) >= 6:
                try:
                    slot = next(i for i, u in enumerate(foe) if u.get("entity_id") == locked)
                except StopIteration:
                    slot = 0
                slot = min(slot, max(0, int(env.micro.target_slots) - 1))
                act = np.array([ACTION_KINDS.index("attack_target"), 4, 0, 0, slot],
                               dtype=np.int64)
            else:
                act = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            _o, _r, t, tr, info = env.step(act)
            if t or tr:
                res = info.get("result")
                tick = env.state.get("tick")
                break
        left = sorted({str(u.get("unit_type")) for u in
                       (env.state.get("visible_enemy_entities") or [])
                       if u.get("is_building") and float(u.get("hp") or 0) > 0})
    finally:
        env.close()
    return {"res": res, "peak": peak, "tick": tick, "enemy_ext_seen": enemy_ext_seen,
            "enemy_buildings_left": left}


def main() -> int:
    port = 29800
    for mode in ("extractor", "factory"):
        rows = []
        wins = 0
        for i in range(8):
            seed = 71000 + i
            r = run(seed, port, mode)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        print(f"mode={mode}: wins={wins}/8 peak={[r['peak'] for r in rows]} "
              f"enemy_ext_seen={[r['enemy_ext_seen'] for r in rows]}", flush=True)
        for r in rows:
            print(f"   seed={r['seed']} {r['res']:8s} enemy_left={r['enemy_buildings_left']}",
                  flush=True)
        Path(f"runs/raid_{mode}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
