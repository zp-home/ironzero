"""Why does the production-targeting army (peak 24-30!) still not destroy a
single enemy factory?

peak 24-30 is double the previous 13-19, so the targeting helps the army survive.
But the objective verdict needs enemy mainBuildings == 0. Two candidate blockers:
  A. The army never reaches the enemy base (water / pathing), so factories take
     no damage.
  B. The army arrives but cannot out-damage repairs + defenders.

This logs the minimum distance the army reaches and the enemy factories' HP over
time, which distinguishes A from B.
"""
from __future__ import annotations

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
from scripts.probe_target_production import pick_target


def main() -> int:
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
    )
    env = RwppDuelEnv(game, port=28650, seed=71003, episode_prefix="reach",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    best_y = None
    best_dist = None
    seen_hp: dict[int, float] = {}
    try:
        env.reset(seed=71003)
        for step in range(19674 // 10 + 60):
            target = pick_target(env)
            foe = env.state.get("visible_enemy_entities") or []
            if target is not None and step >= 8:
                try:
                    slot = next(i for i, u in enumerate(foe)
                                if u.get("entity_id") == target.get("entity_id"))
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
            if com:
                ys = [float(u.get("y") or 0) for u in com]
                y = min(ys)
                best_y = y if best_y is None else min(best_y, y)
            # Track enemy factory hp
            for u in foe:
                if u.get("is_building") and u.get("is_factory"):
                    eid = u.get("entity_id")
                    if eid is not None:
                        seen_hp[int(eid)] = float(u.get("hp") or 0)
            if target is not None and com:
                cx = sum(float(u.get("x") or 0) for u in com) / len(com)
                cy = sum(float(u.get("y") or 0) for u in com) / len(com)
                d = math.hypot(float(target.get("x") or 0) - cx,
                               float(target.get("y") or 0) - cy)
                best_dist = d if best_dist is None else min(best_dist, d)
            if step % 300 == 0 and com:
                print(f"step={step:4d} tick={env.state.get('tick')} army={len(com)} "
                      f"min_y={best_y:.0f} target={target.get('unit_type') if target else None} "
                      f"dist={best_dist:.0f}" if best_dist else "", flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
        print(f"\nclosest approach to a target: {best_dist}", flush=True)
        print(f"army's northernmost y: {best_y}", flush=True)
        print(f"enemy factory HP seen (last): {seen_hp}", flush=True)
        foe = env.state.get("visible_enemy_entities") or []
        facs = [u for u in foe if u.get("is_building") and u.get("is_factory")]
        print(f"enemy factories still alive: {[(u.get('unit_type'), u.get('hp')) for u in facs]}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
