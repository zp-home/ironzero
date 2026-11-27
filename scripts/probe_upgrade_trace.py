"""Does the curriculum's upgrade order actually start progressing?

With the ``_upgrade_in_flight`` latch the stage issues exactly one upgrade per
episode (measured: issued=1 on 4/4 seeds), but the extractor still ends at
tech_level 1. Two possibilities remain:

  * the curriculum's order never reaches the engine (a dispatch-path difference
    from the working ``env.step`` path), or
  * it reaches the engine but starts too late to finish before the tick cap.

This watches the target extractor's ``queue_progress`` and the bank immediately
after the stage fires, which distinguishes the two: a charge plus rising progress
means "started but too late"; no charge means "never dispatched".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


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
        max_entities=8,
        units_per_team=8,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=4,
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=31600, seed=71000, episode_prefix="upgtrace",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56, max_upgrades=4))
    prev_issued = 0
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            issued = int(env._upgrades_issued)
            if issued != prev_issued:
                prev_issued = issued
                gs = env.state.get("global_state") or {}
                own = env.state.get("own_entities") or []
                ex = [
                    (u.get("entity_id"), u.get("unit_type"), u.get("tech_level"),
                     u.get("queue_length"),
                     round(float(u.get("queue_progress") or 0.0), 3))
                    for u in own if "xtractor" in str(u.get("unit_type", "")).lower()
                ]
                print(f"FIRED at step={step} tick={env.state.get('tick')} "
                      f"funds={gs.get('funds')} in_flight={env._upgrade_in_flight} "
                      f"extractors={ex}", flush=True)
                for offset in range(1, 121):
                    _o, _r, t2, tr2, _i2 = env.step(np.array([9, 0, 0, 0, 0]))
                    if offset % 20 == 0:
                        gs2 = env.state.get("global_state") or {}
                        own2 = env.state.get("own_entities") or []
                        ex2 = [
                            (u.get("unit_type"), u.get("tech_level"),
                             round(float(u.get("queue_progress") or 0.0), 3))
                            for u in own2
                            if "xtractor" in str(u.get("unit_type", "")).lower()
                        ]
                        print(f"   +{offset:3d} funds={gs2.get('funds')} "
                              f"income={gs2.get('income')} ex={ex2}", flush=True)
                    if t2 or tr2:
                        break
                break
            if t or tr:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
