"""Sweep credit_reserve to test the hypothesis: expansion is starved because
production spends the bank down to reserve=300 and the extractor gate needs 700.

If raising reserve lets funds accumulate past the extractor price, more than 2
extractors should appear and the army/income should grow. We measure extractors,
income, peak army and win rate at difficulty 1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def run(seed: int, port: int, reserve: float) -> dict:
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
        economy_army_cap=40,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=reserve,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="resweep",
                      strategy=RushStrategy(attack_at=10, max_factories=2,
                                            max_extractors=6, reserve=reserve,
                                            army_cap=40))
    peak = 0
    res = None
    income = None
    extractors = 0
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            slots = env.state.get("resource_slots") or []
            extractors = max(extractors, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "extractors": extractors, "income": income}


def main() -> int:
    port = 28010
    for reserve in (300.0, 700.0, 1000.0, 1500.0):
        outcomes = [run(71000 + i, port + i, reserve) for i in range(4)]
        port += 10
        wins = sum(1 for o in outcomes if o["res"] == "victory")
        print(f"reserve={reserve:6.0f} wins={wins}/4  " +
              json.dumps(outcomes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
