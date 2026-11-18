"""End-to-end: with every fix in place, does income ever exceed 26?

This is the acceptance check for the economy chain, independent of win rate. Income
26 == base 18 + one extractor. If the chain works, income must climb as extractors
accumulate; if it never does, no amount of PPO will beat a difficulty-1 opponent
holding 30-38 units.

Run with a long episode and a *noop* policy so the curriculum is the only actor:
this measures the harness, not a policy.

Usage::

    python scripts/probe_economy_chain.py [--decisions 1900]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1900)
    ap.add_argument("--port", type=int, default=22_100)
    ap.add_argument("--difficulty", type=int, default=1)
    args = ap.parse_args()

    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=args.difficulty,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=1,
        physics_ms=16,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=20,
        economy_max_extractors=9,
        economy_max_factories=3,
        economy_production_queue_depth=3,
    )
    env = RwppDuelEnv(
        game, port=args.port, seed=args.port, episode_prefix="chain", strategy=RuleStrategy()
    )
    best_income = 0.0
    best_occupied = 0
    peak_army = 0
    result = None
    try:
        env.reset()
        for step in range(args.decisions):
            own = env.state.get("own_entities") or []
            income = float((env.state.get("global_state") or {}).get("income") or 0.0)
            occupied = sum(
                1
                for s in (env.state.get("resource_slots") or [])
                if isinstance(s, dict) and s.get("occupied")
            )
            army = sum(
                1
                for u in own
                if u.get("can_attack")
                and not u.get("is_builder")
                and not u.get("is_building")
                and float(u.get("hp") or 0.0) > 0.0
            )
            best_income = max(best_income, income)
            best_occupied = max(best_occupied, occupied)
            peak_army = max(peak_army, army)
            if step % 200 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "tick": env.state.get("tick"),
                            "income": income,
                            "occupied": occupied,
                            "army": army,
                        }
                    )
                )
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                result = info.get("result")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(
        f"\nBEST income={best_income} occupied={best_occupied} peak_army={peak_army} result={result}"
    )
    (ROOT / ".runtime" / "economy_chain.json").write_text(
        json.dumps(
            {
                "best_income": best_income,
                "best_occupied": best_occupied,
                "peak_army": peak_army,
                "result": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
