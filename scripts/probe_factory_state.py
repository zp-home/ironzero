"""Is the factory that accepts the order actually able to work on it?

``produceUnit`` returns true, the queue grows with the requested type, and every item
sits at ``progress = -1.0`` forever. "Accepted" therefore means only "an action id was
dispatched", not "the factory is producing".

This samples the accepting factory's own state at the moment of the order -- build
progress, operational flag, tech level, queue length, and its live ``can_produce``
menu -- so the reason the queue never advances is read from the engine instead of
inferred.

Usage::

    python scripts/probe_factory_state.py [--decisions 400]
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
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--port", type=int, default=23_300)
    args = ap.parse_args()

    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
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
        game, port=args.port, seed=args.port, episode_prefix="facstate", strategy=RuleStrategy()
    )
    printed = 0
    try:
        env.reset()
        for step in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            order = getattr(env, "curriculum_order", None)
            if isinstance(order, dict) and order.get("kind") == "produce_unit" and printed < 10:
                target_id = order.get("factory_entity_id")
                detail = None
                for entry in env.state.get("factories") or []:
                    if isinstance(entry, dict) and entry.get("entity_id") == target_id:
                        detail = entry
                        break
                print(
                    json.dumps(
                        {
                            "step": step,
                            "tick": env.state.get("tick"),
                            "order": order,
                            "factory": detail,
                            "funds": (env.state.get("global_state") or {}).get("funds"),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
                printed += 1
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
