"""Ask the bridge itself why a build order does not move the builder.

Every order in this run comes back ``accepted: true``, yet the builder's distance to
the pool it was sent to *grows* (894 -> 1111 px) and the pool is never built. Since
the refusal -- if any -- happens inside the engine, only the bridge can name it:
``--debug-build`` prints every gate (``canQueueAction``, ``isValidNewWaypoint``,
placement legality, "no builder passed every gate").

This launches one game with that flag, drives a noop policy so the curriculum is the
only actor, and prints the bridge's own build trace.

Usage::

    python scripts/probe_build_trace.py [--difficulty 1] [--decisions 300]
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--port", type=int, default=21_500)
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
        debug_build=True,
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
        game, port=args.port, seed=args.port, episode_prefix="buildtrace", strategy=RuleStrategy()
    )
    try:
        env.reset()
        for _ in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    print("bridge log lines are in the episode log file next to the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
