"""Print the raw receipt payload so its schema is visible instead of guessed.

The earlier probe counted ``kind: None`` on 1000 of 1000 receipts, which means the
field names assumed there were wrong. Receipts carry the bridge's per-order
outcome, and the whole "is production being refused?" question depends on reading
them correctly.

Usage::

    python scripts/probe_receipt_schema.py [--decisions 300]
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
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--port", type=int, default=20_400)
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
        game, port=args.port, seed=args.port, episode_prefix="schema", strategy=RuleStrategy()
    )
    shown = 0
    try:
        env.reset()
        for _ in range(args.decisions):
            before = len(env.receipts)
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            for receipt in env.receipts[before:]:
                if shown < 8:
                    print(json.dumps(receipt, ensure_ascii=False, default=str))
                    shown += 1
            if shown >= 8 or term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
