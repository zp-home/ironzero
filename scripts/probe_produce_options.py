"""Why does produce_options offer a unit the target factory cannot build?

Measured: the factory queue holds ``c_tank`` with ``progress = -1.0`` (never
started) while the land factory's menu is ``[builder, hoverTank, tank, artillery,
scout]`` -- so ``c_tank`` is not buildable there, and a wedged queue blocks
``production_queue_depth`` slots and starves real production.

Two candidate causes, and they have opposite fixes:

  A. ``produce_options`` itself includes an entry whose factory cannot build it
     (wrong source data);
  B. ``produce_options`` is correct at selection time but the index is resolved
     against a *different* list at decode time (busy factories drop out, so the
     ordinal shifts).

This prints ``produce_options`` next to each factory's own ``can_produce`` and the
live queue, at the exact decisions where a queue exists, so A and B separate.

Usage::

    python scripts/probe_produce_options.py [--decisions 700]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.observation.menus import produce_options  # noqa: E402
from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=700)
    ap.add_argument("--port", type=int, default=22_900)
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
        game, port=args.port, seed=args.port, episode_prefix="popts", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            state = env.state
            queue = []
            for entry in state.get("factories") or []:
                if not isinstance(entry, dict):
                    continue
                for item in entry.get("queued") or []:
                    queue.append(
                        {
                            "factory": entry.get("entity_id"),
                            "f_type": entry.get("unit_type"),
                            "unit": item.get("unit_type"),
                            "progress": item.get("progress"),
                        }
                    )
            if queue and len(rows) < 10:
                rows.append(
                    {
                        "step": step,
                        "menus": [
                            {
                                "id": e.get("entity_id"),
                                "type": e.get("unit_type"),
                                "can_produce": e.get("can_produce"),
                            }
                            for e in state.get("factories") or []
                            if isinstance(e, dict)
                        ],
                        "combat_ids": [
                            e.get("entity_id") for e in _combat_factories(state)
                        ],
                        "produce_options": produce_options(state),
                        "queue": queue,
                    }
                )
            env.step(np.array([9, 0, 0, 0, 0]))
            if len(rows) >= 10:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    (ROOT / ".runtime" / "produce_options.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
