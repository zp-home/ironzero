"""What is actually sitting in the factory queue, and why does it never finish?

``probe_stage_blockers`` shows the decisive shape: ``factories=4``, ``queue=3``,
``funds=1624``, and ``army=0`` for the whole episode. The curriculum is topping the
queue up correctly, so the question narrows to the queue contents: which unit type
was enqueued, and does its progress ever advance?

``IronZeroBridge.queueProgress`` reads ``unit.getConstructionProgress()``, which is
the *building's own* construction progress -- not the production item's. On a
finished factory that is 1.0 or 0.0 and says nothing about the queue. The
authoritative per-item progress is in the ``queued`` array
(``queue[].progress``), which the bridge does expose.

This prints both, per sample, so the two can be compared directly.

Usage::

    python scripts/probe_queue_items.py [--decisions 900]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=900)
    ap.add_argument("--port", type=int, default=19_300)
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
    from ironzero.strategy.rule_strategy import RuleStrategy

    env = RwppDuelEnv(
        game, port=args.port, seed=99_000, episode_prefix="queueitems", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            if step % 25 == 0:
                facs = []
                for entry in env.state.get("factories") or []:
                    if not isinstance(entry, dict):
                        continue
                    facs.append(
                        {
                            "id": entry.get("entity_id"),
                            "type": entry.get("unit_type"),
                            "can_produce": entry.get("can_produce"),
                            "prices": entry.get("can_produce_price"),
                            "queue_length": entry.get("queue_length"),
                            "queue_progress": entry.get("queue_progress"),
                            "queued": entry.get("queued"),
                        }
                    )
                own = env.state.get("own_entities") or []
                army = sum(
                    1
                    for u in own
                    if u.get("can_attack")
                    and not u.get("is_builder")
                    and not u.get("is_building")
                    and float(u.get("hp") or 0.0) > 0.0
                )
                rows.append(
                    {
                        "step": step,
                        "tick": env.state.get("tick"),
                        "army": army,
                        "funds": (env.state.get("global_state") or {}).get("funds"),
                        "factories": facs,
                    }
                )
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    (ROOT / ".runtime" / "queue_items.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
