"""Did the direct produce order get accepted, and did the queue drain?

The production stage now emits ``{"kind": "produce_unit", "factory_entity_id": N,
"unit_type": "tank"}``. This checks the two things that could still be wrong:

  1. the bridge rejects it (``PRODUCE_UNAVAILABLE``) -- then the order never queues;
  2. it is accepted and queues, but ``queued[].progress`` stays at -1.0 -- then the
     factory is not actually working (unaffordable, tech-locked, or not operational).

It prints, per decision, the dispatched order, the receipt outcome and the live
queue, so 1 and 2 separate without inference.

Usage::

    python scripts/probe_produce_acceptance.py [--decisions 400]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    ap.add_argument("--port", type=int, default=23_100)
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
        game, port=args.port, seed=args.port, episode_prefix="prodacc", strategy=RuleStrategy()
    )
    outcomes: Counter[str] = Counter()
    dispatched = 0
    printed = 0
    try:
        env.reset()
        for step in range(args.decisions):
            before = len(env.receipts)
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            order = getattr(env, "curriculum_order", None)
            kind = order.get("kind") if isinstance(order, dict) else None
            if kind == "produce_unit":
                dispatched += 1
                for receipt in env.receipts[before:]:
                    if isinstance(receipt, dict):
                        outcomes[
                            str(receipt.get("rejected_reason") or receipt.get("outcome") or "ok")
                        ] += 1
                if printed < 12:
                    queues = []
                    for entry in env.state.get("factories") or []:
                        if not isinstance(entry, dict):
                            continue
                        items = entry.get("queued") or []
                        if items:
                            queues.append(
                                {
                                    "id": entry.get("entity_id"),
                                    "type": entry.get("unit_type"),
                                    "q": [
                                        (i.get("unit_type"), i.get("progress"))
                                        for i in items
                                    ],
                                }
                            )
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "ordered": order,
                                "funds": (env.state.get("global_state") or {}).get("funds"),
                                "queues": queues,
                            },
                            ensure_ascii=False,
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

    print(f"\nproduce orders dispatched: {dispatched}")
    print("receipt outcomes:", json.dumps(dict(outcomes.most_common()), indent=2))
    (ROOT / ".runtime" / "produce_acceptance.json").write_text(
        json.dumps({"dispatched": dispatched, "outcomes": dict(outcomes)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
