"""Who moves the builder? Correlate the builder's trajectory with the order log.

The builder's distance to its target pool *grows* even on legs that start 306 px
away, which rules out "the walk is too long". Something else is ordering it.
Candidates: the attack stage's rotating group orders, the factory stage's build
order, and the policy's own tactical head.

This records, on every decision, the builder's position, the curriculum order that
was dispatched, and whether any order in the dispatch targeted a group that
contains the builder.

Usage::

    python scripts/probe_builder_interference.py [--difficulty 1]
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
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=700)
    ap.add_argument("--port", type=int, default=21_800)
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
        game, port=args.port, seed=args.port, episode_prefix="interfere", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            own = env.state.get("own_entities") or []
            builder = next(
                (u for u in own if u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0),
                None,
            )
            before = {
                "bx": round(float(builder.get("x") or 0), 1) if builder else None,
                "by": round(float(builder.get("y") or 0), 1) if builder else None,
                "builder_order": builder.get("order") if builder else None,
            }
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            current = getattr(env, "curriculum_order", None)
            own_after = env.state.get("own_entities") or []
            builder_after = next(
                (
                    u
                    for u in own_after
                    if u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0
                ),
                None,
            )
            rows.append(
                {
                    "step": step,
                    "curriculum": current.get("kind") if isinstance(current, dict) else None,
                    **before,
                    "bx_after": round(float(builder_after.get("x") or 0), 1) if builder_after else None,
                    "by_after": round(float(builder_after.get("y") or 0), 1) if builder_after else None,
                    "order_after": builder_after.get("order") if builder_after else None,
                }
            )
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Print only the rows around curriculum orders, plus the builder's own order.
    for row in rows[:120]:
        print(json.dumps(row))
    print("\n--- builder order histogram ---")
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("order_after"))
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(dict(sorted(counts.items(), key=lambda kv: -kv[1])), indent=2))
    (ROOT / ".runtime" / "builder_interference.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
