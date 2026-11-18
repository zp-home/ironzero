"""Is the builder actually frozen, and does it ever receive an order?

``probe_builder_interference`` measured a builder that never moves: identical x/y
from step 103 through 119 and for all 700 decisions, with ``order`` always null.
The distance changes seen in ``probe_expansion_timeline`` were therefore the *pool*
moving (different slot ids), never the builder.

A frozen builder means no build order ever reaches it as a movement. This prints its
position over time alongside the dispatched order, so "no order" and "order that the
engine dropped" can be told apart.

Usage::

    python scripts/probe_builder_frozen.py [--decisions 600]
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
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--port", type=int, default=21_900)
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
        game, port=args.port, seed=args.port, episode_prefix="frozen", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if step % 50 == 0:
                own = env.state.get("own_entities") or []
                builders = [u for u in own if u.get("is_builder")]
                rows.append(
                    {
                        "step": step,
                        "tick": env.state.get("tick"),
                        "n_builders": len(builders),
                        "builder_ids": [b.get("entity_id") for b in builders],
                        "pos": [
                            [
                                round(float(b.get("x") or 0.0), 1),
                                round(float(b.get("y") or 0.0), 1),
                            ]
                            for b in builders
                        ],
                        "orders": [b.get("order") for b in builders],
                        "can_build": [b.get("can_build") for b in builders],
                        "curriculum": (
                            getattr(env, "curriculum_order", None) or {}
                        ).get("kind")
                        if isinstance(getattr(env, "curriculum_order", None), dict)
                        else None,
                    }
                )
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for row in rows:
        print(json.dumps(row))
    (ROOT / ".runtime" / "builder_frozen.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
