"""What does the curriculum order on each decision, WITH the loop as it runs?

The isolation probe was misleading: calling stages one at a time mutates latches
(``_send_curriculum_order`` sets ``curriculum_order`` and the stalled counters), so
each stage's own "nothing to do" verdict was partly self-inflicted.

This instead reads only ``env.curriculum_order`` after a normal ``step`` -- the
field the dispatcher itself sets when a stage fires -- so nothing is perturbed.

Usage::

    python scripts/probe_curriculum_orders.py [--difficulty 1]
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
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=1400)
    ap.add_argument("--port", type=int, default=20_900)
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
        game, port=args.port, seed=args.port, episode_prefix="corders", strategy=RuleStrategy()
    )
    orders: Counter[str] = Counter()
    empty = 0
    try:
        env.reset()
        for step in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            current = getattr(env, "curriculum_order", None)
            if isinstance(current, dict) and current.get("kind"):
                orders[str(current.get("kind"))] += 1
            else:
                empty += 1
            if step % 200 == 0:
                own = env.state.get("own_entities") or []
                occ = sum(
                    1
                    for s in (env.state.get("resource_slots") or [])
                    if isinstance(s, dict) and s.get("occupied")
                )
                print(
                    json.dumps(
                        {
                            "step": step,
                            "tick": env.state.get("tick"),
                            "occupied": occ,
                            "income": (env.state.get("global_state") or {}).get("income"),
                            "funds": round(
                                float(
                                    (env.state.get("global_state") or {}).get("funds") or 0
                                )
                            ),
                            "army": sum(
                                1
                                for u in own
                                if u.get("can_attack")
                                and not u.get("is_builder")
                                and not u.get("is_building")
                                and float(u.get("hp") or 0.0) > 0.0
                            ),
                            "last_order": current.get("kind") if isinstance(current, dict) else None,
                        }
                    )
                )
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("\ncurriculum orders:", json.dumps(dict(orders.most_common()), indent=2))
    print(f"decisions with no order: {empty}")
    (ROOT / ".runtime" / "curriculum_orders.json").write_text(
        json.dumps({"orders": dict(orders), "empty": empty}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
