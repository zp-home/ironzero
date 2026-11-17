"""How much of the episode does a single extractor leg actually consume?

The measured facts so far: ``occupied`` reaches 1 at ~tick 2000 and never moves
again, ``build_extractor`` is ordered 23 times in 1400 decisions, and the map has 8
free pools the whole time. A leg to the nearest free pool is documented as ~1147
ticks of walking plus a 140-tick build.

If one leg costs ~1300 ticks and the episode is 19674 ticks, there is time for many
legs -- unless the stage's own re-issue budget keeps declining to start a new one.
This measures the actual wall-clock: when each ``build_extractor`` is dispatched,
to which pool, and how the builder's distance to that pool evolves.

Usage::

    python scripts/probe_expansion_timeline.py [--difficulty 1]
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
    ap.add_argument("--decisions", type=int, default=1800)
    ap.add_argument("--port", type=int, default=21_200)
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
        game, port=args.port, seed=args.port, episode_prefix="timeline", strategy=RuleStrategy()
    )
    events: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            current = getattr(env, "curriculum_order", None)
            kind = current.get("kind") if isinstance(current, dict) else None
            if kind == "build_extractor":
                own = env.state.get("own_entities") or []
                builder = next(
                    (
                        u
                        for u in own
                        if u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0
                    ),
                    None,
                )
                slot_id = current.get("slot_id")
                slot = next(
                    (
                        s
                        for s in (env.state.get("resource_slots") or [])
                        if isinstance(s, dict) and int(s.get("slot_id", -1)) == int(slot_id or -1)
                    ),
                    None,
                )
                dist = None
                if builder and slot:
                    try:
                        dist = round(
                            float(
                                np.hypot(
                                    float(builder.get("x") or 0)
                                    - float(slot.get("x") or 0),
                                    float(builder.get("y") or 0)
                                    - float(slot.get("y") or 0),
                                )
                            ),
                            1,
                        )
                    except (TypeError, ValueError):
                        dist = None
                occupied = sum(
                    1
                    for s in (env.state.get("resource_slots") or [])
                    if isinstance(s, dict) and s.get("occupied")
                )
                events.append(
                    {
                        "step": step,
                        "tick": env.state.get("tick"),
                        "slot_id": slot_id,
                        "occupied": occupied,
                        "builder_dist": dist,
                        "funds": round(
                            float((env.state.get("global_state") or {}).get("funds") or 0)
                        ),
                    }
                )
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for event in events:
        print(json.dumps(event))
    print(f"\ntotal build_extractor dispatches: {len(events)}")
    (ROOT / ".runtime" / "expansion_timeline.json").write_text(
        json.dumps(events, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
