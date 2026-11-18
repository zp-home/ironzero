"""Are the reported pool coordinates places a builder can actually stand?

The bridge reports "ordered=1" for every extractor order, the builder does move, but
no second extractor ever appears. If the coordinates the bridge resolves from
``slot_id`` are not walkable/buildable, the engine accepts the waypoint and then
silently drops it -- which matches every measurement so far.

``buildWithBuilders`` already calls ``Building.getPlacementError`` before accepting,
so a *rejected* order would have printed "placement illegal". Nothing did, which
means either the call silently throws (it is wrapped in a try/catch) or the tile is
legal but unreachable.

This prints each pool's coordinates and the builder's distance to it, and flags
pools that are far from every walkable area we can see.

Usage::

    python scripts/probe_pool_geo.py
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
    ap.add_argument("--decisions", type=int, default=120)
    ap.add_argument("--port", type=int, default=22_000)
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
        game, port=args.port, seed=args.port, episode_prefix="poolgeo", strategy=RuleStrategy()
    )
    try:
        env.reset()
        for _ in range(args.decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
        slots = env.state.get("resource_slots") or []
        own = env.state.get("own_entities") or []
        builder = next(
            (u for u in own if u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0),
            None,
        )
        base = next(
            (
                u
                for u in own
                if "commandcenter" in str(u.get("unit_type") or "").lower()
            ),
            None,
        )
        print(f"builder at {builder.get('x') if builder else None},{builder.get('y') if builder else None}")
        print(f"base    at {base.get('x') if base else None},{base.get('y') if base else None}")
        print(f"map     {game.map_width_px} x {game.map_height_px}")
        print("slots:")
        for index, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            sx, sy = float(slot.get("x") or 0), float(slot.get("y") or 0)
            dist_builder = None
            if builder is not None:
                dist_builder = round(
                    float(
                        np.hypot(
                            sx - float(builder.get("x") or 0),
                            sy - float(builder.get("y") or 0),
                        )
                    ),
                    1,
                )
            print(
                json.dumps(
                    {
                        "index": index,
                        "slot_id": slot.get("slot_id"),
                        "x": sx,
                        "y": sy,
                        "occupied": slot.get("occupied"),
                        "extractor_level": slot.get("extractor_level"),
                        "dist_from_builder": dist_builder,
                    }
                )
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
