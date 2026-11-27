"""At the end of an episode, what actually survives on each side?

``objective`` mode decides victory as "the enemy has no main building left"
(``mainBuildings == 0``) or "the enemy lost all production capability". If our army
never destroys those buildings, the check is unreachable by construction and every
episode is a timeout -- which is exactly what the training logs show.

This prints the surviving buildings per side at the end of a full episode, so
"our army never reached them" and "our army reached them and could not kill them"
are distinguishable.

Usage::

    python scripts/probe_endgame_buildings.py [--difficulty 1]
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

BUILDING_HINTS = (
    "factory",
    "commandcenter",
    "turret",
    "extractor",
    "commandcentre",
)


def classify(units: list[dict]) -> tuple[Counter, Counter]:
    buildings: Counter = Counter()
    mobile: Counter = Counter()
    for unit in units:
        name = str(unit.get("unit_type") or "?")
        name = name.split(".")[-1] if "." in name else name
        if unit.get("is_building") or any(h in name.lower() for h in BUILDING_HINTS):
            buildings[name] += 1
        else:
            mobile[name] += 1
    return buildings, mobile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=1900)
    ap.add_argument("--port", type=int, default=22_400)
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
        game, port=args.port, seed=args.port, episode_prefix="endgame", strategy=RuleStrategy()
    )
    result = None
    closest_seen = None
    try:
        env.reset()
        for _ in range(args.decisions):
            own = env.state.get("own_entities") or []
            foes = env.state.get("visible_enemy_entities") or []
            for a in own:
                for b in foes:
                    try:
                        d = float(
                            np.hypot(
                                float(a.get("x") or 0) - float(b.get("x") or 0),
                                float(a.get("y") or 0) - float(b.get("y") or 0),
                            )
                        )
                        if closest_seen is None or d < closest_seen:
                            closest_seen = d
                    except (TypeError, ValueError):
                        continue
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                result = info.get("result")
                break
        own = env.state.get("own_entities") or []
        foes = env.state.get("visible_enemy_entities") or []
        own_b, own_m = classify(own)
        foe_b, foe_m = classify(foes)
        print(
            json.dumps(
                {
                    "result": result,
                    "tick": env.state.get("tick"),
                    "closest_approach_px": round(closest_seen, 1)
                    if closest_seen is not None
                    else None,
                    "own_buildings": dict(own_b),
                    "own_mobile": dict(own_m),
                    "enemy_buildings": dict(foe_b),
                    "enemy_mobile": dict(foe_m),
                    "own_income": (env.state.get("global_state") or {}).get("income"),
                },
                indent=2,
                ensure_ascii=False,
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
