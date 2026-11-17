"""Would expansion alone fill the map's pools? Remove the competing stage.

Every gate in the expansion stage has been checked, ``SHOULD_ORDER`` appears on 283
of 1400 decisions, and yet the occupied-pool count never leaves 1. The remaining
suspicion is direct competition: the factory stage spends the builder, and the two
cannot both run in one dispatch.

This runs the *same* episode twice -- once with the normal plan, once with
``factory`` removed from ``stage_names`` -- and compares occupied pools and income.
That isolates "expansion is starved by the factory stage" from every other
explanation in one measurement.

Usage::

    python scripts/probe_expansion_starvation.py [--difficulty 1]
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


class NoFactoryStrategy(RuleStrategy):
    """Same rules, but the plan never runs the ``factory`` stage."""

    def propose(self, observation_summary):  # type: ignore[override]
        directive = super().propose(observation_summary)
        macro = getattr(directive, "macro", None)
        if macro is not None and getattr(macro, "build_order", None):
            trimmed = tuple(g for g in macro.build_order if g != "factory")
            if trimmed:
                directive.macro.build_order = trimmed  # type: ignore[misc]
        return directive


def run(difficulty: int, port: int, strategy) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=difficulty,
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
    env = RwppDuelEnv(game, port=port, seed=port, episode_prefix="starve", strategy=strategy)
    best_occupied = 0
    best_income = 0.0
    best_army = 0
    result = None
    try:
        env.reset()
        for _ in range(1400):
            own = env.state.get("own_entities") or []
            occ = sum(
                1
                for s in (env.state.get("resource_slots") or [])
                if isinstance(s, dict) and s.get("occupied")
            )
            income = float((env.state.get("global_state") or {}).get("income") or 0.0)
            army = sum(
                1
                for u in own
                if u.get("can_attack")
                and not u.get("is_builder")
                and not u.get("is_building")
                and float(u.get("hp") or 0.0) > 0.0
            )
            best_occupied = max(best_occupied, occ)
            best_income = max(best_income, income)
            best_army = max(best_army, army)
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                result = info.get("result")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "occupied": best_occupied,
        "income": best_income,
        "army": best_army,
        "result": result,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    args = ap.parse_args()

    normal = run(args.difficulty, 21_000, RuleStrategy())
    no_factory = run(args.difficulty, 21_100, NoFactoryStrategy())
    print("with factory stage   :", json.dumps(normal))
    print("without factory stage:", json.dumps(no_factory))
    (ROOT / ".runtime" / "expansion_starvation.json").write_text(
        json.dumps({"with_factory": normal, "without_factory": no_factory}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
