"""Is the economy time-starved or hard-blocked?

Every measurement so far caps at income 26 (base 18 + one extractor) within one
19674-tick episode. Two very different explanations fit that:

  A. the map is large and a pool leg is ~1220 ticks of walking, so 19674 ticks is
     simply not enough time to grow the economy *and* fight;
  B. something refuses the second extractor permanently, so no amount of time helps.

The distinguishing test is to give the same curriculum far more ticks. If income
climbs with a longer budget the chain is sound and the episode length was the
constraint; if it stays at 26 the block is hard.

Also reads ``order_type`` (the field the bridge actually emits) rather than the
``order`` key earlier probes looked for, so the builder's live order is visible.

Usage::

    python scripts/probe_economy_over_time.py [--ticks 60000]
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


def run(ticks: int, port: int, difficulty: int, decides_per_tick: int = 10) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=int(ticks),
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
    env = RwppDuelEnv(
        game, port=port, seed=port, episode_prefix=f"over{ticks}", strategy=RuleStrategy()
    )
    best_income = 0.0
    best_occupied = 0
    peak_army = 0
    timeline: list[dict] = []
    result = None
    decisions = int(ticks / decides_per_tick) + 50
    try:
        env.reset()
        for step in range(decisions):
            own = env.state.get("own_entities") or []
            income = float((env.state.get("global_state") or {}).get("income") or 0.0)
            occupied = sum(
                1
                for s in (env.state.get("resource_slots") or [])
                if isinstance(s, dict) and s.get("occupied")
            )
            army = sum(
                1
                for u in own
                if u.get("can_attack")
                and not u.get("is_builder")
                and not u.get("is_building")
                and float(u.get("hp") or 0.0) > 0.0
            )
            best_income = max(best_income, income)
            best_occupied = max(best_occupied, occupied)
            peak_army = max(peak_army, army)
            if step % 300 == 0:
                builder = next((u for u in own if u.get("is_builder")), None)
                timeline.append(
                    {
                        "tick": env.state.get("tick"),
                        "income": income,
                        "occupied": occupied,
                        "army": army,
                        "builder_order": builder.get("order_type") if builder else None,
                    }
                )
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
        "ticks": ticks,
        "best_income": best_income,
        "best_occupied": best_occupied,
        "peak_army": peak_army,
        "result": result,
        "timeline": timeline,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--short", type=int, default=19674)
    ap.add_argument("--long", type=int, default=60000)
    args = ap.parse_args()

    short = run(args.short, 22_200, args.difficulty)
    long = run(args.long, 22_300, args.difficulty)

    for label, data in (("SHORT", short), ("LONG", long)):
        print(
            f"{label} ticks={data['ticks']} income={data['best_income']} "
            f"occupied={data['best_occupied']} army={data['peak_army']} result={data['result']}"
        )
        for row in data["timeline"]:
            print("   ", json.dumps(row))

    (ROOT / ".runtime" / "economy_over_time.json").write_text(
        json.dumps({"short": short, "long": long}, indent=2), encoding="utf-8"
    )
    print(
        "\nVERDICT:",
        "time-starved (income grows with a longer budget)"
        if long["best_income"] > short["best_income"] + 1.0
        else "hard-blocked (income does not grow even with 3x the ticks)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
