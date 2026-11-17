"""Why does the army plateau at exactly 6 units, independent of AI difficulty?

The measurement that makes this a puzzle rather than a tuning problem: with the same
economy ceilings (army_cap 32, army_target 20), our peak army is **6 at every
difficulty** (-2, -1, 0, 1) while the enemy peaks at 9/17/15/19. Our number does not
move when the opponent does, so the limit is internal -- production is not keeping
up, or is being blocked by something that has nothing to do with the fight.

Two candidates:

  A. the factories sit idle (production stage refuses for a reason that persists);
  B. the units are produced but die immediately, so the live count is a loss rate
     rather than a production rate.

The distinguishing evidence is ``ever_owned`` (cumulative distinct entity ids) vs
``army`` (live combat units), sampled densely, together with the per-decision
production verdict.

Usage::

    python scripts/probe_production_plateau.py [--difficulty 1]
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
from ironzero.rwpp.env import RwppDuelEnv, _is_operational  # noqa: E402
from ironzero.observation.menus import produce_options  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def production_verdict(env: RwppDuelEnv) -> tuple[str, dict]:
    state = env.state
    own = state.get("own_entities") or []
    facs = [
        e
        for e in (state.get("factories") or [])
        if isinstance(e, dict)
        and any(str(n).lower() != "builder" for n in (e.get("can_produce") or []))
    ]
    army = sum(
        1
        for u in own
        if u.get("can_attack")
        and not u.get("is_builder")
        and not u.get("is_building")
        and float(u.get("hp") or 0.0) > 0.0
    )
    funds = float((state.get("global_state") or {}).get("funds") or 0.0)
    queue = sum(int(f.get("queue_length") or 0) for f in facs)
    opts = produce_options(state)
    info = {
        "army": army,
        "cap": int(env.macro.army_cap),
        "facs": len(facs),
        "queue": queue,
        "funds": round(funds),
        "opts": len(opts),
        "income": (state.get("global_state") or {}).get("income"),
    }
    if not env.macro.allows("production"):
        return "macro_disallows", info
    if not facs:
        return "no_combat_factory", info
    if army >= int(env.macro.army_cap):
        return "at_cap", info
    if queue >= int(env.macro.production_queue_depth):
        return "queue_full", info
    if not opts:
        return "no_produce_option", info
    if funds < 350:
        return "cannot_afford_cheapest", info
    return "ORDERED", info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=1400)
    ap.add_argument("--port", type=int, default=20_200)
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
        game, port=args.port, seed=20_200, episode_prefix="plateau", strategy=RuleStrategy()
    )
    verdicts: Counter[str] = Counter()
    ever: set = set()
    result = None
    try:
        env.reset()
        for step in range(args.decisions):
            for u in env.state.get("own_entities") or []:
                ever.add(u.get("entity_id"))
            verdict, info = production_verdict(env)
            verdicts[verdict] += 1
            if step % 200 == 0:
                print(
                    json.dumps(
                        {"step": step, "verdict": verdict, "ever": len(ever), **info}
                    )
                )
            _o, _r, term, trunc, info2 = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                result = info2.get("result")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("\nproduction verdicts:", json.dumps(dict(verdicts.most_common()), indent=2))
    print(f"result={result} ever_owned={len(ever)}")
    (ROOT / ".runtime" / "production_plateau.json").write_text(
        json.dumps({"verdicts": dict(verdicts), "ever_owned": len(ever), "result": result}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
