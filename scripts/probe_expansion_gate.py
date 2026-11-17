"""Which gate in the expansion stage refuses, decision by decision?

The army's cumulative production (16 units) matches what ``income = 26`` can fund
(26 credits/s over ~200 s / 350 per tank), so production is *not* the limiter --
income is. And income is pinned at 26 (base 18 + one extractor) while the map has
nine pools.

So the question is precise: which of the expansion stage's gates returns ``[]``, and
how often. The gates, in source order:

  1. ``opening_extractor`` flag
  2. ``macro.allows("expansion")`` -- the rule layer's plan
  3. a combat factory must exist
  4. a free pool must exist
  5. no structure may be mid-construction
  6. a living builder must exist
  7. ``funds >= extractor price``
  8. ``occupied < max_extractors``
  9. builder-not-stalled rotation gates

This prints the first gate that refuses, per sampled decision.

Usage::

    python scripts/probe_expansion_gate.py [--difficulty 1]
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
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def gate(env: RwppDuelEnv) -> tuple[str, dict]:
    state = env.state
    own = state.get("own_entities") or []
    slots = state.get("resource_slots") or []
    free = [i for i, s in enumerate(slots) if isinstance(s, dict) and not s.get("occupied")]
    occupied = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
    facs = [
        e
        for e in (state.get("factories") or [])
        if isinstance(e, dict)
        and any(str(n).lower() != "builder" for n in (e.get("can_produce") or []))
    ]
    funds = float((state.get("global_state") or {}).get("funds") or 0.0)
    unfinished = [
        u
        for u in own
        if u.get("is_building") and not _is_operational(u)
    ]
    info = {
        "occupied": occupied,
        "max_extr": int(env.macro.max_extractors),
        "free": len(free),
        "facs": len(facs),
        "funds": round(funds),
        "unfinished": len(unfinished),
        "income": (state.get("global_state") or {}).get("income"),
        "stages": list(env.macro.stage_names),
    }
    if "expansion" not in env.macro.stage_names:
        return "stage_not_in_plan", info
    if not getattr(env.game, "opening_extractor", False):
        return "flag_off", info
    if not env.macro.allows("expansion"):
        return "macro_disallows", info
    if not facs:
        return "no_combat_factory", info
    if not free:
        return "no_free_pool", info
    if unfinished:
        return "structure_under_construction", info
    if not any(
        u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0 for u in own
    ):
        return "no_builder", info
    if funds < 700.0:
        return "funds_below_extractor_price", info
    if occupied >= int(env.macro.max_extractors):
        return "at_max_extractors", info
    return "SHOULD_ORDER", info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=1400)
    ap.add_argument("--port", type=int, default=20_500)
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
        game, port=args.port, seed=args.port, episode_prefix="expgate", strategy=RuleStrategy()
    )
    counts: Counter[str] = Counter()
    try:
        env.reset()
        for step in range(args.decisions):
            verdict, info = gate(env)
            counts[verdict] += 1
            if step % 200 == 0:
                print(json.dumps({"step": step, "gate": verdict, **info}))
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("\nexpansion gate histogram:", json.dumps(dict(counts.most_common()), indent=2))
    (ROOT / ".runtime" / "expansion_gate.json").write_text(
        json.dumps(dict(counts), indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
