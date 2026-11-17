"""When expansion says SHOULD_ORDER, does the order actually build an extractor?

``probe_expansion_gate`` shows ``SHOULD_ORDER`` on 283 of 1400 decisions while the
occupied-pool count never leaves 1. Either the order is dispatched and cancelled, or
it is never dispatched at all.

This pairs the gate verdict with the *actual* dispatch: for every decision where the
gate says SHOULD_ORDER, it records what ``_curriculum_actions`` returned, so
"the gate opened but the stage still did not fire" is separable from "the stage
fired and the order was lost".

Usage::

    python scripts/probe_expansion_dispatch.py [--difficulty 1]
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


def gate_open(env: RwppDuelEnv) -> bool:
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
    unfinished = [u for u in own if u.get("is_building") and not _is_operational(u)]
    return (
        "expansion" in env.macro.stage_names
        and getattr(env.game, "opening_extractor", False)
        and env.macro.allows("expansion")
        and bool(facs)
        and bool(free)
        and not unfinished
        and any(u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0 for u in own)
        and funds >= 700.0
        and occupied < int(env.macro.max_extractors)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=1400)
    ap.add_argument("--port", type=int, default=20_700)
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
        game, port=args.port, seed=args.port, episode_prefix="expdisp", strategy=RuleStrategy()
    )
    stats: Counter[str] = Counter()
    samples: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            opened = gate_open(env)
            # ``_curriculum_actions`` mutates latches, so only call it when the gate
            # is open -- otherwise this probe would perturb the run it measures.
            if opened:
                orders = env._curriculum_actions()
                kinds = [str(o.get("kind")) for o in orders if isinstance(o, dict)]
                if "build_extractor" in kinds:
                    stats["gate_open_and_ordered"] += 1
                else:
                    stats["gate_open_but_no_extractor_order"] += 1
                    if len(samples) < 12:
                        samples.append(
                            {
                                "step": step,
                                "kinds": kinds,
                                "stage_names": list(env.macro.stage_names),
                                "funds": round(
                                    float(
                                        (env.state.get("global_state") or {}).get("funds") or 0
                                    )
                                ),
                            }
                        )
            else:
                stats["gate_closed"] += 1
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("stats:", json.dumps(dict(stats), indent=2))
    print("gate-open-but-not-ordered samples:", json.dumps(samples, indent=2))
    (ROOT / ".runtime" / "expansion_dispatch.json").write_text(
        json.dumps({"stats": dict(stats), "samples": samples}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
