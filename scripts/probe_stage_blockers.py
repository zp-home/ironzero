"""Which curriculum stage actually fires, and what blocks the others?

The raised-ceiling run still shows ``army = 0`` at ``funds = 4063`` with
``extractors = 1``, i.e. a rich base that fields no units. Seven stages each have
their own gates, and a stage that returns ``[]`` is indistinguishable from a stage
that was never reached -- which is what makes this a diagnosis problem rather than
a tuning problem.

This drives a noop policy (so the curriculum is the only actor) and, on every
sampled decision, reports for each stage: fired / blocked-by-name. It prints the
**first** blocking reason of each stage in source order.

Usage::

    python scripts/probe_stage_blockers.py [--difficulty 1] [--decisions 900]
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


def _combat_units(state: dict) -> list[dict]:
    return [
        u
        for u in (state.get("own_entities") or [])
        if u.get("can_attack")
        and not u.get("is_builder")
        and not u.get("is_building")
        and float(u.get("hp") or 0.0) > 0.0
    ]


def _combat_factories(state: dict) -> list[dict]:
    return [
        e
        for e in (state.get("factories") or [])
        if isinstance(e, dict)
        and any(str(n).lower() != "builder" for n in (e.get("can_produce") or []))
    ]


def diagnose(env: RwppDuelEnv) -> dict:
    state = env.state
    own = state.get("own_entities") or []
    army = len(_combat_units(state))
    facs = _combat_factories(state)
    slots = state.get("resource_slots") or []
    free = [i for i, s in enumerate(slots) if isinstance(s, dict) and not s.get("occupied")]
    funds = float((state.get("global_state") or {}).get("funds") or 0.0)
    queue = sum(int(f.get("queue_length") or 0) for f in facs)
    unfinished = [u for u in own if u.get("is_building") and not _is_operational(u)]
    opts = produce_options(state)

    out: dict[str, str] = {}
    # production
    if not getattr(env.game, "opening_production", False):
        out["production"] = "flag_off"
    elif not env.macro.allows("production"):
        out["production"] = "macro_disallows"
    elif not facs:
        out["production"] = "no_combat_factory"
    elif army >= int(env.macro.army_cap):
        out["production"] = "at_cap"
    elif queue >= 2:
        out["production"] = "queue_full"
    elif not opts:
        out["production"] = "no_produce_option"
    else:
        out["production"] = "FIRED"

    # expansion
    if not getattr(env.game, "opening_extractor", False):
        out["expansion"] = "flag_off"
    elif not env.macro.allows("expansion"):
        out["expansion"] = "macro_disallows"
    elif not facs:
        out["expansion"] = "no_combat_factory"
    elif not free:
        out["expansion"] = "no_free_pool"
    elif unfinished:
        out["expansion"] = "builder_busy"
    else:
        out["expansion"] = "FIRED"

    # attack
    if not getattr(env.game, "opening_attack", False):
        out["attack"] = "flag_off"
    elif not env.macro.allows("attack"):
        out["attack"] = "macro_disallows"
    elif army < int(env.macro.attack_at_army):
        out["attack"] = f"army<{env.macro.attack_at_army}"
    else:
        out["attack"] = "FIRED"

    return {
        "tick": state.get("tick"),
        "army": army,
        "cap": int(env.macro.army_cap),
        "target": int(env.macro.army_target),
        "funds": round(funds, 1),
        "income": (state.get("global_state") or {}).get("income"),
        "extractors": sum(1 for u in own if "xtractor" in str(u.get("unit_type") or "")),
        "factories": len(facs),
        "free_pools": len(free),
        "queue": queue,
        "stages": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=900)
    ap.add_argument("--port", type=int, default=19_200)
    ap.add_argument("--army-cap", type=int, default=32)
    ap.add_argument("--army-target", type=int, default=20)
    ap.add_argument("--max-extractors", type=int, default=9)
    ap.add_argument("--max-factories", type=int, default=3)
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
        economy_army_cap=args.army_cap,
        economy_army_target=args.army_target,
        economy_max_extractors=args.max_extractors,
        economy_max_factories=args.max_factories,
        economy_production_queue_depth=3,
    )
    from ironzero.strategy.rule_strategy import RuleStrategy

    env = RwppDuelEnv(
        game, port=args.port, seed=98_000, episode_prefix="blockers", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    try:
        env.reset()
        for step in range(args.decisions):
            if step % 50 == 0:
                row = diagnose(env)
                rows.append(row)
                for stage, verdict in row["stages"].items():
                    if verdict != "FIRED":
                        counts[f"{stage}:{verdict}"] += 1
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print("\nblocker histogram:", json.dumps(dict(counts.most_common()), indent=2))
    (ROOT / ".runtime" / "stage_blockers.json").write_text(
        json.dumps({"rows": rows, "counts": dict(counts)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
