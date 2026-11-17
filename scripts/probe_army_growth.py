"""Does the army actually grow to the target, or does production stall at ~7?

``probe_stage_blockers`` showed ``army`` reaching 6-7 with all three stages FIRED,
which is progress (it used to be 0) but far below the ``army_target`` of 20. Two
possible reasons, and they need different fixes:

  A. production throughput is too low -- the factories can only deliver so many
     units per tick, so the army plateaus below the target naturally;
  B. the army is being spent as fast as it is produced -- units march into the
     enemy and die, which keeps the *live* count flat while production is fine.

The distinguishing evidence is the cumulative count of units ever created vs the
live count, plus where the losses happen (near our base or near theirs).

Usage::

    python scripts/probe_army_growth.py [--difficulty 1] [--decisions 1500]
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
    ap.add_argument("--decisions", type=int, default=1500)
    ap.add_argument("--port", type=int, default=19_700)
    ap.add_argument("--seed", type=int, default=0)
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
        game,
        port=args.port,
        seed=args.seed or args.port,
        episode_prefix="growth",
        strategy=RuleStrategy(),
    )
    seen_ids: set = set()
    rows: list[dict] = []
    result = None
    try:
        env.reset()
        for step in range(args.decisions):
            own = env.state.get("own_entities") or []
            for unit in own:
                uid = unit.get("entity_id")
                if uid is not None:
                    seen_ids.add(uid)
            if step % 100 == 0:
                combat = [
                    u
                    for u in own
                    if u.get("can_attack")
                    and not u.get("is_builder")
                    and not u.get("is_building")
                    and float(u.get("hp") or 0.0) > 0.0
                ]
                g = env.state.get("global_state") or {}
                facs = [
                    f
                    for f in (env.state.get("factories") or [])
                    if isinstance(f, dict)
                    and any(str(n).lower() != "builder" for n in (f.get("can_produce") or []))
                ]
                rows.append(
                    {
                        "step": step,
                        "tick": env.state.get("tick"),
                        "army": len(combat),
                        "ever_owned": len(seen_ids),
                        "extractors": sum(
                            1 for u in own if "xtractor" in str(u.get("unit_type") or "")
                        ),
                        "factories": len(facs),
                        "queue": sum(int(f.get("queue_length") or 0) for f in facs),
                        "income": g.get("income"),
                        "funds": round(float(g.get("funds") or 0.0)),
                        "enemy_alive": len(env.state.get("visible_enemy_entities") or []),
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

    for row in rows:
        print(json.dumps(row))
    print(f"\nresult={result} ever_owned={len(seen_ids)}")
    (ROOT / ".runtime" / "army_growth.json").write_text(
        json.dumps({"rows": rows, "result": result, "ever_owned": len(seen_ids)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
