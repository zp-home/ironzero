"""Is the builder alive? A dead builder freezes the whole economy.

``probe_expansion_gate`` reports ``no_builder`` on 209 of 1400 decisions, i.e. for
roughly 15% of the episode the team has **no living builder at all**. That is a
hard stop for every stage that spends the builder: extractor, factory, expansion.

Two ways this happens and they need different fixes:

  A. the single map-placed builder is killed and never replaced -> the economy is
     permanently stuck at whatever it had;
  B. the command centre keeps producing builders but they die before arriving.

This samples the builder count, the command-centre queue and the builder's position
relative to the enemy, so "dead" can be told apart from "out of position".

Usage::

    python scripts/probe_builder_survival.py [--difficulty 1]
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
    ap.add_argument("--decisions", type=int, default=1400)
    ap.add_argument("--port", type=int, default=20_600)
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
        game, port=args.port, seed=args.port, episode_prefix="builder", strategy=RuleStrategy()
    )
    dead_frames = 0
    total = 0
    ever_builders: set = set()
    try:
        env.reset()
        for step in range(args.decisions):
            own = env.state.get("own_entities") or []
            builders = [
                u
                for u in own
                if u.get("is_builder") and float(u.get("hp") or 0.0) > 0.0
            ]
            for b in builders:
                ever_builders.add(b.get("entity_id"))
            total += 1
            if not builders:
                dead_frames += 1
            if step % 200 == 0:
                cc_queue = 0
                for entry in env.state.get("factories") or []:
                    if isinstance(entry, dict) and entry.get("unit_type") == "commandCenter":
                        cc_queue = int(entry.get("queue_length") or 0)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "tick": env.state.get("tick"),
                            "builders": len(builders),
                            "builder_hp": [round(float(b.get("hp") or 0)) for b in builders],
                            "builder_ids": [b.get("entity_id") for b in builders],
                            "cc_queue": cc_queue,
                            "own_total": len(own),
                            "income": (env.state.get("global_state") or {}).get("income"),
                        }
                    )
                )
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print(
        f"\nno-builder decisions: {dead_frames}/{total} "
        f"({dead_frames / max(1, total):.1%}); distinct builder ids seen: {len(ever_builders)}"
    )
    (ROOT / ".runtime" / "builder_survival.json").write_text(
        json.dumps(
            {
                "dead_frames": dead_frames,
                "total": total,
                "ratio": dead_frames / max(1, total),
                "distinct_builders": len(ever_builders),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
