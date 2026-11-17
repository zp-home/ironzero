"""Call each stage in isolation to see which one actually produces the order.

``probe_expansion_dispatch`` shows the expansion gate open 286 times but only 21
``build_extractor`` orders dispatched -- the other 265 produced ``build_structure``
from the *factory* stage. That means the map from "gate open" to "order emitted" is
being consumed by an earlier stage in the same dispatch.

This calls every stage individually on the same state, so each stage's own verdict
is visible without the earlier stages having already claimed the builder.

Usage::

    python scripts/probe_stage_isolation.py [--difficulty 1]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=700)
    ap.add_argument("--port", type=int, default=20_800)
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
        game, port=args.port, seed=args.port, episode_prefix="isolation", strategy=RuleStrategy()
    )
    fired: Counter[str] = Counter()
    stage_names_seen: Counter[str] = Counter()
    try:
        env.reset()
        for step in range(args.decisions):
            if step % 40 == 0:
                stage_names_seen[",".join(env.macro.stage_names)] += 1
                own = env.state.get("own_entities") or []
                occupied = sum(
                    1
                    for s in (env.state.get("resource_slots") or [])
                    if isinstance(s, dict) and s.get("occupied")
                )
                funds = round(
                    float((env.state.get("global_state") or {}).get("funds") or 0)
                )
                row = {"step": step, "occupied": occupied, "funds": funds}
                for label, fn in (
                    ("upgrade", env._issue_upgrades),
                    ("extractor", env._issue_opening_extractor),
                    ("factory", env._issue_opening_factory),
                    ("production", env._issue_opening_production),
                    ("defence", env._issue_defence),
                    ("expansion", env._issue_late_expansion),
                    ("attack", env._issue_attack_drive),
                ):
                    try:
                        orders = fn()
                    except Exception as exc:  # noqa: BLE001
                        orders = [{"kind": f"ERROR:{exc}"}]
                    kinds = [str(o.get("kind")) for o in orders if isinstance(o, dict)]
                    if kinds:
                        fired[label] += 1
                    row[label] = kinds
                print(json.dumps(row))
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("\nstage fired counts (of sampled decisions):", json.dumps(dict(fired), indent=2))
    print("stage_names seen:", json.dumps(dict(stage_names_seen), indent=2))
    (ROOT / ".runtime" / "stage_isolation.json").write_text(
        json.dumps({"fired": dict(fired), "stage_names": dict(stage_names_seen)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
