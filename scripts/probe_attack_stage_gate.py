"""Does the attack stage ever fire when the army is big enough?

``probe_attack_target`` showed ``curriculum: null`` for 1500 decisions while the
army grew 0 -> 15 and stayed 1250+ px from any enemy building. The rule layer sets
``attack_at_army = 15``, so the gate should open exactly at the end. This calls the
stage directly at high army counts, reporting each internal gate, to find which one
keeps the march from firing.

Usage::

    python scripts/probe_attack_stage_gate.py [--decisions 1600]
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
    ap.add_argument("--decisions", type=int, default=1600)
    ap.add_argument("--port", type=int, default=25_000)
    args = ap.parse_args()

    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
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
        game, port=args.port, seed=args.port, episode_prefix="atkgate", strategy=RuleStrategy()
    )
    verdicts: Counter = Counter()
    detail: dict[str, dict] = {}
    try:
        env.reset()
        for step in range(args.decisions):
            if step % 20 == 0:
                ours = [
                    u for u in (env.state.get("own_entities") or [])
                    if u.get("can_attack") and not u.get("is_builder")
                    and not u.get("is_building") and float(u.get("hp") or 0) > 0
                ]
                attack_at = int(env.macro.attack_at_army)
                allows = env.macro.allows("attack")
                verdict = None
                if not getattr(game, "opening_attack", False):
                    verdict = "flag_off"
                elif not allows:
                    verdict = "macro_disallows"
                elif len(ours) < attack_at:
                    verdict = f"army<{attack_at}"
                else:
                    verdict = "gates_open"
                verdicts[verdict] += 1
                if verdict not in detail:
                    detail[verdict] = {
                        "step": step,
                        "army": len(ours),
                        "attack_at": attack_at,
                        "allows": allows,
                        "stage_names": list(env.macro.stage_names),
                        "visible_foes": len(env.state.get("visible_enemy_entities") or []),
                    }
                # Call the stage itself; it may return [] for other reasons.
                orders = env._issue_attack_drive()
                kinds = [str(o.get("kind")) for o in orders if isinstance(o, dict)]
                if kinds:
                    verdicts["STAGE_FIRED"] += 1
                    if "STAGE_FIRED" not in detail:
                        detail["STAGE_FIRED"] = {
                            "step": step, "army": len(ours), "kinds": kinds,
                        }
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("attack gate verdicts:", json.dumps(dict(verdicts.most_common()), indent=2))
    print("first occurrences:", json.dumps(detail, indent=2, ensure_ascii=False))
    (ROOT / ".runtime" / "attack_stage_gate.json").write_text(
        json.dumps({"verdicts": dict(verdicts), "detail": detail}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
