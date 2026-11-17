"""Are the production orders being accepted, and if not, why?

``probe_production_plateau`` reports ``ORDERED`` on 1157 of 1400 decisions while
``ever_owned`` is only 16 and the live army peaks at 6, with ``queue = 0`` at every
sample. An order that is dispatched but leaves the queue empty was **refused by the
engine** -- the queue only grows for accepted orders.

This counts receipts by (kind, accepted/rejected_reason) so the refusal reason is
named instead of inferred.

Usage::

    python scripts/probe_order_receipts.py [--decisions 1000]
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
    ap.add_argument("--decisions", type=int, default=1000)
    ap.add_argument("--port", type=int, default=20_300)
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
        game, port=args.port, seed=args.port, episode_prefix="receipts", strategy=RuleStrategy()
    )
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    samples: dict[str, dict] = {}
    result = None
    try:
        env.reset()
        for _ in range(args.decisions):
            before = len(env.receipts)
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            for receipt in env.receipts[before:]:
                if not isinstance(receipt, dict):
                    continue
                kind = str(receipt.get("kind"))
                if receipt.get("accepted", True):
                    accepted[kind] += 1
                    continue
                reason = str(receipt.get("rejected_reason"))
                key = f"{kind}:{reason}"
                rejected[key] += 1
                if key not in samples:
                    samples[key] = {
                        "tick": env.state.get("tick"),
                        "funds": (env.state.get("global_state") or {}).get("funds"),
                        "army": sum(
                            1
                            for u in (env.state.get("own_entities") or [])
                            if u.get("can_attack")
                            and not u.get("is_builder")
                            and not u.get("is_building")
                            and float(u.get("hp") or 0.0) > 0.0
                        ),
                        "receipt": receipt,
                    }
            if term or trunc:
                result = info.get("result")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("ACCEPTED:", json.dumps(dict(accepted.most_common()), indent=2))
    print("REJECTED:", json.dumps(dict(rejected.most_common()), indent=2))
    print("FIRST OCCURRENCE:", json.dumps(samples, indent=2, default=str)[:1600])
    print(f"result={result}")
    (ROOT / ".runtime" / "order_receipts.json").write_text(
        json.dumps(
            {"accepted": dict(accepted), "rejected": dict(rejected), "samples": samples},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
