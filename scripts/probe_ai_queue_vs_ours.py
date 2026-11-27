"""What does a *working* factory queue look like under the built-in AI?

Our curriculum's produce_unit orders are accepted, the item appears in the queue,
and its progress stays at -1.0 forever. To tell whether -1.0 means "never started"
or is just how this bridge reports a normal queued item, run the exact same map with
``--ai-pilot``: the built-in AI is known to field 196 units on this map, so its
queues are by definition functional.

The probe samples every factory's ``queued`` array from both modes side by side:

  * curriculum (our code), noop policy
  * AI pilot (engine drives our team)

If the AI's queue items also show -1.0 but units still appear, the progress field is
a red herring and the real bug is elsewhere (e.g. our order being cancelled). If the
AI's items show advancing progress, the difference is in how the order was issued.

Usage::

    python scripts/probe_ai_queue_vs_ours.py [--decisions 1200]
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


def sample(game: RwppGameConfig, port: int, ai_pilot: bool, decisions: int) -> dict:
    game.ai_pilot = ai_pilot
    strategy = None if ai_pilot else RuleStrategy()
    env = RwppDuelEnv(
        game, port=port, seed=port, episode_prefix="aivso" if ai_pilot else "ours",
        strategy=strategy,
    )
    progress_values: Counter[str] = Counter()
    queue_samples: list[dict] = []
    ever_mobile = 0
    try:
        env.reset()
        for step in range(decisions):
            own = env.state.get("own_entities") or []
            mobile = sum(
                1 for u in own
                if u.get("can_attack") and not u.get("is_builder")
                and not u.get("is_building") and float(u.get("hp") or 0.0) > 0.0
            )
            ever_mobile = max(ever_mobile, mobile)
            for entry in env.state.get("factories") or []:
                if not isinstance(entry, dict):
                    continue
                for item in entry.get("queued") or []:
                    progress = item.get("progress")
                    progress_values[str(progress)] += 1
                    if len(queue_samples) < 6:
                        queue_samples.append({
                            "step": step,
                            "tick": env.state.get("tick"),
                            "factory": entry.get("entity_id"),
                            "unit": item.get("unit_type"),
                            "progress": progress,
                            "count": item.get("count"),
                        })
            env.step(np.array([9, 0, 0, 0, 0]))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "ai_pilot": ai_pilot,
        "peak_mobile": ever_mobile,
        "progress_values": dict(progress_values.most_common()),
        "queue_samples": queue_samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1200)
    args = ap.parse_args()

    base = RwppGameConfig(
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
    ours = sample(base, 23_700, False, args.decisions)
    ai = sample(base, 23_800, True, args.decisions)

    for label, data in (("OURS", ours), ("AI PILOT", ai)):
        print(f"\n=== {label} ===")
        print(f"peak mobile units: {data['peak_mobile']}")
        print("queue progress histogram:", json.dumps(data["progress_values"], indent=2))
        for row in data["queue_samples"]:
            print("  ", json.dumps(row))

    (ROOT / ".runtime" / "ai_queue_vs_ours.json").write_text(
        json.dumps({"ours": ours, "ai": ai}, indent=2), encoding="utf-8"
    )
    print(
        "\nVERDICT:",
        "progress=-1.0 is NORMAL (AI shows it too)"
        if "-1.0" in ai["progress_values"] and ai["peak_mobile"] > 0
        else "progress=-1.0 is the bug (AI shows advancing progress)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
