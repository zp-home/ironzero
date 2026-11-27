"""Does the built-in AI ever produce combat units at a landFactory on this map?

``probe_ai_queue_vs_ours`` showed the AI's command-centre builder queue advancing
normally while ours sits at -1.0, but the samples were dominated by the command
centre. The decisive comparison is narrower: when the AI drives our team, does a
``landFactory`` ever queue a combat unit, and does its progress advance?

If the AI also never produces at landFactory, the map or engine forbids it and the
curriculum should not try. If the AI does, compare the unit type and the action it
used with ours -- that is the last difference left.

Usage::

    python scripts/probe_ai_factory_units.py [--decisions 1600]
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


def run(ai_pilot: bool, port: int, decisions: int) -> dict:
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
        ai_pilot=ai_pilot,
    )
    env = RwppDuelEnv(
        game, port=port, seed=port, episode_prefix="aifac",
        strategy=None if ai_pilot else RuleStrategy(),
    )
    by_factory: Counter = Counter()
    combat_progress: Counter = Counter()
    samples: list[dict] = []
    peak_mobile = 0
    try:
        env.reset()
        for step in range(decisions):
            own = env.state.get("own_entities") or []
            peak_mobile = max(
                peak_mobile,
                sum(
                    1 for u in own
                    if u.get("can_attack") and not u.get("is_builder")
                    and not u.get("is_building") and float(u.get("hp") or 0.0) > 0.0
                ),
            )
            for entry in env.state.get("factories") or []:
                if not isinstance(entry, dict):
                    continue
                ftype = str(entry.get("unit_type") or "?")
                for item in entry.get("queued") or []:
                    unit = str(item.get("unit_type") or "?")
                    by_factory[f"{ftype}:{unit}"] += 1
                    if ftype.lower() == "landfactory":
                        combat_progress[str(item.get("progress"))] += 1
                        if len(samples) < 8:
                            samples.append({
                                "step": step,
                                "tick": env.state.get("tick"),
                                "factory_id": entry.get("entity_id"),
                                "unit": unit,
                                "progress": item.get("progress"),
                                "menu": entry.get("can_produce"),
                            })
            env.step(np.array([9, 0, 0, 0, 0]))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "ai_pilot": ai_pilot,
        "peak_mobile": peak_mobile,
        "by_factory": dict(by_factory.most_common(12)),
        "landfactory_progress": dict(combat_progress.most_common(12)),
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1600)
    args = ap.parse_args()

    ours = run(False, 23_900, args.decisions)
    ai = run(True, 24_000, args.decisions)
    for label, data in (("OURS", ours), ("AI PILOT", ai)):
        print(f"\n=== {label} ===")
        print(f"peak mobile: {data['peak_mobile']}")
        print("queued by factory:unit:", json.dumps(data["by_factory"], indent=2))
        print("landFactory queue progress:", json.dumps(data["landfactory_progress"], indent=2))
        for row in data["samples"]:
            print("  ", json.dumps(row))
    (ROOT / ".runtime" / "ai_factory_units.json").write_text(
        json.dumps({"ours": ours, "ai": ai}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
