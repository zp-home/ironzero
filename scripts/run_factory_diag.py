"""Run one episode with --debug-build and capture the bridge's factory diagnostics.

The bridge now prints ``[ironzero] factory diag ...`` each time it reports a
factory. This runs one episode with the curriculum (noop policy) and one with the
built-in AI, with --debug-build on, and prints the last distinct diagnostic lines
per factory. That tells us whether our factories are actually operable.

Usage::

    python scripts/run_factory_diag.py [--decisions 400]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def run(ai_pilot: bool, port: int, decisions: int) -> None:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        debug_build=True,
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
        game, port=port, seed=port, episode_prefix="fdiag",
        strategy=None if ai_pilot else RuleStrategy(),
    )
    try:
        env.reset()
        for _ in range(decisions):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    log = ROOT / "runs" / "_rwpp_ppo" / f"game-{port}-*.log"
    files = sorted(ROOT.joinpath("runs", "_rwpp_ppo").glob(f"game-{port}-*.log"))
    if not files:
        print(f"no log for port {port}")
        return
    lines = [
        line.strip()
        for line in files[-1].read_text(encoding="utf-8", errors="replace").splitlines()
        if "factory diag" in line
    ]
    # Keep the last distinct line per factory id.
    last: dict[str, str] = {}
    for line in lines:
        fid = line.split("id=")[1].split()[0] if "id=" in line else "?"
        last[fid] = line
    print(f"\n=== {'AI PILOT' if ai_pilot else 'OURS'} (port {port}) ===")
    for line in last.values():
        print("  ", line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=400)
    args = ap.parse_args()
    run(False, 24_100, args.decisions)
    run(True, 24_200, args.decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
