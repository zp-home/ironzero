"""Capture bridge game-over logs for seed 71001 (victory) and 71000 (timeout).

Goal: determine whether victory came from destroying the enemy commandCenter
(native objective) or from the dominance rule, and how close the timeout game
came to either condition.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def trace(seed: int, port: int) -> None:
    print(f"\n=== seed={seed} ===", flush=True)
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=24,
        economy_max_extractors=4,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=350.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="vdiag",
                      strategy=RushStrategy(attack_at=6, max_factories=2, max_extractors=4))
    try:
        env.reset(seed=seed)
        for step in range(2000):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                print(f"RESULT={info.get('result')} at tick={env.state.get('tick')}", flush=True)
                break
    finally:
        env.close()
    log = ROOT / "runs" / "_rwpp_ppo" / f"game-{port}-*.log"
    files = sorted((ROOT / "runs" / "_rwpp_ppo").glob(f"game-{port}-*.log"))
    if files:
        content = files[-1].read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            if "game over" in line or "victory" in line or "defeat" in line:
                print("BRIDGE:", line, flush=True)


def main() -> int:
    trace(71001, 27301)
    trace(71000, 27300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
