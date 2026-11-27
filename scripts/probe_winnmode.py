"""Test win_mode variants (objective/annihilation/native) at difficulty 1, Lake, 8 seeds.

Hypothesis from 10 rounds: `objective`'s dominance gate (3x army + 3x HP) and
`mainBuildings==0` are unreachable vs difficulty-1 AI (1.4x income + T2 units),
so every episode times out and PPO never sees a victory signal.

This probe uses the best fast-rush config (attack_at=6, 2 factories, 0 turrets)
and reports the outcome distribution under each win_mode. If `annihilation`
(or native) yields a reachable win, that is the correct training signal.
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


def run(mode: str, n: int = 8) -> None:
    wins = 0
    seeds = [71000 + i for i in range(n)]
    for i, seed in enumerate(seeds):
        game = RwppGameConfig(
            map_relative="maps/skirmish/[p2]Lake (2p).tmx",
            map_name="Lake",
            map_width_px=2600.0,
            map_height_px=2600.0,
            episode_ticks=19674,
            enemy_ai_difficulty=1,
            unlock_ai=True,
            win_mode=mode,
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
        env = RwppDuelEnv(game, port=27500 + i, seed=seed, episode_prefix=f"wm_{mode}",
                          strategy=RushStrategy(attack_at=6, max_factories=2, max_extractors=4))
        try:
            env.reset(seed=seed)
            for _ in range(19674 // 10 + 50):
                _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
                if t or tr:
                    res = info.get("result")
                    if res == "victory":
                        wins += 1
                    print(f"[{mode}] ep={i} res={res} tick={env.state.get('tick')}", flush=True)
                    break
        finally:
            env.close()
    print(f"=== {mode}: {wins}/{n} ===\n", flush=True)


def main() -> int:
    for mode in ("objective", "annihilation", ""):
        run(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
