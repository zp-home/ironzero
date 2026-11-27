"""Read actual engine-reported prices for every combat unit on Lake."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=3000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=8,
        units_per_team=8,
    )
    env = RwppDuelEnv(game, port=27100, seed=1, episode_prefix="price")
    try:
        env.reset(seed=1)
        for step in range(300):
            env.step(np.array([9, 0, 0, 0, 0]))
            for f in env.state.get("factories") or []:
                if str(f.get("unit_type") or "").lower() == "landfactory" and f.get("is_built"):
                    menu = f.get("can_produce") or []
                    prices = f.get("can_produce_price") or []
                    print(f"tick={env.state.get('tick')} landFactory menu/prices:", flush=True)
                    for name, price in zip(menu, prices):
                        print(f"  {name:16s} {price:>6} cr", flush=True)
                    return 0
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
