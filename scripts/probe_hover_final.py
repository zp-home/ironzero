"""Final decisive test: hoverTank (ONLY land unit that can attack air, amphibious)
with concentrated attack on enemy commandCenter, at difficulty 1.

Prior hover probe failed because it used grid-target attack at cadence and only
22 tanks peaked at 9. Here we use exact target coords, attack_at=4 (earlier),
3 land factories, and check if the air-capable hoverTank breaks the stalemate
where c_tank (no AA) cannot touch the 6-10 enemy lightGunships.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from collections import Counter


# Force hoverTank as the preferred combat unit (can_attack_air=True, amphibious)
env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
env_mod._UNIT_PRICE["hoverTank"] = 400


def run(seed: int, port: int) -> None:
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
        economy_max_factories=3,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=400.0,
        economy_production_queue_depth=4,
    )
    from scripts.probe_lake_winrate import RushStrategy
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="hfinal",
                      strategy=RushStrategy(attack_at=4, max_factories=3, max_extractors=4, reserve=400.0))
    try:
        env.reset(seed=seed)
        peak = 0
        for _ in range(19674 // 10 + 50):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = sum(1 for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building"))
            peak = max(peak, com)
            if t or tr:
                break
        end_own = Counter(str(u.get("unit_type")) for u in (env.state.get("own_entities") or []))
        end_foe = Counter(str(u.get("unit_type")) for u in (env.state.get("visible_enemy_entities") or []))
        print(f"seed={seed} res={info.get('result')} peak={peak} own={dict(end_own)} foe={dict(end_foe)}", flush=True)
    finally:
        env.close()


def main() -> int:
    wins = 0
    for i in range(6):
        seed = 71000 + i
        run(seed, 27700 + i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
