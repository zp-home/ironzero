"""Trace which unit the curriculum actually picks, and why hoverTank loses."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories, _unit_value_per_credit


env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
env_mod._UNIT_PRICE["hoverTank"] = 400
from scripts.probe_lake_winrate import RushStrategy


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=4000,
        enemy_ai_difficulty=-2,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=24,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=350.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=27800, seed=1, episode_prefix="pick",
                      strategy=RushStrategy(attack_at=6, max_factories=2, max_extractors=4))
    try:
        env.reset(seed=1)
        printed = 0
        for step in range(600):
            co = getattr(env, "curriculum_order", None)
            if co and co.get("kind") == "produce_unit" and printed < 8:
                print(f"step={step} PICK: {co}", flush=True)
                cf = _combat_factories(env.state)
                for f in cf:
                    menu = f.get("can_produce") or []
                    print(f"  factory id={f.get('entity_id')} built={f.get('is_built')} menu={menu}", flush=True)
                    for name in menu:
                        print(f"    {name}: score={_unit_value_per_credit(name):.4f} price={env_mod._UNIT_PRICE.get(name, 500)}", flush=True)
                printed += 1
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
