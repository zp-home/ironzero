"""Dump all buildable/upgradeable factory menus & unit capabilities in RWPP 1.15."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.observation.menus import build_menu
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=12000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=16,
        units_per_team=16,
    )
    env = RwppDuelEnv(game, port=26910, seed=1, episode_prefix="tree")
    env._curriculum_disabled = True
    try:
        env.reset(seed=1)
        menu = build_menu(env.obs_state)
        print("MENU:", menu, flush=True)
        for stype, gx, gy, steps in [
            ("airFactory", 2, 6, 90),
            ("landFactory", 3, 6, 90),
        ]:
            idx = menu.index(stype)
            env.step(np.array([ACTION_KINDS.index("build_structure"), 8, gx, gy, idx]))
            for _ in range(steps):
                env.step(np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0]))

        print("\n=== T1 FACTORIES ===")
        for f in env.state.get("factories") or []:
            print("factory:", f, flush=True)
        for u in env.state.get("own_entities") or []:
            if u.get("is_building"):
                print("own_bldg:", {k: u.get(k) for k in ("entity_id", "unit_type", "tech_level", "can_upgrade", "upgrade_cost", "is_operational")}, flush=True)

        # Upgrade any upgradable building using target slots mask!
        for _attempt in range(2):
            masks = env.action_masks()
            tmask = masks["target"]
            upgraded_any = False
            for t_idx in range(len(tmask)):
                if tmask[t_idx]:
                    # Try upgrade_building at target t_idx if kind mask allows
                    if masks["kind"][ACTION_KINDS.index("upgrade_building")]:
                        env.step(np.array([ACTION_KINDS.index("upgrade_building"), 0, 0, 0, t_idx]))
                        upgraded_any = True
                        break
            print("upgrade attempt:", upgraded_any, "kind_mask:", bool(masks["kind"][ACTION_KINDS.index("upgrade_building")]), flush=True)
            for _ in range(150):
                env.step(np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0]))

        print("\n=== AFTER UPGRADES ===")
        for f in env.state.get("factories") or []:
            print("factory:", f, flush=True)
        for u in env.state.get("own_entities") or []:
            if u.get("is_building"):
                print("own_bldg:", {k: u.get(k) for k in ("entity_id", "unit_type", "tech_level", "can_upgrade", "upgrade_cost", "is_operational")}, flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
