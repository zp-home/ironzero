"""Safely dump T2 factory menus / unit capabilities with a weak (-2) enemy.

Difficulty-1 AI fields mechFactory -> mechArtillery/amphibiousJet and
heavyInterceptor (shield 450) vs our T1-only curriculum. Question: can an
upgraded T2 factory produce units that counter those?
"""
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


def wait(env: RwppDuelEnv, n: int) -> bool:
    noop = np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0], dtype=np.int64)
    for _ in range(n):
        _, _, t, tr, _ = env.step(noop)
        if t or tr:
            return True
    return False


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=12000,
        enemy_ai_difficulty=-2,  # weak enemy so we survive to inspect tech
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=16,
        units_per_team=16,
    )
    env = RwppDuelEnv(game, port=27400, seed=1, episode_prefix="t2")
    env._curriculum_disabled = True
    try:
        env.reset(seed=1)
        # Build land + air factories
        for stype, gx, gy in [("airFactory", 2, 6), ("landFactory", 4, 6)]:
            menu = build_menu(env.obs_state)
            idx = menu.index(stype)
            env.step(np.array([ACTION_KINDS.index("build_structure"), 8, gx, gy, idx], dtype=np.int64))
            wait(env, 90)

        print("=== BEFORE UPGRADE ===", flush=True)
        for f in env.state.get("factories") or []:
            print("FACTORY:", {k: f.get(k) for k in ("entity_id", "unit_type", "tech_level", "is_built", "can_produce", "can_produce_price", "can_upgrade")}, flush=True)

        # Upgrade each factory via direct order, waiting for funds
        for _ in range(3):
            ups = [u for u in (env.state.get("own_entities") or [])
                   if u.get("is_building") and u.get("can_upgrade") and u.get("is_operational")
                   and ("factory" in str(u.get("unit_type", "")).lower() or "extractor" in str(u.get("unit_type", "")).lower())]
            if not ups:
                break
            ups.sort(key=lambda u: (0 if "factory" in str(u.get("unit_type", "")).lower() else 1))
            t = ups[0]
            bid = int(t["entity_id"])
            for _ in range(300):
                f = float(t.get("upgrade_cost") or 1500.0)
                if float((env.state.get("global_state") or {}).get("funds") or 0) >= f:
                    break
                if wait(env, 5):
                    return 0
            print(f"UPGRADING {t.get('unit_type')} id={bid}", flush=True)
            env._send_direct_order({"kind": "upgrade_building", "unit_group": 0, "building_entity_id": bid}, {"kind": "upgrade_building"})
            wait(env, 200)

        print("\n=== AFTER UPGRADE ===", flush=True)
        for f in env.state.get("factories") or []:
            print("FACTORY:", {k: f.get(k) for k in ("entity_id", "unit_type", "tech_level", "is_built", "can_produce", "can_produce_price", "can_upgrade")}, flush=True)
        for u in env.state.get("own_entities") or []:
            if u.get("is_building"):
                print("BUILDING:", {k: u.get(k) for k in ("entity_id", "unit_type", "tech_level", "can_upgrade")}, flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
