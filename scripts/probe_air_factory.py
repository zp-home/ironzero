"""Inspect airFactory and T2 landFactory menus, prices, and unit stats on Lake."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.rwpp.decoding import ACTION_KINDS


def main() -> int:
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
        opening_extractor=False,
        opening_factory=False,
        opening_production=False,
        opening_attack=False,
    )
    env = RwppDuelEnv(game, port=26_400, seed=1, episode_prefix="airmenu")
    try:
        env.reset()
        # Build an airFactory at (990, 2010) + (90, 0) using direct order
        built_air = False
        for step in range(250):
            own = env.state.get("own_entities") or []
            air_fac = [u for u in own if str(u.get("unit_type") or "").lower() == "airfactory"]
            if not air_fac:
                # order build airFactory near command center
                env.step(np.array([6, 4, 3, 6, 1]))  # or direct via socket? Let's just use step with menu index!
            else:
                if air_fac[0].get("is_operational") and float(air_fac[0].get("build_progress") or 0) >= 1.0:
                    print("airFactory built at step", step)
                    print("factories:", json.dumps(env.state.get("factories"), indent=2))
                    built_air = True
                    break
                env.step(np.array([9, 0, 0, 0, 0]))
        if not built_air:
            # Find airFactory index in build_menu and send direct order via _send_direct_order
            env.reset()
            for step in range(250):
                own = env.state.get("own_entities") or []
                air_fac = [u for u in own if str(u.get("unit_type") or "").lower() == "airfactory"]
                if not air_fac and step % 40 == 0:
                    # Send direct build_structure order
                    from ironzero.rwpp.stepping import send_step
                    send_step(env, np.array([9, 0, 0, 0, 0]), [{
                        "kind": "build_structure", "unit_group": 4,
                        "structure_type": "airFactory", "x": 1080.0, "y": 2010.0,
                    }])
                    # Wait for response by reading manually? No, can't double send_step!
                    break
                env.step(np.array([9, 0, 0, 0, 0]))
        # Better: test using build_menu index!
        env.reset()
        from ironzero.observation.menus import build_menu
        menu = build_menu(env.state)
        print("builder menu:", menu)
        air_idx = menu.index("airFactory")
        print("airFactory index:", air_idx)
        # Map base (990, 2010) -> grid
        gx = int(round(1080.0 / 2600.0 * 7))
        gy = int(round(2010.0 / 2600.0 * 7))
        for step in range(300):
            own = env.state.get("own_entities") or []
            air_fac = [u for u in own if str(u.get("unit_type") or "").lower() == "airfactory"]
            if not air_fac and step == 0:
                env.step(np.array([ACTION_KINDS.index("build_structure"), 4, gx, gy, air_idx]))
            else:
                env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            air_fac = [u for u in own if str(u.get("unit_type") or "").lower() == "airfactory"]
            if air_fac and float(air_fac[0].get("build_progress") or 0) >= 1.0 and air_fac[0].get("is_operational"):
                print("airFactory operational at step", step)
                print("factories:", json.dumps(env.state.get("factories"), indent=2))
                break
        # Also upgrade a landFactory? Wait: let's queue one lightGunship and print its stats!
        for entry in env.state.get("factories") or []:
            if str(entry.get("unit_type") or "").lower() == "airfactory":
                fid = entry["entity_id"]
                menu_air = entry.get("can_produce") or []
                print("airFactory can_produce:", menu_air)
                # produce lightGunship (or first combat unit)
                from ironzero.rwpp.decoding import produce_options
                opts = produce_options(env.state)
                print("produce_options:", opts)
                for idx, (f, u) in enumerate(opts):
                    if f == fid:
                        print("queuing", u, "idx", idx)
                        env.step(np.array([ACTION_KINDS.index("produce_unit"), 4, 3, 6, idx]))
                        break
        for step in range(250):
            env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            air_units = [u for u in own if u.get("is_air_unit")]
            if air_units:
                print("produced air unit:", json.dumps(air_units[0]))
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
