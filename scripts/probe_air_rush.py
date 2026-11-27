"""Test what composition actually BEATS difficulty-1 AI on Lake & other 2p maps.

Earlier probes showed difficulty-1 AI builds an ``airFactory`` (3-5 lightGunship,
c_helicopter) on EVERY 2p map, while our curriculum only built ``landFactory`` ->
``c_tank`` (can_attack_air=False).

Here we keep the exact same RwppDuelEnv + bridge, and simply test:
  1. Pure airFactory (c_helicopter, can fly over Lake + shoot air & land!)
  2. T2 airFactory (upgrade airFactory -> lightGunship / gunShip!)
  3. 1 landFactory + 1 airFactory
across Lake, Two_cold_sides, Dire_Straight, Hills, Fire_Bridge!
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.observation.menus import build_menu
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _is_operational
from ironzero.rwpp.decoding import ACTION_KINDS


def test_air_rush(map_rel: str, name: str, w: float, h: float, port: int, *,
                  upgrade_air: bool = False, attack_at: int = 8, max_air_fac: int = 2,
                  max_ext: int = 4, ticks: int = 19674) -> dict:
    game = RwppGameConfig(
        map_relative=map_rel,
        map_name=name,
        map_width_px=w,
        map_height_px=h,
        episode_ticks=ticks,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=8,
        units_per_team=8,
        opening_extractor=True,
        opening_factory=False,   # we drive airFactory directly below
        opening_production=False,
        opening_attack=True,
        economy_army_cap=32,
        economy_army_target=24,
        economy_max_extractors=max_ext,
        economy_max_upgrades=2 if upgrade_air else 0,
    )
    # Use a simple wrapper over env to test pure airFactory build + produce + attack
    env = RwppDuelEnv(game, port=port, seed=port, episode_prefix="airrush")
    t0 = time.monotonic()
    res = None
    peak_army = 0
    best_income = 0.0
    try:
        env.reset()
        # Override _issue_opening_factory / _issue_opening_production dynamically
        last_build_step = -999
        last_prod_step = -999
        last_upg_step = -999
        site_idx = 0
        for step in range(ticks // 10 + 50):
            own = env.state.get("own_entities") or []
            g = env.state.get("global_state") or {}
            funds = float(g.get("funds") or 0.0)
            best_income = max(best_income, float(g.get("income") or 0.0))
            extractors = [u for u in own if "xtractor" in str(u.get("unit_type") or "").lower()]
            air_facs = [u for u in own if str(u.get("unit_type") or "").lower() == "airfactory"]
            built_air = [u for u in air_facs if _is_operational(u)]
            unfinished = [u for u in own if u.get("is_building") and not _is_operational(u)]
            combat = [
                u for u in own
                if u.get("can_attack") and not u.get("is_builder")
                and not u.get("is_building") and float(u.get("hp") or 0.0) > 0.0
            ]
            peak_army = max(peak_army, len(combat))

            # Run normal step with noop, EXCEPT let's issue airFactory build / produce via direct orders
            # Wait: RwppDuelEnv.step sends action + _curriculum_actions().
            # Let's temporarily set env.macro.attack_at_army = attack_at
            object.__setattr__(env.macro, "attack_at_army", attack_at)

            act = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            # Check if opening extractor is done & no unfinished building:
            if extractors and not unfinished and len(air_facs) < max_air_fac and funds >= 1000 and step - last_build_step >= 18:
                cc = next((u for u in own if "commandcenter" in str(u.get("unit_type") or "").lower()), None)
                if cc:
                    offsets = [(90, 0), (-90, 0), (0, 110), (0, -110), (130, 100), (-130, 100)]
                    dx, dy = offsets[site_idx % len(offsets)]
                    site_idx += 1
                    wx = max(60.0, min(w - 60.0, float(cc["x"]) + dx))
                    wy = max(60.0, min(h - 60.0, float(cc["y"]) + dy))
                    gx = max(0, min(7, int(round(wx / w * 7))))
                    gy = max(0, min(7, int(round(wy / h * 7))))
                    menu = build_menu(env.state)
                    if "airFactory" in menu:
                        act = np.array([ACTION_KINDS.index("build_structure"), 4, gx, gy, menu.index("airFactory")], dtype=np.int64)
                        last_build_step = step
            elif built_air:
                # Upgrade first airFactory to T2 if requested and affordable
                fac_entries = [
                    f for f in (env.state.get("factories") or [])
                    if isinstance(f, dict) and str(f.get("unit_type") or "").lower() == "airfactory"
                    and f.get("is_built") is not False
                ]
                queued = sum(int(f.get("queue_length") or 0) for f in fac_entries)
                if upgrade_air and any(u.get("can_upgrade") and int(u.get("tech_level") or 1) < 2 for u in built_air) and funds >= 1800 and step - last_upg_step >= 20:
                    up_u = next(u for u in built_air if u.get("can_upgrade"))
                    # Use env._send_direct_order? Instead, we can just trigger upgrade action!
                    from ironzero.observation.menus import upgradable_buildings
                    ups = upgradable_buildings(env.state)
                    if ups:
                        uidx = next((i for i, u in enumerate(ups) if u.get("entity_id") == up_u.get("entity_id")), None)
                        if uidx is not None:
                            act = np.array([ACTION_KINDS.index("upgrade_building"), 4, 3, 3, uidx], dtype=np.int64)
                            last_upg_step = step
                elif queued < 3 and funds >= 450 and step - last_prod_step >= 2:
                    from ironzero.observation.menus import produce_options
                    opts = produce_options(env.state)
                    # Prefer strongest air unit in menu (gunShip / lightGunship / helicopter)
                    pref = ["gunShip", "lightGunship", "c_helicopter", "helicopter", "c_interceptor", "interceptor"]
                    best_idx = None
                    best_rank = 999
                    air_ids = {int(f["entity_id"]) for f in fac_entries if f.get("entity_id") is not None}
                    for idx, (fid, uname) in enumerate(opts):
                        if fid not in air_ids:
                            continue
                        r = pref.index(uname) if uname in pref else 50
                        if r < best_rank:
                            best_rank = r
                            best_idx = idx
                    if best_idx is not None:
                        act = np.array([ACTION_KINDS.index("produce_unit"), 4, 3, 3, best_idx], dtype=np.int64)
                        last_prod_step = step

            _o, _r, t, tr, info = env.step(act)
            if t or tr:
                res = info.get("result")
                break
        own_types = Counter(str(u.get("unit_type") or "?") for u in (env.state.get("own_entities") or []))
        foe_types = Counter(str(u.get("unit_type") or "?") for u in (env.state.get("visible_enemy_entities") or []))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "map": name,
        "upgrade_air": upgrade_air,
        "attack_at": attack_at,
        "result": res,
        "tick": env.state.get("tick"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "peak_army": peak_army,
        "best_income": best_income,
        "own_end": dict(own_types),
        "foe_end": dict(foe_types),
    }


def main() -> int:
    maps = [
        ("maps/skirmish/[p2]Lake (2p).tmx", "Lake", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Two_cold_sides (2p).tmx", "Two_cold_sides", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Hills_(2p)_[By Tstis & KPSS].tmx", "Hills", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Dire_Straight (2p) [by uber].tmx", "Dire_Straight", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Fire Bridge (2p) [by uber].tmx", "Fire_Bridge", 2600.0, 2600.0),
    ]
    for i, (mrel, name, w, h) in enumerate(maps):
        out = test_air_rush(mrel, name, w, h, 26_600 + i * 20, upgrade_air=False, attack_at=8)
        print("AIR_T1:", json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
