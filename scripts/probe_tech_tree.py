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


def wait_steps(env: RwppDuelEnv, n: int) -> bool:
    noop = np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0], dtype=np.int64)
    for _ in range(n):
        _, _, terminated, truncated, info = env.step(noop)
        if terminated or truncated:
            print(f"[DONE] tick={env.last_tick} term={terminated} trunc={truncated} info={info}", flush=True)
            return True
    return False


def dump_units(env: RwppDuelEnv, seen_types: set[str]) -> None:
    for u in env.state.get("own_entities") or []:
        ut = str(u.get("unit_type") or "")
        if not ut or ut in seen_types:
            continue
        seen_types.add(ut)
        print("UNIT:", {
            k: u.get(k)
            for k in (
                "unit_type",
                "is_building",
                "tech_level",
                "max_hp",
                "speed",
                "attack_range",
                "can_attack",
                "can_attack_air",
                "can_attack_land",
                "can_attack_water",
                "movement_type",
                "can_build",
                "can_build_price",
                "can_upgrade",
                "upgrade_cost",
            )
            if k in u
        }, flush=True)


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=60000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=32,
        units_per_team=32,
    )
    env = RwppDuelEnv(game, port=26910, seed=1, episode_prefix="tree")
    env._curriculum_disabled = True
    seen_types: set[str] = set()
    try:
        env.reset(seed=1)
        menu = build_menu(env.obs_state)
        print("MENU:", menu, flush=True)
        for b in env.state.get("builders") or []:
            print("BUILDER:", b, flush=True)

        # Build extractors first for fast income (up to 4)
        for _ in range(4):
            slots = env.state.get("resource_pools") or []
            free = [s["slot_id"] for s in slots if not s.get("occupied")]
            if not free:
                break
            for _ in range(100):
                if float(env.state.get("funds", 0.0) or 0.0) >= 750.0:
                    break
                if wait_steps(env, 5):
                    return 0
            env._send_direct_order({"kind": "build_extractor", "unit_group": 8, "slot_id": int(free[0])}, kind="build_extractor")
            for _ in range(80):
                if wait_steps(env, 5):
                    return 0
                cur_slots = env.state.get("resource_pools") or []
                if any(s["slot_id"] == free[0] and s.get("occupied") for s in cur_slots):
                    break

        # Build available production structures near CC
        targets = [
            ("airFactory", 2, 6),
            ("landFactory", 4, 6),
        ]

        for stype, gx, gy in targets:
            if stype not in menu:
                continue
            need_funds = 1000.0
            for _ in range(100):
                if float(env.state.get("funds", 0.0) or 0.0) >= need_funds:
                    break
                if wait_steps(env, 5):
                    return 0
            idx = menu.index(stype)
            before_ids = {int(u["entity_id"]) for u in (env.state.get("own_entities") or [])}
            res = env.step(np.array([ACTION_KINDS.index("build_structure"), 8, gx, gy, idx], dtype=np.int64))
            print(f"build {stype} @ ({gx},{gy}) funds={env.state.get('funds')}", flush=True)
            for _ in range(30):
                if wait_steps(env, 5):
                    return 0
                own_now = env.state.get("own_entities") or []
                new_bld = [u for u in own_now if int(u["entity_id"]) not in before_ids and u.get("is_building")]
                if new_bld and new_bld[0].get("is_operational"):
                    break

        dump_units(env, seen_types)
        print("\n=== T1 FACTORIES ===", flush=True)
        for f in env.state.get("factories") or []:
            print("T1 factory:", f, flush=True)

        # Upgrade landFactory, airFactory, extractor(s)
        upgraded_ids: set[int] = set()
        for _round in range(6):
            own = env.state.get("own_entities") or []
            candidates = [
                u for u in own
                if u.get("is_building") and u.get("can_upgrade") and u.get("is_operational")
                and int(u.get("entity_id", -1)) not in upgraded_ids
            ]
            if not candidates:
                break
            candidates.sort(key=lambda u: (0 if "factory" in str(u.get("unit_type", "")).lower() else 1, float(u.get("upgrade_cost") or 0)))
            target = candidates[0]
            cost = float(target.get("upgrade_cost") or 1500.0)
            for _ in range(300):
                if float(env.state.get("funds", 0.0) or 0.0) >= cost:
                    break
                if wait_steps(env, 5):
                    return 0
            bid = int(target["entity_id"])
            res = env._send_direct_order({"kind": "upgrade_building", "unit_group": 0, "building_entity_id": bid}, {"kind": "upgrade_building"})
            print(f"Upgrade sent: {target.get('unit_type')} id={bid} cost={cost} res={res}", flush=True)
            upgraded_ids.add(bid)
            for step_i in range(350):
                if wait_steps(env, 1):
                    return 0
                cur = next((u for u in (env.state.get("own_entities") or []) if int(u.get("entity_id", -1)) == bid), None)
                f_cur = next((f for f in (env.state.get("factories") or []) if int(f.get("entity_id", -1)) == bid), None)
                if cur and int(cur.get("tech_level", 1) or 1) >= 2 and not (f_cur and f_cur.get("queued")):
                    print(f"  -> upgraded in {step_i + 1} steps! tech_level={cur.get('tech_level')}", flush=True)
                    break

        print("\n=== T2 FACTORIES ===", flush=True)
        factories = env.state.get("factories") or []
        for f in factories:
            print("T2 factory:", f, flush=True)

        # Produce 1 of every unit type from each factory
        produced_types: set[str] = set()
        for f in list(factories):
            fid = int(f["entity_id"])
            can_p = list(f.get("can_produce") or [])
            prices = list(f.get("prices") or [])
            for utype, price in zip(can_p, prices):
                if utype in produced_types or utype == "builder":
                    continue
                produced_types.add(utype)
                for _ in range(300):
                    if float(env.state.get("funds", 0.0) or 0.0) >= float(price):
                        break
                    if wait_steps(env, 5):
                        return 0
                res = env._send_direct_order({"kind": "produce_unit", "factory_entity_id": fid, "unit_type": utype}, kind="produce_unit")
                print(f"Produce {utype} (price={price}) from {f.get('unit_type')} id={fid}: res={res}", flush=True)
                for _ in range(250):
                    if wait_steps(env, 1):
                        return 0
                    f_now = next((x for x in (env.state.get("factories") or []) if int(x.get("entity_id", -1)) == fid), None)
                    if not (f_now and f_now.get("queued")):
                        break
                dump_units(env, seen_types)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
