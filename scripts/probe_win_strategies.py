"""Diagnose and benchmark difficulty-1 winning strategies on Lake.

Tests cleanly without PPO noise:
  1. hover: HoverRush (force hoverTank only, reserve=450, crosses Lake water)
  2. air_t1: AirRush T1 (exact-coord airFactory -> c_helicopter, flies + AA)
  3. air_t2: AirRush T2 (upgrade airFactory -> lightGunship / gunShip)
  4. tank_cadence: c_tank with attack_move every 20 decisions + exact target (tx, ty)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _is_operational
from ironzero.strategy.directive import MacroPlan
from ironzero.strategy.rule_strategy import RuleStrategy


class CustomMacroStrategy(RuleStrategy):
    def __init__(self, *, build_order: tuple[str, ...], attack_at: int = 6,
                 max_factories: int = 2, max_extractors: int = 4,
                 max_upgrades: int = 0, reserve: float = 350.0):
        super().__init__()
        object.__setattr__(self, "_macro_cfg", dict(
            build_order=build_order,
            attack_at=attack_at,
            max_factories=max_factories,
            max_extractors=max_extractors,
            max_upgrades=max_upgrades,
            reserve=reserve,
        ))

    def propose(self, observation_summary):
        d = super().propose(observation_summary)
        c = self._macro_cfg
        return replace(
            d,
            macro=MacroPlan(
                build_order=c["build_order"],
                army_target=24,
                army_cap=32,
                attack_at_army=c["attack_at"],
                max_turrets=0,
                max_factories=c["max_factories"],
                max_extractors=c["max_extractors"],
                max_upgrades=c["max_upgrades"],
                credit_reserve=c["reserve"],
                production_queue_depth=4,
                engage_distance=120.0,
            ),
        )


def _pick_target(foes: list[dict]) -> dict | None:
    cc = next((u for u in foes if "commandcenter" in str(u.get("unit_type") or "").lower()), None)
    if cc is not None:
        return cc
    facs = [u for u in foes if u.get("is_factory")]
    if facs:
        return facs[0]
    blds = [u for u in foes if u.get("is_building")]
    if blds:
        return blds[0]
    return foes[0] if foes else None


def run_strategy(mode: str, n_seeds: int = 4, port_base: int = 27200) -> dict:
    wins = 0
    details = []
    printed_menu = False
    for i in range(n_seeds):
        seed = 71000 + i
        build_order = ("extractor", "factory", "production", "expansion", "attack")
        reserve = 350.0
        if mode == "hover":
            reserve = 450.0
            build_order = ("extractor", "factory", "expansion", "attack")  # drive production directly for hoverTank only
        elif mode in ("air_t1", "air_t2"):
            build_order = ("extractor", "expansion")  # drive airFactory + production + attack cleanly
        elif mode == "tank_cadence":
            build_order = ("extractor", "factory", "production", "expansion")  # drive exact attack every 20 steps

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
            economy_credit_reserve=reserve,
            economy_production_queue_depth=4,
        )
        strat = CustomMacroStrategy(
            build_order=build_order,
            attack_at=6 if mode != "air_t2" else 8,
            max_factories=2,
            max_extractors=4,
            reserve=reserve,
        )
        env = RwppDuelEnv(game, port=port_base + i, seed=seed, episode_prefix=f"win_{mode}", strategy=strat)
        t0 = time.monotonic()
        res = None
        peak_army = 0
        own_built_types: Counter[str] = Counter()
        seen_ids: set[int] = set()
        last_build_step = -999
        last_prod_step = -999
        last_atk_step = -999
        last_upg_step = -999
        site_idx = 0
        try:
            env.reset(seed=seed)
            noop = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            # Hook into env._send_step? Instead just monkeypatch env._curriculum_actions cleanly!
            orig_curr = env._curriculum_actions
            step_box = [0]

            def custom_curr():
                orders = list(orig_curr())
                step = step_box[0]
                state = env.state
                own = state.get("own_entities") or []
                foes = state.get("visible_enemy_entities") or []
                facs = [f for f in (state.get("factories") or []) if isinstance(f, dict)]
                g = state.get("global_state") or {}
                funds = float(g.get("funds") or 0.0)
                nonlocal printed_menu, last_build_step, last_prod_step, last_atk_step, last_upg_step, site_idx

                if mode == "hover":
                    # Queue ONLY hoverTank from built landFactory
                    land_facs = [
                        f for f in facs
                        if "landfactory" in str(f.get("unit_type") or "").lower()
                        and f.get("is_built") is not False
                    ]
                    if land_facs and not printed_menu:
                        printed_menu = True
                        print(f"[MENU {mode}] {[(f.get('unit_type'), f.get('can_produce'), f.get('can_produce_price')) for f in land_facs]}", flush=True)
                    if land_facs and funds >= 450 and step - last_prod_step >= 1:
                        for f in land_facs:
                            q = int(f.get("queue_length") or 0)
                            if q < 3:
                                menu = [str(x) for x in (f.get("can_produce") or [])]
                                ht = next((x for x in menu if "hover" in x.lower()), None)
                                if ht:
                                    orders.append({"kind": "produce_unit", "factory_entity_id": int(f["entity_id"]), "unit_type": ht})
                                    last_prod_step = step
                                    break

                elif mode in ("air_t1", "air_t2"):
                    extractors = [u for u in own if "xtractor" in str(u.get("unit_type") or "").lower()]
                    air_facs = [u for u in own if "airfactory" in str(u.get("unit_type") or "").lower()]
                    unfinished = [u for u in own if u.get("is_building") and not _is_operational(u)]
                    if extractors and not unfinished and len(air_facs) < 2 and funds >= 1000 and step - last_build_step >= 18:
                        cc = next((u for u in own if "commandcenter" in str(u.get("unit_type") or "").lower()), None)
                        if cc:
                            offsets = [(110, 0), (-110, 0), (0, -110), (140, -90), (-140, -90), (0, 120), (170, 0)]
                            dx, dy = offsets[site_idx % len(offsets)]
                            site_idx += 1
                            wx = max(80.0, min(2520.0, float(cc["x"]) + dx))
                            wy = max(80.0, min(2520.0, float(cc["y"]) + dy))
                            orders.insert(0, {"kind": "build_structure", "structure_type": "airFactory", "x": wx, "y": wy})
                            last_build_step = step

                    built_air_facs = [
                        f for f in facs
                        if "airfactory" in str(f.get("unit_type") or "").lower()
                        and f.get("is_built") is not False
                    ]
                    if built_air_facs and not printed_menu:
                        printed_menu = True
                        print(f"[MENU {mode}] {[(f.get('unit_type'), f.get('tech_level'), f.get('can_produce'), f.get('can_produce_price')) for f in built_air_facs]}", flush=True)

                    t1_air = [
                        u for u in own
                        if "airfactory" in str(u.get("unit_type") or "").lower()
                        and _is_operational(u) and u.get("can_upgrade") and int(u.get("tech_level") or 1) < 2
                    ]
                    # Upgrade first airFactory to T2 in air_t2 mode after 2 extractors or 1 factory
                    if mode == "air_t2" and t1_air and len(built_air_facs) >= 1 and funds >= 1800 and step - last_upg_step >= 20:
                        orders.append({"kind": "upgrade_building", "entity_id": int(t1_air[0]["entity_id"])})
                        last_upg_step = step
                    elif built_air_facs and funds >= 450 and step - last_prod_step >= 1:
                        # Prefer best air unit
                        pref = ["gunship", "lightgunship", "helicopter"]
                        for f in built_air_facs:
                            if int(f.get("queue_length") or 0) >= 3:
                                continue
                            menu = [str(x) for x in (f.get("can_produce") or [])]
                            prices = f.get("can_produce_price") or []
                            chosen = None
                            for p in pref:
                                for idx, uname in enumerate(menu):
                                    if p in uname.lower():
                                        pr = float(prices[idx]) if idx < len(prices) else 500.0
                                        if pr <= funds:
                                            chosen = uname
                                            break
                                if chosen:
                                    break
                            if chosen:
                                orders.append({"kind": "produce_unit", "factory_entity_id": int(f["entity_id"]), "unit_type": chosen})
                                last_prod_step = step
                                break

                    # Attack drive for air units every 20 steps with exact target
                    com = [u for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building") and float(u.get("hp") or 0) > 0]
                    atk_gate = 6 if mode == "air_t1" else 6
                    if len(com) >= atk_gate and step - last_atk_step >= 20:
                        tgt = _pick_target(foes)
                        if tgt is not None:
                            orders.append({
                                "kind": "attack_move",
                                "unit_group": env.micro.groups,
                                "x": float(tgt["x"]),
                                "y": float(tgt["y"]),
                            })
                            last_atk_step = step

                elif mode == "tank_cadence":
                    com = [u for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building") and float(u.get("hp") or 0) > 0]
                    if len(com) >= 6 and step - last_atk_step >= 20:
                        tgt = _pick_target(foes)
                        if tgt is not None:
                            orders.append({
                                "kind": "attack_move",
                                "unit_group": env.micro.groups,
                                "x": float(tgt["x"]),
                                "y": float(tgt["y"]),
                            })
                            last_atk_step = step

                return orders

            env._curriculum_actions = custom_curr
            for step in range(2020):
                step_box[0] = step
                _o, _r, t, tr, info = env.step(noop)
                own = env.state.get("own_entities") or []
                for u in own:
                    eid = int(u.get("entity_id") or 0)
                    if eid not in seen_ids and not u.get("is_building") and not u.get("is_builder"):
                        seen_ids.add(eid)
                        own_built_types[str(u.get("unit_type") or "?")] += 1
                com = [u for u in own if u.get("can_attack") and not u.get("is_builder") and not u.get("is_building") and float(u.get("hp") or 0) > 0]
                peak_army = max(peak_army, len(com))
                if t or tr:
                    res = info.get("result")
                    break
            if res == "victory":
                wins += 1
            rec = {
                "ep": i,
                "res": res,
                "tick": env.state.get("tick"),
                "peak_army": peak_army,
                "built": dict(own_built_types),
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
            details.append(rec)
            print(f"[{mode}] {json.dumps(rec)}", flush=True)
        finally:
            env.close()
    print(f"=== {mode}: {wins}/{n_seeds} ===", flush=True)
    return {"mode": mode, "wins": wins, "n": n_seeds, "details": details}


def main() -> int:
    for idx, mode in enumerate(("hover", "air_t1", "air_t2", "tank_cadence")):
        run_strategy(mode, n_seeds=4, port_base=27200 + idx * 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
