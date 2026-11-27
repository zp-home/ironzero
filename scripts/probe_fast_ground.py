"""Test ground-unit timing & composition vs difficulty-1 AI across 2p maps.

Discovery from probe_aipilot_d1.py:
  At tick 3011 the difficulty-1 AI has ONLY:
    commandCenter, 1 extractor, 1 c_antiAirTurret, 2 scouts (NO ground turrets, NO landFactory!)
  At tick 6011 it has ONLY:
    1 landFactory, 1 airFactory, 1 c_tank, 1 c_helicopter (STILL NO ground turrets!)

Our default curriculum wasted the first 4000 credits on:
  - 2 turrets (1000 cr)
  - 3 landFactories (2100+ cr)
  - credit_reserve=900 (freezing production whenever funds < 900!)
so our first wave didn't leave base until tick 10001 -- AFTER the AI fortified.

Here we test with ``noop`` policy (pure harness physics):
  - 0 turrets, 2 landFactories, credit_reserve=350 (never freeze tank queue)
  - attack_at = 6 or 8, wave arrives ~tick 3500-5000!
  - also test hoverTank (can cross water on Lake/Islands!) vs c_tank
  - across Lake, Two_cold_sides, Dire_Straight, Hills, Fire_Bridge, Small_Island.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.strategy.directive import MacroPlan
from ironzero.strategy.rule_strategy import RuleStrategy


class FastGroundStrategy(RuleStrategy):
    def __init__(self, *, attack_at: int = 8, max_factories: int = 2, max_extractors: int = 4, reserve: float = 350.0):
        super().__init__()
        object.__setattr__(self, "_attack_at", attack_at)
        object.__setattr__(self, "_max_factories", max_factories)
        object.__setattr__(self, "_max_extractors", max_extractors)
        object.__setattr__(self, "_reserve", reserve)

    def propose(self, observation_summary):
        d = super().propose(observation_summary)
        # Keep production ahead of expansion so factories spend starting 4000 cr on tanks IMMEDIATELY.
        macro = MacroPlan(
            build_order=("extractor", "factory", "production", "expansion", "attack"),
            army_target=24,
            army_cap=32,
            attack_at_army=self._attack_at,
            max_turrets=0,
            max_factories=self._max_factories,
            max_extractors=self._max_extractors,
            max_upgrades=0,
            credit_reserve=self._reserve,
            production_queue_depth=4,
            engage_distance=120.0,
        )
        return replace(d, macro=macro)


def run_one(map_rel: str, name: str, w: float, h: float, port: int, *,
            attack_at: int = 6, max_factories: int = 2, ticks: int = 19674,
            prefer_hover: bool = False) -> dict:
    import ironzero.rwpp.env as env_mod
    old_rank = dict(env_mod._UNIT_COMBAT_RANK)
    if prefer_hover:
        env_mod._UNIT_COMBAT_RANK["hoverTank"] = 6
    try:
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
            economy_army_cap=32,
            economy_army_target=24,
            economy_max_extractors=4,
            economy_max_factories=max_factories,
            economy_max_turrets=0,
            economy_max_upgrades=0,
            economy_credit_reserve=350.0,
            economy_production_queue_depth=4,
        )
        env = RwppDuelEnv(
            game, port=port, seed=port, episode_prefix="fastrush",
            strategy=FastGroundStrategy(attack_at=attack_at, max_factories=max_factories),
        )
        t0 = time.monotonic()
        res = None
        peak_army = 0
        best_income = 0.0
        first_attack_tick = None
        try:
            env.reset()
            for _ in range(ticks // 10 + 50):
                own = env.state.get("own_entities") or []
                army = sum(
                    1 for u in own
                    if u.get("can_attack") and not u.get("is_builder")
                    and not u.get("is_building") and float(u.get("hp") or 0) > 0
                )
                peak_army = max(peak_army, army)
                best_income = max(best_income, float((env.state.get("global_state") or {}).get("income") or 0))
                _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
                co = getattr(env, "curriculum_order", None)
                if first_attack_tick is None and isinstance(co, dict) and co.get("kind") == "attack_move":
                    first_attack_tick = env.state.get("tick")
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
    finally:
        env_mod._UNIT_COMBAT_RANK.clear()
        env_mod._UNIT_COMBAT_RANK.update(old_rank)
    return {
        "map": name,
        "hover": prefer_hover,
        "atk_at": attack_at,
        "first_atk_tick": first_attack_tick,
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
        ("maps/skirmish/[p2]Small_Island (2p).tmx", "Small_Island", 2600.0, 2600.0),
    ]
    for i, (mrel, name, w, h) in enumerate(maps):
        out_t = run_one(mrel, name, w, h, 26_800 + i * 20, attack_at=6, max_factories=2, prefer_hover=False)
        print("TANK :", json.dumps(out_t, ensure_ascii=False))
        out_h = run_one(mrel, name, w, h, 26_810 + i * 20, attack_at=6, max_factories=2, prefer_hover=True)
        print("HOVER:", json.dumps(out_h, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
