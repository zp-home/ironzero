"""Test fast-economy / early-push variants vs difficulty 1 on Lake (curriculum only).

In ``probe_aipilot_d1.py``, at tick 6011 the difficulty-1 AI had ONLY:
  1 landFactory, 1 airFactory, 1 c_tank, 1 c_helicopter, 0 ground turrets!
By tick 13001 (when our old attack_at=11..20 arrived), the AI had 10 buildings,
artillery, gunships, and ground turrets.

Also:
  * ``economy_credit_reserve=900`` froze our factories whenever funds were 350..899.
  * ``max_factories=3, max_turrets=2`` burned ~2500 of our 4000 starting credits on
    extra buildings instead of early tanks + extractors.
  * ``hoverTank`` can cross the central lake directly instead of walking around it.

This tests variations cleanly on Lake difficulty 1 with lockstep=True, frame_ms=0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.directive import MacroPlan  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


class TunedStrategy(RuleStrategy):
    def __init__(self, *, attack_at: int, army_target: int, max_turrets: int, reserve: float, max_factories: int):
        super().__init__()
        object.__setattr__(self, "_attack_at", attack_at)
        object.__setattr__(self, "_army_target", army_target)
        object.__setattr__(self, "_max_turrets", max_turrets)
        object.__setattr__(self, "_reserve", reserve)
        object.__setattr__(self, "_max_factories", max_factories)

    def propose(self, observation_summary):
        d = super().propose(observation_summary)
        # Keep attack & expansion even if outnumbered/hurt, prioritize fast push
        macro = MacroPlan(
            build_order=("extractor", "factory", "expansion", "production", "upgrade", "attack"),
            army_target=self._army_target,
            army_cap=32,
            attack_at_army=self._attack_at,
            max_turrets=self._max_turrets,
            max_factories=self._max_factories,
            max_extractors=9,
            credit_reserve=self._reserve,
            production_queue_depth=4,
        )
        from dataclasses import replace
        return replace(d, macro=macro)


def run_one(port: int, seed: int, *, attack_at: int, army_target: int, max_turrets: int, reserve: float, max_factories: int, ticks: int, lockstep: bool) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=ticks,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0 if lockstep else 1,
        physics_ms=16,
        lockstep=lockstep,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=army_target,
        economy_max_extractors=9,
        economy_max_factories=max_factories,
        economy_max_turrets=max_turrets,
        economy_credit_reserve=reserve,
        economy_production_queue_depth=4,
    )
    strat = TunedStrategy(
        attack_at=attack_at,
        army_target=army_target,
        max_turrets=max_turrets,
        reserve=reserve,
        max_factories=max_factories,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="rush", strategy=strat)
    t0 = time.monotonic()
    res = None
    peak_army = 0
    best_income = 0.0
    decisions = 0
    try:
        env.reset()
        for step in range(ticks // 10 + 100):
            decisions += 1
            own = env.state.get("own_entities") or []
            army = sum(
                1 for u in own
                if u.get("can_attack") and not u.get("is_builder")
                and not u.get("is_building") and float(u.get("hp") or 0) > 0
            )
            peak_army = max(peak_army, army)
            best_income = max(best_income, float((env.state.get("global_state") or {}).get("income") or 0))
            _o, _r, term, trunc, info = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
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
        "seed": seed,
        "result": res,
        "tick": env.state.get("tick"),
        "decisions": decisions,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "peak_army": peak_army,
        "best_income": best_income,
        "own_end": dict(own_types),
        "foe_end": dict(foe_types),
    }


def main() -> int:
    for label, kw in [
        ("early8_1fac_res350", dict(attack_at=8, army_target=24, max_turrets=0, reserve=350.0, max_factories=1, ticks=19674, lockstep=True)),
        ("early8_2fac_res350", dict(attack_at=8, army_target=24, max_turrets=0, reserve=350.0, max_factories=2, ticks=19674, lockstep=True)),
        ("early6_2fac_res350", dict(attack_at=6, army_target=24, max_turrets=0, reserve=350.0, max_factories=2, ticks=19674, lockstep=True)),
    ]:
        out = run_one(25_900, 101, **kw)
        print(label, json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
