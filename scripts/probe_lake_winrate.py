"""Check difficulty-1 win rate on Lake (8 seeds) and test multi-map fixes."""
from __future__ import annotations

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


class RushStrategy(RuleStrategy):
    def __init__(
        self,
        *,
        attack_at: int = 6,
        max_factories: int = 2,
        max_extractors: int = 4,
        max_turrets: int = 0,
        max_upgrades: int = 0,
        reserve: float = 350.0,
        army_target: int = 24,
        army_cap: int = 32,
        unit_pref: tuple[str, ...] = (),
    ):
        super().__init__()
        object.__setattr__(self, "_cfg", dict(
            attack_at=attack_at,
            max_factories=max_factories,
            max_extractors=max_extractors,
            max_turrets=max_turrets,
            max_upgrades=max_upgrades,
            reserve=reserve,
            army_target=army_target,
            army_cap=army_cap,
            unit_pref=unit_pref,
        ))

    def propose(self, observation_summary):
        d = super().propose(observation_summary)
        c = self._cfg
        macro = MacroPlan(
            build_order=("extractor", "factory", "production", "expansion", "attack"),
            army_target=c["army_target"],
            army_cap=c["army_cap"],
            attack_at_army=c["attack_at"],
            max_turrets=c["max_turrets"],
            max_factories=c["max_factories"],
            max_extractors=c["max_extractors"],
            max_upgrades=c["max_upgrades"],
            credit_reserve=c["reserve"],
            production_queue_depth=4,
            engage_distance=120.0,
        )
        return replace(d, macro=macro)


def run_ep(map_rel: str, name: str, w: float, h: float, seed: int, port: int, *, ticks: int = 19674, **kw) -> dict:
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
        economy_army_cap=kw.get("army_cap", 32),
        economy_army_target=kw.get("army_target", 24),
        economy_max_extractors=kw.get("max_extractors", 4),
        economy_max_factories=kw.get("max_factories", 2),
        economy_max_turrets=kw.get("max_turrets", 0),
        economy_max_upgrades=kw.get("max_upgrades", 0),
        economy_credit_reserve=kw.get("reserve", 350.0),
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(
        game, port=port, seed=seed, episode_prefix="lakewr",
        strategy=RushStrategy(**kw),
    )
    t0 = time.monotonic()
    res = None
    try:
        env.reset(seed=seed)
        for _ in range(ticks // 10 + 50):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                res = info.get("result")
                break
        own = Counter(str(u.get("unit_type") or "?") for u in (env.state.get("own_entities") or []))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {"seed": seed, "res": res, "tick": env.state.get("tick"), "s": round(time.monotonic() - t0, 2), "own": dict(own)}


def main() -> int:
    wins = 0
    n = 8
    for i in range(n):
        r = run_ep("maps/skirmish/[p2]Lake (2p).tmx", "Lake", 2600.0, 2600.0, 71000 + i, 26400 + i, attack_at=6)
        if r["res"] == "victory":
            wins += 1
        print(f"Lake atk=6 ep={i}: {r}", flush=True)
    print(f"Lake atk=6 wins: {wins}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
