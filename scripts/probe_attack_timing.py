"""Test only attack timing tweaks, keeping the verified production config unchanged."""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.directive import MacroPlan  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402
from dataclasses import replace  # noqa: E402


class AttackTune(RuleStrategy):
    def __init__(self, attack_at: int):
        super().__init__()
        object.__setattr__(self, "_attack_at", attack_at)

    def propose(self, observation_summary):
        d = super().propose(observation_summary)
        macro = d.macro
        if macro is not None:
            d = replace(d, macro=replace(macro, attack_at_army=self._attack_at))
        return d


def run(port: int, attack_at: int, ticks: int) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
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
        economy_army_target=20,
        economy_max_extractors=9,
        economy_max_factories=3,
        economy_production_queue_depth=3,
    )
    env = RwppDuelEnv(game, port=port, seed=port, episode_prefix="atune", strategy=AttackTune(attack_at))
    res = None
    peak_army = 0
    min_enemy_building_hp = None
    t0 = time.monotonic()
    try:
        env.reset()
        for _ in range(ticks // 10 + 50):
            own = env.state.get("own_entities") or []
            army = sum(1 for u in own if u.get("can_attack") and not u.get("is_builder")
                       and not u.get("is_building") and float(u.get("hp") or 0) > 0)
            peak_army = max(peak_army, army)
            foes = env.state.get("visible_enemy_entities") or []
            ehp = sum(float(u.get("hp") or 0) for u in foes if u.get("is_building") or u.get("is_factory"))
            if min_enemy_building_hp is None or ehp < min_enemy_building_hp:
                min_enemy_building_hp = ehp
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                res = info.get("result")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "attack_at": attack_at,
        "result": res,
        "tick": env.state.get("tick"),
        "peak_army": peak_army,
        "min_enemy_building_hp": min_enemy_building_hp,
        "elapsed": round(time.monotonic() - t0, 2),
    }


def main() -> int:
    for attack_at in (6, 8, 10):
        print(json.dumps(run(26_000 + attack_at, attack_at, 19674)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
