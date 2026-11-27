"""Log every internal state variable of _issue_late_expansion per call.

Round 12 fixed the menu death spiral (extractors 1 -> 2). Expansion now stops at
2 of 6 configured. probe_expansion_trace showed only 4 state-changing events
(tick 1571, 3331, 3341, 4581) and then nothing for 15000 ticks, so the stage is
parked in a branch that returns [] without touching state. This records the
variables that decide those branches so the parking branch is identified rather
than guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories, _is_operational
from scripts.probe_lake_winrate import RushStrategy

FIELDS = (
    "_expansion_target",
    "_expansion_ordered_at",
    "_expansion_distance",
    "_expansion_leg_distance",
    "_expansion_stalled",
    "_expansion_occupied",
    "_expansion_pool_offset",
)


def snapshot(env) -> tuple:
    return tuple(getattr(env, name, None) for name in FIELDS)


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
        max_entities=16,
        units_per_team=16,
        economy_army_cap=40,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=300.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=28001, seed=71000, episode_prefix="expstate",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=300.0, army_cap=40))
    orig = env._issue_late_expansion
    prev = [None]

    def wrapper():
        before = snapshot(env)
        out = orig()
        after = snapshot(env)
        step = env.episode_steps
        # Log every call from the point where expansion goes quiet.
        if step >= 440 and step % 1 == 0 and (out or before != after or step % 100 == 0):
            slots = env.state.get("resource_slots") or []
            free = [i for i, s in enumerate(slots) if isinstance(s, dict) and not s.get("occupied")]
            own = env.state.get("own_entities") or []
            builder = next((u for u in own if u.get("is_builder")), None)
            tgt = next((s for s in slots if s.get("slot_id") == after[0]), None)
            dist = None
            if builder and tgt:
                dist = round(((float(builder["x"]) - float(tgt["x"])) ** 2
                              + (float(builder["y"]) - float(tgt["y"])) ** 2) ** 0.5)
            budget = env._expansion_walk_budget(after[3] if after[3] is not None else dist)
            age = None if after[1] is None else step - int(after[1])
            print(
                f"step={step:4d} out={bool(out)} occ={sum(1 for s in slots if isinstance(s,dict) and s.get('occupied'))} "
                f"free={free} tgt={after[0]} age={age} budget={budget} dist={dist} "
                f"leg={after[3]} off={after[6]} stuck={after[4]}",
                flush=True,
            )
        prev[0] = after
        return out

    env._issue_late_expansion = wrapper
    try:
        env.reset(seed=71000)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                print(f"\nRESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
        print("final income:", (env.state.get("global_state") or {}).get("income"), flush=True)
        print("combat_factories:", len(_combat_factories(env.state)), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
