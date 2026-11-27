"""Trace _issue_late_expansion's internal decision path per call.

The gate probe said the gates open on ~294 decisions but the stage emitted only
3 orders. So the refusal happens *after* the gates: the distance/stall logic.
This wraps the real method and records the builder->pool distance each call.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


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
    env = RwppDuelEnv(game, port=27993, seed=71000, episode_prefix="exptrace",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=300.0, army_cap=40))

    orig = env._issue_late_expansion
    log: list[str] = []

    def wrapper():
        before = (
            getattr(env, "_expansion_target", None),
            getattr(env, "_expansion_ordered_at", None),
            getattr(env, "_expansion_pool_offset", 0),
        )
        out = orig()
        after = (
            getattr(env, "_expansion_target", None),
            getattr(env, "_expansion_ordered_at", None),
            getattr(env, "_expansion_pool_offset", 0),
        )
        if out or before != after:
            slots = env.state.get("resource_slots") or []
            occ = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
            builder = next((u for u in (env.state.get("own_entities") or [])
                            if u.get("is_builder")), None)
            tgt = next((s for s in slots if s.get("slot_id") == after[0]), None)
            dist = None
            if builder and tgt:
                dist = round(((float(builder["x"]) - float(tgt["x"])) ** 2
                              + (float(builder["y"]) - float(tgt["y"])) ** 2) ** 0.5)
            log.append(
                f"tick={env.state.get('tick')} out={bool(out)} occ={occ} "
                f"before={before} after={after} dist={dist} "
                f"step={env.episode_steps}"
            )
        return out

    env._issue_late_expansion = wrapper
    try:
        env.reset(seed=71000)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                print(f"RESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
        print(f"\nexpansion events: {len(log)}", flush=True)
        for line in log[:40]:
            print(" ", line, flush=True)
        print("\nfinal income:", (env.state.get("global_state") or {}).get("income"), flush=True)
        slots = env.state.get("resource_slots") or []
        print("occupied slots:", sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")),
              "of", len(slots), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
