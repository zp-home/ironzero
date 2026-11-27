"""Why does the second landFactory never progress (build_progress stuck at 0)?

Every probe so far shows ('landFactory', True, [...]) and ('landFactory', False, [])
-- the second hull exists but never advances. Production rate (one working
factory, queue depth 4) is what caps peak army at 13-18, so a dead second factory
is a real 2x throughput loss.

Traces the factory stage's stall/progress state across the episode to see whether
the order is being re-issued (restarting construction) or never re-issued at all.
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
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=28390, seed=71000, episode_prefix="fact2",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    factory_orders: Counter[str] = Counter()
    orig = env._issue_opening_factory

    def wrap():
        out = orig()
        if out:
            factory_orders["sent"] += 1
            for d in out:
                payload = d.get("payload") or {}
                factory_orders[f"{payload.get('structure_type')}@{d.get('target')}"] += 1
        else:
            factory_orders["idle"] += 1
        return out

    env._issue_opening_factory = wrap
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step in (200, 400, 600, 800, 1200, 1600, 1900):
                land = [u for u in (env.state.get("own_entities") or [])
                        if "landfactory" in str(u.get("unit_type", "")).lower()]
                desc = ", ".join(
                    f"id={u['entity_id']} op={u.get('is_operational')} "
                    f"prog={u.get('build_progress')} hp={u.get('hp')}"
                    for u in land
                )
                print(f"step={step} tick={env.state.get('tick')} | {desc}", flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')}", flush=True)
                break
        print("\nfactory stage outcomes:", dict(factory_orders.most_common(12)), flush=True)
        print("stall counters:", {k: getattr(env, k, None) for k in
              ("_factory_progress", "_factory_stalled", "_factory_orders_at_site",
               "_factory_candidate_offset", "factory_site")}, flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
