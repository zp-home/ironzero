"""Instrument why the factory stage stops at 2 operational landFactories even
when economy_max_factories is 4 or 6 (probe_factory_quota: dead_sites=0, so it
is not the candidate ladder; the order budget is also far from exhausted).

Logs the first gate that refuses on each decision after the second factory is up.
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
from ironzero.rwpp.env import RwppDuelEnv, _is_operational
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
        economy_max_factories=6,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=28850, seed=71000, episode_prefix="facgate",
                      strategy=RushStrategy(attack_at=8, max_factories=6,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    reasons: Counter[str] = Counter()
    orig = env._issue_opening_factory

    def wrap():
        out = orig()
        state = env.state
        own = state.get("own_entities") or []
        land = [u for u in own
                if str(u.get("unit_type", "")).lower() == "landfactory"]
        ops = [u for u in land if _is_operational(u)]
        if len(ops) >= 2 and not out:
            # Classify the refusal using the same gates as the method.
            if getattr(env, "factory_site", None) is not None:
                reasons["site_latched_waiting"] += 1
            elif int(getattr(env, "_factory_orders", 0)) >= (len(ops) + 1) * 56:
                reasons["order_budget"] += 1
            elif not any(u.get("is_builder") for u in own):
                reasons["no_builder"] += 1
            else:
                ex = [u for u in own if "xtractor" in str(u.get("unit_type", "")).lower()]
                if ex and any(not _is_operational(u) for u in ex):
                    reasons["extractor_busy"] += 1
                else:
                    reasons["other"] += 1
        return out

    env._issue_opening_factory = wrap
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step % 400 == 0:
                own = env.state.get("own_entities") or []
                land = [u for u in own
                        if str(u.get("unit_type", "")).lower() == "landfactory"]
                ops = sum(1 for u in land if _is_operational(u))
                print(f"step={step:4d} tick={env.state.get('tick')} factories={ops}/{len(land)} "
                      f"orders={env._factory_orders} site={getattr(env,'factory_site',None)} "
                      f"funds={(env.state.get('global_state') or {}).get('funds')}", flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')}", flush=True)
                break
        print("\nrefusal reasons after 2 factories:", dict(reasons.most_common()), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
