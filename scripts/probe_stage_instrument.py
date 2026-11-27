"""Instrument the REAL curriculum stages: which ran, what they returned, and why
the expansion stage refuses. Also track each landFactory's is_built over time.

The reimplemented-gate probe said ``no_combat_factory`` on 1612/1973 decisions,
i.e. no *built* non-command-centre factory with a real combat menu. Since the
army still grew, factories must be built only in short windows. This probe
records the truth from the engine stream instead of guessing.
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
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories
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
    env = RwppDuelEnv(game, port=27991, seed=71000, episode_prefix="stagediag",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=300.0, army_cap=40))

    stage_calls: Counter[str] = Counter()
    stage_orders: Counter[str] = Counter()
    factory_states: Counter[str] = Counter()

    orig = {}
    for name in ("_issue_upgrades", "_issue_opening_extractor", "_issue_opening_factory",
                 "_issue_opening_production", "_issue_late_expansion", "_issue_attack_drive"):
        orig[name] = getattr(env, name)

        def make(n, fn):
            def wrapper():
                stage_calls[n] += 1
                out = fn()
                if out:
                    stage_orders[n] += 1
                return out
            return wrapper

        setattr(env, name, make(name, orig[name]))

    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step % 100 == 0:
                facs = env.state.get("factories") or []
                land = [f for f in facs if "landfactory" in str(f.get("unit_type", "")).lower()]
                desc = ",".join(
                    f"{f.get('entity_id')}:built={f.get('is_built')}:menu={len(f.get('can_produce') or [])}"
                    for f in land
                ) or "none"
                cf = len(_combat_factories(env.state))
                print(f"step={step:4d} tick={env.state.get('tick')} combat_factories={cf} land=[{desc}]", flush=True)
                factory_states[desc.split(":")[1] if land else "none"] += 1
            if t or tr:
                print(f"RESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
        print("\n=== stage calls / times-produced-an-order ===", flush=True)
        for name in orig:
            print(f"  {name:32s} calls={stage_calls[name]:5d} orders={stage_orders[name]:5d}", flush=True)
        print("\nfinal income:", (env.state.get("global_state") or {}).get("income"), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
