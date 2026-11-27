"""Inspect full unit tech tree (landFactory T1/T2, airFactory T1/T2) & Lake wave battle trace."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def main() -> int:
    # 1. Trace seed 71000 vs seed 26500 (from probe_fast_ground) on Lake to see WHY 71000 timed out
    for seed, atk_at, max_fact, ext, upg in [
        (71000, 6, 2, 4, 0),
        (71000, 12, 2, 6, 1),
        (71000, 16, 3, 6, 2),
    ]:
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
            economy_army_cap=32,
            economy_army_target=24,
            economy_max_extractors=ext,
            economy_max_factories=max_fact,
            economy_max_turrets=0,
            economy_max_upgrades=upg,
            economy_credit_reserve=350.0,
            economy_production_queue_depth=4,
        )
        env = RwppDuelEnv(
            game, port=26600 + atk_at, seed=seed, episode_prefix="trace",
            strategy=RushStrategy(attack_at=atk_at, max_factories=max_fact, max_extractors=ext, max_upgrades=upg),
        )
        try:
            env.reset(seed=seed)
            for step in range(2100):
                _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
                if step in (300, 600, 900, 1200, 1500, 1800) or t or tr:
                    own = env.state.get("own_entities") or []
                    foe = env.state.get("visible_enemy_entities") or []
                    from collections import Counter
                    oc = dict(Counter(str(u.get("unit_type")) for u in own))
                    fc = dict(Counter(str(u.get("unit_type")) for u in foe))
                    inc = (env.state.get("global_state") or {}).get("income")
                    print(f"[atk={atk_at} upg={upg}] step={step} tick={env.state.get('tick')} inc={inc} own={oc} foe={fc}", flush=True)
                    if step == 600:
                        for f in env.state.get("factories") or []:
                            print("  factory:", {k: f.get(k) for k in ("entity_id", "unit_type", "tech_level", "can_produce", "can_upgrade", "upgrade_cost")}, flush=True)
                        for b in env.state.get("builders") or []:
                            print("  builder:", {k: b.get(k) for k in ("entity_id", "unit_type", "can_build")}, flush=True)
                if t or tr:
                    print(f"[atk={atk_at} upg={upg}] DONE={info.get('result')}", flush=True)
                    break
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
