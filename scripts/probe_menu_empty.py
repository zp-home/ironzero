"""Why does a built landFactory's can_produce menu become empty at tick ~5011?

From probe_stage_instrument: factory 27 is_built=True from tick 2011, menu=5
(usable) until tick 4011, then menu=0 for the remaining 15000 ticks, so
_combat_factories() returns [] and production/expansion both stop.

Hypotheses:
  A. The engine genuinely cannot produce (no credits / tech lost / damaged).
  B. The bridge's can_produce probe is gated on something transient.
  C. The factory is queued-full: `can_produce` is computed only when idle.

This probe dumps the raw factory JSON plus the queue fields across the transition.
"""
from __future__ import annotations

import json
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
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=8000,
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
    env = RwppDuelEnv(game, port=27992, seed=71000, episode_prefix="menudiag",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=300.0, army_cap=40))
    try:
        env.reset(seed=71000)
        for step in range(700):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step in (200, 300, 400, 450, 480, 500, 520, 560, 600, 700):
                print(f"\n--- step={step} tick={env.state.get('tick')} funds={(env.state.get('global_state') or {}).get('funds')}", flush=True)
                for f in env.state.get("factories") or []:
                    print("  FACTORY:", json.dumps(f, ensure_ascii=False), flush=True)
                for u in env.state.get("own_entities") or []:
                    if u.get("is_building") and "factory" in str(u.get("unit_type", "")).lower():
                        print("  ENTITY:", {k: u.get(k) for k in
                              ("entity_id", "unit_type", "is_operational", "build_progress",
                               "tech_level", "hp", "max_hp", "can_upgrade", "queue_length",
                               "queue_progress")}, flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
