"""Test the new economy_max_builders option: does a second builder help?

Round 17 measured the enemy at 2-3 builders by tick 9011 while we were pinned at
1 forever (builder scores 0 in _UNIT_COMBAT_RANK, so production never chose it).
All construction -- extractors, factories, turrets -- therefore serialised behind
a single unit.

This sweeps economy_max_builders = 0 (old behaviour) / 2 / 3 at difficulty 1,
measuring builder count, extractors, income, peak army and win rate.
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


def run(seed: int, port: int, max_builders: int) -> dict:
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
        economy_max_builders=max_builders,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"bld{max_builders}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak_builders = peak_army = ext = 0
    income = None
    res = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            peak_builders = max(peak_builders, sum(
                1 for u in own if u.get("is_builder") and float(u.get("hp") or 0) > 0))
            peak_army = max(peak_army, sum(
                1 for u in own if u.get("can_attack") and not u.get("is_builder")
                and not u.get("is_building") and float(u.get("hp") or 0) > 0))
            slots = env.state.get("resource_slots") or []
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "builders": peak_builders, "peak_army": peak_army,
            "ext": ext, "income": income}


def main() -> int:
    port = 29050
    for quota in (0, 2, 3):
        rows = []
        wins = 0
        for i in range(6):
            seed = 71000 + i
            r = run(seed, port, quota)
            port += 1
            rows.append({"seed": seed, **r})
            wins += r["res"] == "victory"
        print(f"max_builders={quota}: wins={wins}/6 builders={[r['builders'] for r in rows]} "
              f"peak_army={[r['peak_army'] for r in rows]} "
              f"ext={[r['ext'] for r in rows]} inc={[r['income'] for r in rows]}", flush=True)
        Path(f"runs/builders_{quota}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
