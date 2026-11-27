"""Can we actually build more than 2 land factories?

The enemy ends with 4-6 factories (landFactory x2, airFactory x2, seaFactory)
against our 2, which is why killing one is instantly replaced. But the factory
stage has three separate ceilings that could each stop us at 2:

  * ``economy_max_factories`` quota (we have been passing 2);
  * the exact-coordinate candidate ladder (9 offsets from the base anchor) --
    every dead site is retired into ``_factory_dead_sites`` forever;
  * a per-episode order budget.

This sweeps the quota and reports how many operational landFactories actually
appear, plus the stall counters that explain any shortfall.
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


def run(seed: int, port: int, max_factories: int) -> dict:
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
        economy_max_factories=max_factories,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"fac{max_factories}",
                      strategy=RushStrategy(attack_at=8, max_factories=max_factories,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak_fac = 0
    peak_army = 0
    produced = 0
    seen: set[int] = set()
    res = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            land = [u for u in own
                    if str(u.get("unit_type", "")).lower() == "landfactory"
                    and u.get("is_operational")]
            peak_fac = max(peak_fac, len(land))
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak_army = max(peak_army, len(com))
            for u in com:
                if u.get("entity_id") is not None:
                    seen.add(int(u["entity_id"]))
            if t or tr:
                res = info.get("result")
                break
        produced = len(seen)
        dead = len(getattr(env, "_factory_dead_sites", set()) or set())
        offset = getattr(env, "_factory_candidate_offset", 0)
    finally:
        env.close()
    return {"res": res, "peak_factories": peak_fac, "peak_army": peak_army,
            "produced": produced, "dead_sites": dead, "candidate_offset": offset}


def main() -> int:
    port = 28800
    for quota in (2, 4, 6):
        rows = []
        for i in range(4):
            r = run(71000 + i, port, quota)
            port += 1
            rows.append({"seed": 71000 + i, **r})
        facs = [r["peak_factories"] for r in rows]
        arms = [r["peak_army"] for r in rows]
        prod = [r["produced"] for r in rows]
        dead = [r["dead_sites"] for r in rows]
        wins = sum(1 for r in rows if r["res"] == "victory")
        print(f"quota={quota}: wins={wins}/4 peak_factories={facs} "
              f"peak_army={arms} produced={prod} dead_sites={dead}", flush=True)
        Path(f"runs/quota_{quota}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
