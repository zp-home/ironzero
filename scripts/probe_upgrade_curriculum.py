"""Does the curriculum's own upgrade stage now complete a tier?

``probe_upgrade_functional`` proved the ENGINE accepts and completes an extractor
upgrade (extractorT1 -> extractorT2, income 26 -> 30) when the order goes through
``env.step``. The remaining question is whether ``_issue_upgrades`` -- the stage
that actually runs in training -- reaches the same outcome.

Rounds 19-20 concluded it never did, but those measurements predate the
``_upgrade_in_flight`` latch that stops the stage from re-issuing (and therefore
restarting) the upgrade it already has in flight.

This runs the real curriculum with upgrades enabled and reports the final
extractor tiers, the income rate, and how many upgrade orders were issued.
"""
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


def run(seed: int, port: int, upgrades: int) -> dict:
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
        max_entities=8,
        units_per_team=8,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=upgrades,
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="upgcur",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56, max_upgrades=upgrades))
    peak_income = 0.0
    final_income = 0.0
    res = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            income = float((env.state.get("global_state") or {}).get("income") or 0.0)
            peak_income = max(peak_income, income)
            final_income = income
            if t or tr:
                res = info.get("result")
                break
        own = env.state.get("own_entities") or []
        tiers = sorted(
            (str(u.get("unit_type")), u.get("tech_level"))
            for u in own if "xtractor" in str(u.get("unit_type", "")).lower()
        )
        issued = int(env._upgrades_issued)
    finally:
        env.close()
    return {"res": res, "tiers": tiers, "peak_income": peak_income,
            "final_income": final_income, "issued": issued}


def main() -> int:
    port = 31500
    for upgrades in (0, 4):
        rows = []
        for i in range(4):
            r = run(71000 + i, port, upgrades)
            port += 1
            rows.append(r)
        t2 = sum(1 for r in rows if any(t[0].endswith("T2") for t in r["tiers"]))
        print(f"max_upgrades={upgrades}: T2-present={t2}/4 "
              f"peak_income={[r['peak_income'] for r in rows]} "
              f"issued={[r['issued'] for r in rows]}", flush=True)
        for r in rows:
            print(f"   {r['res']:8s} tiers={r['tiers']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
