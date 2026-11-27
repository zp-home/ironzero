"""Test the previously-disabled upgrade stage at difficulty 1.

Every probe from rounds 10-17 passed economy_max_upgrades=0. The config comment
says 'T1 extractor -> T2 is the single most valuable investment' and the enemy
roster consistently includes extractorT2, heavyInterceptor (shield 450) and
mechFactory units. We have an _issue_upgrades stage that has never been enabled
in any of these measurements.

This enables upgrades (economy_upgrade=True, economy_max_upgrades=2/4) and measures
whether tech_level actually rises, income grows, and difficulty-1 win rate moves.
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
from ironzero.rwpp.decoding import ACTION_KINDS
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
        max_entities=16,
        units_per_team=16,
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"upg{upgrades}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56, max_upgrades=upgrades))
    peak = 0
    res = None
    max_tech = 1
    upg_orders = 0
    income = None
    ext = 0
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            co = getattr(env, "curriculum_order", None)
            if co and co.get("kind") == "upgrade_building":
                upg_orders += 1
            own = env.state.get("own_entities") or []
            combat = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                      and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(combat))
            for u in own:
                if u.get("is_building"):
                    try:
                        max_tech = max(max_tech, int(u.get("tech_level") or 1))
                    except (TypeError, ValueError):
                        pass
            slots = env.state.get("resource_slots") or []
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "max_tech": max_tech,
            "upg_orders": upg_orders, "income": income, "ext": ext,
            "issued": getattr(env, "_upgrades_issued", None)}


def main() -> int:
    port = 29200
    for upgrades in (0, 2, 4):
        rows = []
        wins = 0
        for i in range(6):
            seed = 71000 + i
            r = run(seed, port, upgrades)
            port += 1
            rows.append({"seed": seed, **r})
            wins += r["res"] == "victory"
        print(f"max_upgrades={upgrades}: wins={wins}/6 "
              f"tech={[r['max_tech'] for r in rows]} "
              f"orders={[r['upg_orders'] for r in rows]} "
              f"income={[r['income'] for r in rows]} "
              f"peak={[r['peak'] for r in rows]}", flush=True)
        Path(f"runs/upgrades_{upgrades}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
