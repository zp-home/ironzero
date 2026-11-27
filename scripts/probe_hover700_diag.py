"""Why does hover-only lock the economy at ext=1 income=26 (deterministically),
while the normal tank config reaches ext=3 income=42 at the same reserve=700?

Suspect: _UNIT_PRICE['hoverTank'] is set to 400 by the forced-hover probe while
the engine charges 900, so the production stage's `price > funds` gate and the
reserve accounting disagree with reality. Log funds trajectory, the gate
reasons for expansion, and which unit production actually ordered.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories, _is_operational
from scripts.probe_lake_winrate import RushStrategy


def main() -> int:
    hover_only = len(sys.argv) > 1 and sys.argv[1] == "hover"
    if hover_only:
        for name in ("tank", "c_tank", "artillery", "c_artillery", "scout"):
            env_mod._UNIT_COMBAT_RANK[name] = 0
        env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
        env_mod._UNIT_PRICE["hoverTank"] = 400

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
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=28090, seed=71000, episode_prefix="diag700",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0, army_cap=40))
    orders: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            co = getattr(env, "curriculum_order", None)
            if co:
                key = str(co.get("kind"))
                if key == "produce_unit":
                    key += f":{co.get('unit_type')}"
                if key == "build_extractor":
                    key += f":slot{co.get('slot_id')}"
                orders[key] += 1
            # expansion gate reason
            state = env.state
            own = state.get("own_entities") or []
            if not _combat_factories(state):
                gates["no_combat_factory"] += 1
            elif not _is_operational(next((u for u in own if u.get("is_builder")), {}) or {}):
                gates["no_builder"] += 1
            else:
                slots = state.get("resource_slots") or []
                occ = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
                if occ >= 6:
                    gates["cap"] += 1
                else:
                    funds = float((state.get("global_state") or {}).get("funds") or 0)
                    if funds < 700.0:
                        gates[f"poor<{int(funds)}"] += 1
                    else:
                        gates["open"] += 1
            if step in (100, 300, 500, 700, 900, 1100):
                funds = (env.state.get("global_state") or {}).get("funds")
                print(f"step={step} tick={env.state.get('tick')} funds={funds} "
                      f"income={(env.state.get('global_state') or {}).get('income')}", flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')}", flush=True)
                break
        print("\norders:", dict(orders.most_common(12)), flush=True)
        print("expansion gates (top):", dict(gates.most_common(8)), flush=True)
        print("engine menu:", [f.get("can_produce") for f in (env.state.get("factories") or [])], flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
