"""Why does hoverTank production (rank 6) collapse expansion to ext=1 income=26
while the identical reserve=700 config with tank reaches ext=3 income=42?

Production is greedy argmax (value/credit), so hover_rank=6 yields ALL hoverTank
(not a mix). hoverTank costs 450 vs tank 350. Expansion needs funds >= 700 and
runs BEFORE production in the stage order (upgrade, extractor, expansion,
factory, production...), so a pure price difference should not matter -- unless
some other gate differs. This logs the expansion gate distribution for both
configs to find the real blocker.
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
    hover_rank = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    env_mod._UNIT_COMBAT_RANK["hoverTank"] = hover_rank

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
    env = RwppDuelEnv(game, port=28230, seed=71000, episode_prefix="gathr",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=40))
    gates: Counter[str] = Counter()
    orders: Counter[str] = Counter()
    orig = env._issue_late_expansion

    def wrap():
        out = orig()
        if out:
            orders["expansion"] += 1
        return out

    env._issue_late_expansion = wrap
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            co = getattr(env, "curriculum_order", None)
            if co:
                k = str(co.get("kind"))
                if k == "produce_unit":
                    k += f":{co.get('unit_type')}"
                orders[k] += 1
            state = env.state
            own = state.get("own_entities") or []
            if not _combat_factories(state):
                gates["no_combat_factory"] += 1
            else:
                slots = state.get("resource_slots") or []
                occ = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
                if occ >= 6:
                    gates["cap"] += 1
                else:
                    funds = float((state.get("global_state") or {}).get("funds") or 0)
                    if funds < 700.0:
                        gates["poor"] += 1
                    else:
                        builder = next((u for u in own if u.get("is_builder")), None)
                        if builder is None:
                            gates["no_builder"] += 1
                        else:
                            # how many structures are mid-build (0<p<1)?
                            mid = [u for u in own if u.get("is_building")
                                   and isinstance(u.get("build_progress"), (int, float))
                                   and 0.0 < float(u.get("build_progress")) < 1.0
                                   and not _is_operational(u)]
                            if mid:
                                gates[f"mid_build:{dict(Counter(str(u.get('unit_type')) for u in mid))}"] += 1
                            else:
                                gates["OPEN"] += 1
            if step in (500, 900, 1300):
                print(f"step={step} tick={env.state.get('tick')} "
                      f"funds={(state.get('global_state') or {}).get('funds')} "
                      f"income={(state.get('global_state') or {}).get('income')}", flush=True)
            if t or tr:
                break
        print(f"\nhover_rank={hover_rank}", flush=True)
        print("gates:", dict(gates.most_common(10)), flush=True)
        print("orders:", dict(orders.most_common(12)), flush=True)
        print("factories:", [(f.get('unit_type'), f.get('is_built'), f.get('can_produce'))
                             for f in (env.state.get('factories') or [])], flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
