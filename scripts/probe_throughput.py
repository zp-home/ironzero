"""Did the round-14 second-factory fix actually raise production throughput?

Baseline32 shows peak army 14-19 but end army 5-14 -- the army decays because
reinforcement cannot keep up with losses. The second landFactory was stuck at
progress 0.0 for 17000 ticks; if the fix works it should now finish and double
the reinforcement rate.

Measure: number of landFactories operational over time, total units produced
(observed as cumulative new combat entities), and peak army.
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
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=28450, seed=71003, episode_prefix="thru",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    seen_ids: set[int] = set()
    peak = 0
    try:
        env.reset(seed=71003)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            for u in com:
                eid = u.get("entity_id")
                if eid is not None:
                    seen_ids.add(int(eid))
            if step % 200 == 0:
                facs = [u for u in own if "landfactory" in str(u.get("unit_type", "")).lower()]
                ops = sum(1 for f in facs if f.get("is_operational"))
                q = sum(int(f.get("queue_length") or 0) for f in own if f.get("is_factory"))
                print(f"step={step:4d} tick={env.state.get('tick')} factories={ops}/{len(facs)} "
                      f"army={len(com)} peak={peak} produced={len(seen_ids)} queue={q} "
                      f"funds={(env.state.get('global_state') or {}).get('funds')}", flush=True)
            if t or tr:
                print(f"RESULT={info.get('result')}", flush=True)
                break
        print(f"\ntotal distinct combat units ever seen: {len(seen_ids)}", flush=True)
        print(f"peak simultaneous army: {peak}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
