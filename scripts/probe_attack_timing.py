"""Sweep attack timing at reserve=700 (the config that first produced a win).

Now that the menu death-spiral is fixed, the economy reaches ext=3 income=42
and peak army 17-22. The dominance win needs enemyCombat <= ownCombat/3.
Earlier rounds attacked too late (enemy already fielded 6-10 lightGunships +
T2). This sweeps attack_at to find whether hitting BEFORE the T2/air wave
snowballs into dominance.
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


def run(seed: int, port: int, attack_at: int, reserve: float = 700.0) -> dict:
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
        economy_credit_reserve=reserve,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"atk{attack_at}",
                      strategy=RushStrategy(attack_at=attack_at, max_factories=2,
                                            max_extractors=6, reserve=reserve,
                                            army_cap=40))
    peak = 0
    res = None
    extractors = 0
    income = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            slots = env.state.get("resource_slots") or []
            extractors = max(extractors, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "ext": extractors, "income": income}


def main() -> int:
    port = 28130
    report = {}
    for attack_at in (4, 6, 8):
        wins = 0
        n = 8
        rows = []
        for i in range(n):
            seed = 71000 + i
            r = run(seed, port, attack_at)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        report[str(attack_at)] = {"wins": wins, "n": n, "rows": rows}
        peaks = [r["peak"] for r in rows]
        print(f"attack_at={attack_at}: wins={wins}/{n} peak={min(peaks)}-{max(peaks)} "
              f"ext={rows[0]['ext']} income={rows[0]['income']}", flush=True)
    Path("runs/attack_timing_sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
