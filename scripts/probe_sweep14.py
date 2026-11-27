"""Grid-sweep (credit_reserve x army_cap) after the round-13 economy fix.

Round 13 reached 4/16 (25% in one sample, 18.8% on re-run) at reserve=700 with
production protecting the expansion budget. But peak army fell 17-22 -> 13-16,
so the protection may be too strict: the closest loss (seed 71003) was 13 own vs
5 enemy combat, HP 2730 vs 350 -- the HP ratio (7.8x) already clears the 3x
dominance bar, only the ARMY ratio (2.6x vs 3x required) is short.

Hypothesis: a slightly higher army_cap / lower reserve lets the army reach ~17
while income still climbs, converting those near-misses into wins.

Pinned engine seeds (the round-11 fix makes --seed reach the bridge), 8 seeds per
cell to keep the sweep affordable.
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


def run(seed: int, port: int, reserve: float, army_cap: int,
        attack_at: int = 8) -> dict:
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
        economy_army_cap=army_cap,
        economy_army_target=min(24, army_cap),
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=reserve,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="sweep14",
                      strategy=RushStrategy(attack_at=attack_at, max_factories=2,
                                            max_extractors=6, reserve=reserve,
                                            army_cap=army_cap))
    peak = 0
    res = None
    ext = 0
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
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "ext": ext, "income": income}


def main() -> int:
    cells = [
        (700.0, 40),
        (700.0, 56),
        (500.0, 40),
        (900.0, 40),
    ]
    n = 8
    port = 28300
    report = []
    for reserve, army_cap in cells:
        rows = []
        wins = 0
        for i in range(n):
            seed = 71000 + i
            r = run(seed, port, reserve, army_cap)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        peaks = [r["peak"] for r in rows]
        exts = [r["ext"] for r in rows]
        report.append({"reserve": reserve, "army_cap": army_cap,
                       "wins": wins, "n": n, "rows": rows})
        print(f"reserve={reserve:5.0f} cap={army_cap:3d}  wins={wins}/{n}  "
              f"peak={min(peaks)}-{max(peaks)} ext={min(exts)}-{max(exts)}", flush=True)
    Path("runs/sweep14.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
