"""32-seed measurement of the best config found so far (reserve=700, cap=56).

Round 14 confirmed cap=56 beats cap=40 (2/16 vs 0/16) and cap=72 (1/16). Samples
of 8-16 swing widely (round 13: 4/16 then 3/16 on identical config), so this runs
32 pinned seeds to get a Wilson interval that can actually be quoted as the
current baseline.

Also records the end-of-episode combat ratio so the remaining gap to dominance
(3x army AND 3x HP) is visible per seed rather than only as a win/loss bit.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def combat(entities) -> list:
    return [u for u in entities
            if u.get("can_attack") and not u.get("is_builder")
            and not u.get("is_building") and float(u.get("hp") or 0) > 0]


def run(seed: int, port: int) -> dict:
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="base32",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    ext = 0
    income = None
    end_own = end_foe = 0
    end_own_hp = end_foe_hp = 0.0
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            oc = combat(env.state.get("own_entities") or [])
            peak = max(peak, len(oc))
            slots = env.state.get("resource_slots") or []
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                fc = combat(env.state.get("visible_enemy_entities") or [])
                end_own, end_foe = len(oc), len(fc)
                end_own_hp = sum(float(u.get("hp") or 0) for u in oc)
                end_foe_hp = sum(float(u.get("hp") or 0) for u in fc)
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "ext": ext, "income": income,
            "end_own": end_own, "end_foe": end_foe,
            "end_own_hp": round(end_own_hp), "end_foe_hp": round(end_foe_hp)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main() -> int:
    n = 32
    rows = []
    wins = 0
    port = 28400
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port + i)
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1:2d}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} "
              f"ext={r['ext']} inc={r['income']} end={r['end_own']}/{r['end_foe']} "
              f"hp={r['end_own_hp']}/{r['end_foe_hp']}", flush=True)
    lo, hi = wilson(wins, n)
    print(f"\n=== reserve=700 cap=56 d1 Lake: {wins}/{n} = {100*wins/n:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%) ===", flush=True)
    Path("runs/baseline32.json").write_text(
        json.dumps({"wins": wins, "n": n, "ci": [lo, hi], "rows": rows}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
