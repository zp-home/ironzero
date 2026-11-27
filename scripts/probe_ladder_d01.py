"""Difficulty ladder: how does the same config score at -1 / 0 / 1?

Baseline32 at difficulty 1 (1.4x enemy income + T2 units) = 0/32. The tactical
framework may be sound and simply out-economised. Testing the same config one
difficulty step down isolates "the framework works" from "the framework cannot
beat a 1.4x economy".

This is also the measurement the objective's ladder needs: the goal asks for
difficulty 1, so the ladder shows exactly what each step costs.
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


def run(seed: int, port: int, difficulty: int) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=difficulty,
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"lad{difficulty}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            if t or tr:
                res = info.get("result")
                tick = env.state.get("tick")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "tick": tick}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main() -> int:
    n = 12
    port = 28500
    report = {}
    for difficulty in (0, 1):
        rows = []
        wins = 0
        for i in range(n):
            seed = 71000 + i
            r = run(seed, port, difficulty)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        lo, hi = wilson(wins, n)
        report[str(difficulty)] = {"wins": wins, "n": n, "ci": [lo, hi], "rows": rows}
        print(f"difficulty={difficulty}: {wins}/{n} = {100*wins/n:.1f}% "
              f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)", flush=True)
    Path("runs/ladder_d01.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
