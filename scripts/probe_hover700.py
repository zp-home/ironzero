"""Combine the two proven Round-13 fixes: hoverTank-only + reserve=700.

Fix A (round 12): factory menu death spiral -> economy reaches 2-3 extractors,
peak army 17-22 instead of the 6-9 plateau.
Fix B: hovering tank is the ONLY ground unit with can_attack_air=True and it
crosses Lake's water (verified: y=327-551 vs c_tank's y=1037-1450).

Prior hover runs (0/12) used reserve=300/250, i.e. the starved economy.
This combines both at difficulty 1 with pinned seeds, 16 episodes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def force_hover_only() -> None:
    for name in ("tank", "c_tank", "artillery", "c_artillery", "scout"):
        env_mod._UNIT_COMBAT_RANK[name] = 0
    env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
    env_mod._UNIT_PRICE["hoverTank"] = 400


def run(seed: int, port: int, reserve: float = 700.0) -> dict:
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="hover700",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=reserve,
                                            army_cap=40))
    peak = 0
    res = None
    extractors = 0
    income = None
    t0 = time.monotonic()
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
    return {"res": res, "peak": peak, "ext": extractors, "income": income,
            "s": round(time.monotonic() - t0, 1)}


def main() -> int:
    force_hover_only()
    n = 16
    wins = 0
    outcomes = []
    for i in range(n):
        seed = 71000 + i
        r = run(seed, 28070 + i)
        outcomes.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} "
              f"ext={r['ext']} income={r['income']} ({r['s']}s)", flush=True)
    print(f"\n=== hover-only + reserve700 difficulty=1 Lake: {wins}/{n} = {100*wins/n:.1f}% ===", flush=True)
    Path("runs/hover700_sweep.json").write_text(
        json.dumps({"wins": wins, "n": n, "outcomes": outcomes}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
