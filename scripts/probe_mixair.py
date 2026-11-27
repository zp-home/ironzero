"""tank (healthy economy) + c_helicopter (attacks air AND land) at reserve=700.

Victory log: dominance needs 3x army + 3x HP, i.e. we must clear the enemy's
combat units. Their core force is air (lightGunship, c_helicopter). c_tank
cannot attack air. c_helicopter: hp 150, can_attack_air=True, can_attack_land=True.

We keep tank as the affordable ground backbone (economy stays healthy at
reserve=700, measured ext=3 income=42) and force airFactory production so air
killers exist alongside ground.
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
from ironzero.observation.menus import build_menu
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def wait(env, n: int) -> bool:
    noop = np.array([ACTION_KINDS.index("noop"), 0, 0, 0, 0], dtype=np.int64)
    for _ in range(n):
        _, _, t, tr, _ = env.step(noop)
        if t or tr:
            return True
    return False


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
        economy_army_cap=40,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=3,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="mixair",
                      strategy=RushStrategy(attack_at=10, max_factories=3,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=40))
    peak = 0
    res = None
    extractors = 0
    income = None
    air = 0
    try:
        env.reset(seed=seed)
        # Build an airFactory early so helicopters exist alongside tanks.
        for _ in range(3):
            menu = build_menu(env.obs_state)
            if "airFactory" in menu:
                idx = menu.index("airFactory")
                env.step(np.array([ACTION_KINDS.index("build_structure"), 8,
                                   6, 4, idx], dtype=np.int64))
                if wait(env, 90):
                    break
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            air = max(air, sum(1 for u in com if u.get("is_air_unit")))
            slots = env.state.get("resource_slots") or []
            extractors = max(extractors, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "air": air, "ext": extractors, "income": income}


def main() -> int:
    n = 16
    wins = 0
    outcomes = []
    for i in range(n):
        seed = 71000 + i
        r = run(seed, 28110 + i)
        outcomes.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} air={r['air']} "
              f"ext={r['ext']} income={r['income']}", flush=True)
    print(f"\n=== tank+helicopter reserve700 d1 Lake: {wins}/{n} = {100*wins/n:.1f}% ===", flush=True)
    Path("runs/mixair_sweep.json").write_text(
        json.dumps({"wins": wins, "n": n, "outcomes": outcomes}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
