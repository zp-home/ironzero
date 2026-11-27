"""Raise hoverTank's value rank above tank so production builds a MIX.

Evidence (runs/gap700.json, reserve=700, d1, 8 seeds): our end army is 8-13
units, ALL c_tank (can_attack_air=false); the enemy fields 5-11 units, mostly
air (lightGunship/c_interceptor/c_helicopter). We cannot kill air, so their
combat count never drops and dominance (ownCombat >= 3*enemyCombat) is
unreachable. Closest loss: seed 71003, 13 vs 5, HP 2730 vs 350 (HP ratio 7.8x
satisfied; army ratio 2.6x, just 2 survivors short).

hoverTank: hp 150, can_attack_air=TRUE, can_attack_land=TRUE, amphibious
(crosses Lake), price 450, built from the landFactory we already make.
Previous hover-only runs failed because tank was REMOVED (economy hurt) and
price was mis-set to 400. Here tank stays and only the rank changes.

value/credit: tank 4/350 = 0.0114; hoverTank needs rank/450 > 0.0114 -> rank 6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.env as env_mod
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def combat(entities) -> list:
    return [u for u in entities
            if u.get("can_attack") and not u.get("is_builder")
            and not u.get("is_building") and float(u.get("hp") or 0) > 0]


def run(seed: int, port: int, hover_rank: int) -> dict:
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"hr{hover_rank}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=40))
    peak = 0
    res = None
    air = 0
    ext = 0
    income = None
    types: dict[str, int] = {}
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = combat(env.state.get("own_entities") or [])
            peak = max(peak, len(own))
            air = max(air, sum(1 for u in own if u.get("can_attack_air")))
            slots = env.state.get("resource_slots") or []
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                income = (env.state.get("global_state") or {}).get("income")
                for u in combat(env.state.get("own_entities") or []):
                    n = str(u.get("unit_type"))
                    types[n] = types.get(n, 0) + 1
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "air": air, "ext": ext,
            "income": income, "types": types}


def main() -> int:
    hover_rank = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n = 16
    wins = 0
    rows = []
    port = 28200
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port + i, hover_rank)
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} air={r['air']} "
              f"ext={r['ext']} income={r['income']} types={r['types']}", flush=True)
    print(f"\n=== hover_rank={hover_rank} reserve700 d1 Lake: {wins}/{n} = {100*wins/n:.1f}% ===", flush=True)
    Path(f"runs/hover_rank{hover_rank}.json").write_text(
        json.dumps({"hover_rank": hover_rank, "wins": wins, "n": n, "rows": rows},
                   indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
