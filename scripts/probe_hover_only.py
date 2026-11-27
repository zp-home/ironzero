"""Decisive test: hoverTank-only rush on Lake at difficulty 1.

Rationale (unverified assumption this round attacks):
  * c_tank movement_type=LAND. Lake has water between the bases; the army in
    probe_battle_diag stalled at y~1037-1450 while the enemy CC sits at y=210,
    i.e. at the waterline. If LAND cannot cross, c_tank can NEVER reach the
    enemy CC regardless of army size -- which would explain 12 rounds of 0%.
  * hoverTank movement_type=HOVER and can_attack_air=True: it crosses water AND
    answers the enemy lightGunships that c_tank cannot shoot.

Prior hover probes failed to actually produce hoverTanks because the curriculum
scored tank above it once funds dipped. Here we remove tank/c_tank/artillery
from the rank table entirely, so hoverTank is the only producible combat unit,
and we sweep extractor count + attack threshold.
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
    """Make hoverTank the only affordable/valuable combat unit."""
    for name in ("tank", "c_tank", "artillery", "c_artillery", "scout"):
        env_mod._UNIT_COMBAT_RANK[name] = 0
    env_mod._UNIT_COMBAT_RANK["hoverTank"] = 8
    env_mod._UNIT_PRICE["hoverTank"] = 400


def run(seed: int, port: int, *, extractors: int, factories: int,
        attack_at: int, reserve: float, army_cap: int = 40) -> dict:
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
        economy_max_extractors=extractors,
        economy_max_factories=factories,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=reserve,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(
        game, port=port, seed=seed, episode_prefix="honly",
        strategy=RushStrategy(attack_at=attack_at, max_factories=factories,
                              max_extractors=extractors, reserve=reserve,
                              army_cap=army_cap),
    )
    t0 = time.monotonic()
    peak = 0
    types: dict[str, int] = {}
    min_y = None
    res = None
    try:
        env.reset(seed=seed)
        for _ in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            com = [u for u in own
                   if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            for u in com:
                # Track how far north the army ever got (enemy CC is at y~210).
                y = float(u.get("y") or 0)
                min_y = y if min_y is None else min(min_y, y)
            if t or tr:
                res = info.get("result")
                break
        for u in env.state.get("own_entities") or []:
            n = str(u.get("unit_type"))
            types[n] = types.get(n, 0) + 1
    finally:
        env.close()
    return {
        "seed": seed, "res": res, "peak": peak,
        "min_y": None if min_y is None else round(min_y),
        "types": types, "s": round(time.monotonic() - t0, 1),
        "cfg": f"ext={extractors},fac={factories},atk={attack_at},res={reserve}",
    }


def main() -> int:
    force_hover_only()
    configs = [
        dict(extractors=4, factories=2, attack_at=6, reserve=300.0),
        dict(extractors=6, factories=2, attack_at=8, reserve=300.0),
        dict(extractors=6, factories=3, attack_at=10, reserve=250.0),
    ]
    port = 27900
    for cfg in configs:
        wins = 0
        print(f"\n### {cfg}", flush=True)
        for i in range(4):
            r = run(71000 + i, port, **cfg)
            port += 1
            if r["res"] == "victory":
                wins += 1
            print(json.dumps(r), flush=True)
        print(f"### wins {wins}/4  {cfg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
