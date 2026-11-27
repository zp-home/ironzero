"""Attack the enemy's PRODUCTION (factories + builders), not the command centre.

All 14 prior rounds ordered the army at the enemy commandCenter (4000 HP, sitting
inside anti-air range). But the `objective` verdict is
    enemies.mainBuildings == 0 || !enemies.hasCapability
where mainBuildings counts *factory buildings that are not turrets or extractors*
and hasCapability is `factories > 0 || builders > 0` (IronZeroBridge.capabilityOf).

So victory is reachable by destroying the enemy's landFactory/airFactory
(1200/1000 HP -- a third of the CC) and their builders, WITHOUT ever touching
the command centre or having to win the air war.

This probe orders attack_target at the nearest enemy FACTORY, and falls back to
enemy builders. It measures whether the objective verdict fires.
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
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def pick_target(env) -> dict | None:
    """Nearest enemy factory building; then enemy builder; then any building."""
    foe = env.state.get("visible_enemy_entities") or []
    own = env.state.get("own_entities") or []
    com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
           and not u.get("is_building") and float(u.get("hp") or 0) > 0]
    if not com or not foe:
        return None
    cx = sum(float(u.get("x") or 0) for u in com) / len(com)
    cy = sum(float(u.get("y") or 0) for u in com) / len(com)

    def dist(u):
        return math.hypot(float(u.get("x") or 0) - cx, float(u.get("y") or 0) - cy)

    def alive(u):
        return float(u.get("hp") or 0) > 0

    factories = [u for u in foe if alive(u) and u.get("is_building")
                 and u.get("is_factory") and not u.get("is_turret")
                 and "extractor" not in str(u.get("unit_type", "")).lower()]
    if factories:
        return min(factories, key=dist)
    builders = [u for u in foe if alive(u) and u.get("is_builder")]
    if builders:
        return min(builders, key=dist)
    buildings = [u for u in foe if alive(u) and u.get("is_building")]
    if buildings:
        return min(buildings, key=dist)
    return None


def run(seed: int, port: int, mode: str) -> dict:
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"tgt_{mode}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            target = pick_target(env) if mode == "production" else None
            if target is not None and step >= 8:
                # attack_target indexes the *visible* enemy list.
                foe = env.state.get("visible_enemy_entities") or []
                try:
                    slot = next(i for i, u in enumerate(foe)
                                if u.get("entity_id") == target.get("entity_id"))
                except StopIteration:
                    slot = 0
                slot = min(slot, max(0, int(env.micro.target_slots) - 1))
                act = np.array([ACTION_KINDS.index("attack_target"), 4, 0, 0, slot],
                               dtype=np.int64)
            else:
                act = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            _o, _r, t, tr, info = env.step(act)
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


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "production"
    n = 8
    wins = 0
    port = 28600
    rows = []
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port, mode)
        port += 1
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} tick={r['tick']}", flush=True)
    print(f"\n=== mode={mode} d1 Lake: {wins}/{n} ===", flush=True)
    Path(f"runs/target_{mode}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
