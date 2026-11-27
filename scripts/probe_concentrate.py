"""Concentrate the WHOLE army on one enemy factory each decision.

probe_reach showed an enemy landFactory reached 25 HP (from 1200) -- damage is
being dealt -- yet the army's closest approach stays 1167-1619 px and the other
factories are at full HP. So the force is split: group 4 is the broadcast group
in the bridge (GROUP_BROADCAST = GROUP_COUNT = 4, orders every living friendly
unit), while group 0 only addresses id%4==0 units.

Prior probes used group=4 for attack_target already, but the *curriculum's*
attack_move also fires every decision at a grid cell, which re-tasks the army
away from the target. Here we suppress the curriculum attack stage entirely so
the only order the army receives is "attack this factory".
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
from scripts.probe_target_production import pick_target


def run(seed: int, port: int, suppress_attack: bool) -> dict:
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
        opening_attack=not suppress_attack,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="conc",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    best_dist = None
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            target = pick_target(env)
            foe = env.state.get("visible_enemy_entities") or []
            if target is not None and step >= 4:
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
            if target is not None and com:
                cx = sum(float(u.get("x") or 0) for u in com) / len(com)
                cy = sum(float(u.get("y") or 0) for u in com) / len(com)
                d = math.hypot(float(target.get("x") or 0) - cx,
                               float(target.get("y") or 0) - cy)
                best_dist = d if best_dist is None else min(best_dist, d)
            if t or tr:
                res = info.get("result")
                break
        facs = [(u.get("unit_type"), round(float(u.get("hp") or 0)))
                for u in (env.state.get("visible_enemy_entities") or [])
                if u.get("is_building") and u.get("is_factory")]
    finally:
        env.close()
    return {"res": res, "peak": peak, "best_dist": round(best_dist or -1), "facs": facs}


def main() -> int:
    suppress = len(sys.argv) > 1 and sys.argv[1] == "suppress"
    n = 8
    wins = 0
    port = 28700
    rows = []
    for i in range(n):
        seed = 71000 + i
        r = run(seed, port, suppress)
        port += 1
        rows.append({"seed": seed, **r})
        if r["res"] == "victory":
            wins += 1
        print(f"[{i+1}/{n}] seed={seed} {r['res']:8s} peak={r['peak']:3d} "
              f"closest={r['best_dist']} facs={r['facs']}", flush=True)
    print(f"\n=== suppress_attack={suppress} d1 Lake: {wins}/{n} ===", flush=True)
    Path(f"runs/concentrate_{suppress}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
