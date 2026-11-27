"""All-in early rush: attack before the enemy can field T2.

The one difficulty-1 victory that was fast happened at tick 4717 (round 15,
probe_focus_weakest seed 71003) -- well before the enemy's T2 air force exists.
Every other configuration waits for army_target=24, by which point the enemy has
4-6 factories, shield-450 interceptors and mechFactory units, and the army
decays from peak 17-24 down to 5-14.

Hypothesis: waiting for a large army is the wrong shape. A smaller force that
arrives at tick 3000-5000 can kill factories before the enemy economy compounds.

Sweeps attack_at (army size gate) down to 2-4 and stops production spending on
anything but the cheapest unit, to maximise early tempo.
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
from ironzero.rwpp.decoding import ACTION_KINDS
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


def enemy_factories(env) -> list[dict]:
    return [u for u in (env.state.get("visible_enemy_entities") or [])
            if u.get("is_building") and u.get("is_factory")
            and "extractor" not in str(u.get("unit_type", "")).lower()
            and not u.get("is_turret") and float(u.get("hp") or 0) > 0]


def run(seed: int, port: int, attack_at: int) -> dict:
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
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
        opening_attack=False,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="allin",
                      strategy=RushStrategy(attack_at=attack_at, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    locked = None
    try:
        env.reset(seed=seed)
        for step in range(19674 // 10 + 60):
            facs = enemy_factories(env)
            ids = {int(u["entity_id"]) for u in facs if u.get("entity_id") is not None}
            if locked is not None and locked not in ids:
                locked = None
            if locked is None and facs:
                locked = int(min(facs, key=lambda u: float(u.get("hp") or 0))["entity_id"])
            own = env.state.get("own_entities") or []
            com = [u for u in own if u.get("can_attack") and not u.get("is_builder")
                   and not u.get("is_building") and float(u.get("hp") or 0) > 0]
            peak = max(peak, len(com))
            # Only march once the gate is met, and only if a target is visible.
            if locked is not None and len(com) >= attack_at:
                foe = env.state.get("visible_enemy_entities") or []
                try:
                    slot = next(i for i, u in enumerate(foe) if u.get("entity_id") == locked)
                except StopIteration:
                    slot = 0
                slot = min(slot, max(0, int(env.micro.target_slots) - 1))
                act = np.array([ACTION_KINDS.index("attack_target"), 4, 0, 0, slot],
                               dtype=np.int64)
            else:
                act = np.array([9, 0, 0, 0, 0], dtype=np.int64)
            _o, _r, t, tr, info = env.step(act)
            if t or tr:
                res = info.get("result")
                tick = env.state.get("tick")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "tick": tick}


def main() -> int:
    port = 29700
    for attack_at in (2, 3, 4):
        rows = []
        wins = 0
        for i in range(8):
            seed = 71000 + i
            r = run(seed, port, attack_at)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        ticks = [r["tick"] for r in rows if r["res"] == "victory"]
        print(f"attack_at={attack_at}: wins={wins}/8 "
              f"peak={[r['peak'] for r in rows]} victory_ticks={ticks}", flush=True)
        Path(f"runs/allin_{attack_at}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
