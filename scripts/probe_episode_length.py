"""Does a longer episode let the economy compound into a winnable army?

Every measurement since round 10 used episode_ticks=19674. But the win condition
is ``enemies.mainBuildings == 0``, which counts the command centre (4000 HP) plus
every landFactory/airFactory -- and the enemy rebuilds them. Killing that list
inside ~20000 ticks is a very different task from doing it inside 60000.

Rusted Warfare games are not short. The tick cap is ours, chosen for throughput,
and ``frame_ms=0`` + ``lockstep`` makes ticks cheap, so a longer cap costs wall
clock but changes nothing else. If the army keeps growing with income (which now
reaches 50-58), a longer episode is the one dimension that has never been varied.

Sweeps episode_ticks 19674 / 40000 / 60000 at difficulty 1 on Lake.
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


def run(seed: int, port: int, ticks: int) -> dict:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=ticks,
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"len{ticks}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    tick = None
    locked = None
    income = None
    try:
        env.reset(seed=seed)
        for _ in range(ticks // 10 + 80):
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
            if locked is not None:
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
                income = (env.state.get("global_state") or {}).get("income")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "tick": tick, "income": income}


def main() -> int:
    port = 30100
    for ticks in (19674, 40000, 60000):
        rows = []
        wins = 0
        for i in range(4):
            seed = 71000 + i
            r = run(seed, port, ticks)
            port += 1
            rows.append({"seed": seed, **r})
            if r["res"] == "victory":
                wins += 1
        print(f"episode_ticks={ticks}: wins={wins}/4 peak={[r['peak'] for r in rows]} "
              f"income={[r['income'] for r in rows]}", flush=True)
        Path(f"runs/len_{ticks}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
