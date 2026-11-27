"""Sweep all 9 stock 2p maps at difficulty 1 with the focused-production strategy.

Round 15 found that attacking the enemy commandCentre throws the army into its
anti-air net (peak 13-19), while targeting enemy factories doubles peak army to
24-30 and produced a tick-4717 victory at seed 71003 on Lake.

Lake's central water channel forces a land/air split. Other 2p maps have no such
choke point, so the same tactic may behave very differently. The objective asks
for >80% on original maps and "ideally multiple maps" -- this identifies which
maps are viable at difficulty 1 rather than assuming Lake is representative.

4 pinned seeds per map, difficulty 1, focused weakest-factory targeting.
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

RUNTIME_MAPS = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15\maps\skirmish")
MAPS = [
    "[p2]Beach landing (2p) [by hxyy].tmx",
    "[p2]Big Island (2p).tmx",
    "[p2]Dire_Straight (2p) [by uber].tmx",
    "[p2]Fire Bridge (2p) [by uber].tmx",
    "[p2]Hills_(2p)_[By Tstis & KPSS].tmx",
    "[p2]Ice Island (2p).tmx",
    "[p2]Lake (2p).tmx",
    "[p2]Small_Island (2p).tmx",
    "[p2]Two_cold_sides (2p).tmx",
]


def enemy_factories(env) -> list[dict]:
    return [u for u in (env.state.get("visible_enemy_entities") or [])
            if u.get("is_building") and u.get("is_factory")
            and "extractor" not in str(u.get("unit_type", "")).lower()
            and not u.get("is_turret") and float(u.get("hp") or 0) > 0]


def run(map_name: str, seed: int, port: int) -> dict:
    game = RwppGameConfig(
        map_relative=f"maps/skirmish/{map_name}",
        map_name=map_name,
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
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix="mapsweep",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
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
            own = env.state.get("own_entities") or []
            peak = max(peak, sum(1 for u in own if u.get("can_attack")
                                 and not u.get("is_builder") and not u.get("is_building")
                                 and float(u.get("hp") or 0) > 0))
            if t or tr:
                res = info.get("result")
                break
    except Exception as exc:
        return {"res": f"ERROR:{type(exc).__name__}:{exc}", "peak": peak}
    finally:
        env.close()
    return {"res": res, "peak": peak}


def main() -> int:
    n = 4
    port = 29100
    report = {}
    for map_name in MAPS:
        rows = []
        wins = 0
        for i in range(n):
            r = run(map_name, 71000 + i, port)
            port += 1
            rows.append({"seed": 71000 + i, **r})
            if r["res"] == "victory":
                wins += 1
        report[map_name] = {"wins": wins, "n": n, "rows": rows}
        peaks = [r["peak"] for r in rows]
        print(f"{map_name[:38]:38s} wins={wins}/{n} peak={peaks}", flush=True)
    Path("runs/mapsweep_d1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
