"""Compare curriculum battle outcomes across 2p maps and unit compositions at difficulty 1."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.strategy.rule_strategy import RuleStrategy


def run_map(map_rel: str, name: str, w: float, h: float, port: int, ticks: int = 19674) -> dict:
    game = RwppGameConfig(
        map_relative=map_rel,
        map_name=name,
        map_width_px=w,
        map_height_px=h,
        episode_ticks=ticks,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=20,
        economy_max_extractors=9,
        economy_max_factories=3,
        economy_production_queue_depth=3,
    )
    env = RwppDuelEnv(game, port=port, seed=port, episode_prefix="mapprobe", strategy=RuleStrategy())
    t0 = time.monotonic()
    res = None
    peak_army = 0
    best_income = 0.0
    try:
        env.reset()
        for _ in range(ticks // 10 + 50):
            own = env.state.get("own_entities") or []
            army = sum(
                1 for u in own
                if u.get("can_attack") and not u.get("is_builder")
                and not u.get("is_building") and float(u.get("hp") or 0) > 0
            )
            peak_army = max(peak_army, army)
            best_income = max(best_income, float((env.state.get("global_state") or {}).get("income") or 0))
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if t or tr:
                res = info.get("result")
                break
        own_types = Counter(str(u.get("unit_type") or "?") for u in (env.state.get("own_entities") or []))
        foe_types = Counter(str(u.get("unit_type") or "?") for u in (env.state.get("visible_enemy_entities") or []))
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "map": name,
        "result": res,
        "tick": env.state.get("tick"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "peak_army": peak_army,
        "best_income": best_income,
        "own_end": dict(own_types),
        "foe_end": dict(foe_types),
    }


def main() -> int:
    maps = [
        ("maps/skirmish/[p2]Lake (2p).tmx", "Lake", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Two_cold_sides (2p).tmx", "Two_cold_sides", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Hills_(2p)_[By Tstis & KPSS].tmx", "Hills", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Dire_Straight (2p) [by uber].tmx", "Dire_Straight", 2600.0, 2600.0),
        ("maps/skirmish/[p2]Fire Bridge (2p) [by uber].tmx", "Fire_Bridge", 2600.0, 2600.0),
    ]
    for i, (mrel, name, w, h) in enumerate(maps):
        out = run_map(mrel, name, w, h, 26_200 + i * 20)
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
