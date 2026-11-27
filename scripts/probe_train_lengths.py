"""Is episode_ticks=1200 the reason training sees only timeouts?

The long run's first update shows the shape of the problem:
  tick=1200, own units=3, enemy=4, funds=2390, income=26, extractors=0,
  factories=0, under_construction=true, result=timeout

At 1200 ticks the economy has not even finished the opening extractor. Every
episode is therefore a timeout at a point where neither side can win, so the
policy gets `timeout_penalty=-2.8` as its only terminal signal and no gradient
toward winning exists.

This compares training-relevant episode lengths on the SAME policy (uniform
random via noop curriculum), measuring how often a decisive result appears at
all. A length where *some* episodes decide is the minimum requirement for
learning; a length where none do cannot teach anything regardless of algorithm.
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
from ironzero.rwpp.env import RwppDuelEnv
from scripts.probe_lake_winrate import RushStrategy


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
        max_entities=8,
        units_per_team=8,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=port, seed=seed, episode_prefix=f"len{ticks}",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    peak = 0
    res = None
    ext = 0
    try:
        env.reset(seed=seed)
        for _ in range(ticks // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            own = env.state.get("own_entities") or []
            peak = max(peak, sum(1 for u in own if u.get("can_attack")
                                 and not u.get("is_builder") and not u.get("is_building")
                                 and float(u.get("hp") or 0) > 0))
            slots = env.state.get("resource_slots") or []
            ext = max(ext, sum(1 for s in slots if isinstance(s, dict) and s.get("occupied")))
            if t or tr:
                res = info.get("result")
                break
    finally:
        env.close()
    return {"res": res, "peak": peak, "ext": ext}


def main() -> int:
    # Ports 30500+ so this can run while a training job holds 30300-30400.
    port = 30500
    report = {}
    for ticks in (1200, 4000, 8000, 19674):
        rows = []
        for i in range(6):
            seed = 71000 + i
            r = run(seed, port, ticks)
            port += 1
            rows.append({"seed": seed, **r})
        decided = sum(1 for r in rows if r["res"] in ("victory", "defeat"))
        wins = sum(1 for r in rows if r["res"] == "victory")
        peaks = [r["peak"] for r in rows]
        exts = [r["ext"] for r in rows]
        report[str(ticks)] = {"rows": rows, "decided": decided, "wins": wins}
        print(f"ticks={ticks:6d} decided={decided}/6 wins={wins}/6 "
              f"peak={peaks} ext={exts}", flush=True)
    Path("runs/train_lengths.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
