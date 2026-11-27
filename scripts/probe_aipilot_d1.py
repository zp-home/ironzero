"""Measure what the built-in AI pilot achieves vs difficulty 1 on Lake."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402


def main() -> int:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=30000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=1,
        physics_ms=16,
        max_entities=8,
        units_per_team=8,
        ai_pilot=True,
    )
    env = RwppDuelEnv(game, port=25_500, seed=25_500, episode_prefix="aipilot_d1")
    res = None
    try:
        env.reset()
        for step in range(3200):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step % 300 == 0 or t or tr:
                g = env.state.get("global_state") or {}
                own = env.state.get("own_entities") or []
                foes = env.state.get("visible_enemy_entities") or []
                own_types = Counter(str(u.get("unit_type") or "?") for u in own)
                foe_types = Counter(str(u.get("unit_type") or "?") for u in foes)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "tick": env.state.get("tick"),
                            "income": g.get("income"),
                            "funds": round(float(g.get("funds") or 0), 1),
                            "own_n": len(own),
                            "foe_n": len(foes),
                            "own_types": dict(own_types),
                            "foe_types": dict(foe_types),
                        }
                    )
                )
            if t or tr:
                res = info.get("result")
                print(f"DONE step={step} tick={env.state.get('tick')} result={res}")
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
