"""What does the enemy actually field at each difficulty, and can we ever reach it?

The measured picture at difficulty 1 is: we hold 19 units at the end and the enemy
holds 30-38, every game a timeout, no side ever decided. Two explanations fit and
they call for opposite plans:

  A. the enemy is a real army and we are simply out-produced -> the fix is economy;
  B. the enemy is a scout swarm that never attacks and we never attack either ->
     the fix is aggression and a reachable objective.

The distinguishing evidence is the **enemy unit composition** and **how close the
two armies ever get**. This prints, per difficulty, the enemy unit types seen, the
closest approach, and whether either side's building ever drops.

Usage::

    python scripts/probe_enemy_composition.py --difficulty 1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402

LAKE = "maps/skirmish/[p2]Lake (2p).tmx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--port", type=int, default=18_000)
    ap.add_argument("--unlock-ai", action="store_true", default=True)
    ap.add_argument("--no-unlock-ai", dest="unlock_ai", action="store_false")
    ap.add_argument("--win-mode", default="objective")
    args = ap.parse_args()

    game = RwppGameConfig(
        map_relative=LAKE,
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=args.difficulty,
        unlock_ai=args.unlock_ai,
        win_mode=args.win_mode,
        frame_ms=1,
        physics_ms=16,
        max_entities=8,
        units_per_team=8,
    )
    env = RwppDuelEnv(game, port=args.port, seed=93_000, episode_prefix="enemy")
    enemy_types: Counter[str] = Counter()
    own_types: Counter[str] = Counter()
    closest = 10**9
    enemy_buildings_last = None
    own_buildings_last = None
    try:
        env.reset()
        for _ in range(args.decisions):
            for u in env.state.get("visible_enemy_entities") or []:
                enemy_types[str(u.get("unit_type"))] += 1
            for u in env.state.get("own_entities") or []:
                own_types[str(u.get("unit_type"))] += 1
            # Closest approach between any own unit and any enemy unit.
            own_xy = [(u.get("x"), u.get("y")) for u in (env.state.get("own_entities") or [])]
            enemy_xy = [
                (u.get("x"), u.get("y"))
                for u in (env.state.get("visible_enemy_entities") or [])
            ]
            for ox, oy in own_xy:
                for ex, ey in enemy_xy:
                    if None in (ox, oy, ex, ey):
                        continue
                    d = ((ox - ex) ** 2 + (oy - ey) ** 2) ** 0.5
                    if d < closest:
                        closest = d
            enemy_buildings_last = [
                u.get("unit_type")
                for u in (env.state.get("visible_enemy_entities") or [])
                if u.get("is_building")
            ]
            own_buildings_last = [
                u.get("unit_type")
                for u in (env.state.get("own_entities") or [])
                if u.get("is_building")
            ]
            # A noop policy: whatever happens is the curriculum's doing.
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    report = {
        "difficulty": args.difficulty,
        "unlock_ai": args.unlock_ai,
        "win_mode": args.win_mode,
        "enemy_unit_frames": dict(enemy_types.most_common()),
        "own_unit_frames": dict(own_types.most_common()),
        "closest_approach_px": None if closest == 10**9 else round(closest, 1),
        "enemy_buildings_final": enemy_buildings_last,
        "own_buildings_final": own_buildings_last,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = ROOT / ".runtime" / f"enemy_composition_d{args.difficulty}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
