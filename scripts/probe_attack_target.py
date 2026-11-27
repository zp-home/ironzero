"""Trace where the army actually fights and whether enemy buildings lose HP.

The trained policy on Lake at difficulty 1 ends every episode with both sides'
``hp_frac = 1.0`` -- no enemy building loses any HP -- despite fielding 16-35 tanks
and closing to ~94 px. The attack stage targets the enemy command centre when
visible, so the question is what it targets the rest of the time and whether the
army even receives the order.

This runs with the trained checkpoint's *noop* equivalent (curriculum only) and
records, per sample:
  * the curriculum's attack order and its target grid cell,
  * which enemy buildings are visible,
  * the nearest distance from any of our units to an enemy building,
  * total enemy building HP (to see if anything is actually damaged).

Usage::

    python scripts/probe_attack_target.py [--decisions 1500]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


def enemy_building_hp(units: list[dict]) -> float:
    total = 0.0
    for u in units:
        if u.get("is_building") or u.get("is_factory"):
            total += float(u.get("hp") or 0.0)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1500)
    ap.add_argument("--port", type=int, default=24_900)
    args = ap.parse_args()

    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=1,
        physics_ms=16,
        max_entities=8,
        units_per_team=8,
        economy_army_cap=32,
        economy_army_target=20,
        economy_max_extractors=9,
        economy_max_factories=3,
        economy_production_queue_depth=3,
    )
    env = RwppDuelEnv(
        game, port=args.port, seed=args.port, episode_prefix="atktgt", strategy=RuleStrategy()
    )
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(args.decisions):
            if step % 100 == 0:
                state = env.state
                own = state.get("own_entities") or []
                foes = state.get("visible_enemy_entities") or []
                buildings = [u for u in foes if u.get("is_building") or u.get("is_factory")]
                closest = None
                for a in own:
                    if a.get("is_building") or a.get("is_builder"):
                        continue
                    for b in buildings:
                        try:
                            d = float(
                                np.hypot(
                                    float(a.get("x") or 0) - float(b.get("x") or 0),
                                    float(a.get("y") or 0) - float(b.get("y") or 0),
                                )
                            )
                            closest = d if closest is None else min(closest, d)
                        except (TypeError, ValueError):
                            continue
                order = getattr(env, "curriculum_order", None)
                rows.append(
                    {
                        "step": step,
                        "tick": state.get("tick"),
                        "army": sum(
                            1 for u in own
                            if u.get("can_attack") and not u.get("is_builder")
                            and not u.get("is_building") and float(u.get("hp") or 0) > 0
                        ),
                        "visible_buildings": len(buildings),
                        "building_hp": round(enemy_building_hp(buildings), 1),
                        "closest_to_building_px": round(closest, 1) if closest else None,
                        "curriculum": order.get("kind") if isinstance(order, dict) else None,
                    }
                )
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    for row in rows:
        print(json.dumps(row))
    (ROOT / ".runtime" / "attack_target.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
