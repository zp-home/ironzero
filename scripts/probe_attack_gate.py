"""Does the attack stage actually fire? Trace its gates live, decision by decision.

The seven-stage curriculum builds a real base (8 structures, closest approach 80 px
in one probe) but the army still peaks at 16 against an enemy holding 30-38, and
every episode times out. ``_issue_attack_drive`` has three gates and only one of
them can explain a *stationary* army:

  * ``opening_attack`` config flag,
  * ``macro.allows("attack")`` -- the rule layer's plan,
  * ``len(ours) >= attack_at_army`` (with a 0.6 * cap plateau fallback).

This prints all three, plus the actual distance to the enemy, so "the army never
moved" can be attributed to a specific gate instead of guessed at.
"""

from __future__ import annotations

import json
import sys
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
        episode_ticks=19674,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=1,
        physics_ms=16,
        max_entities=8,
        units_per_team=8,
    )
    env = RwppDuelEnv(game, port=18_500, seed=96_000, episode_prefix="atk")
    rows: list[dict] = []
    try:
        env.reset()
        for step in range(900):
            if step % 60 == 0:
                own = env.state.get("own_entities") or []
                enemies = env.state.get("visible_enemy_entities") or []
                army = sum(
                    1
                    for u in own
                    if u.get("can_attack")
                    and not u.get("is_builder")
                    and not u.get("is_building")
                    and float(u.get("hp") or 0.0) > 0.0
                )
                closest = None
                for a in own:
                    for e in enemies:
                        try:
                            d = ((a["x"] - e["x"]) ** 2 + (a["y"] - e["y"]) ** 2) ** 0.5
                            closest = d if closest is None else min(closest, d)
                        except (KeyError, TypeError):
                            continue
                allowed = env.macro.allows("attack")
                gate = "unknown"
                if not getattr(game, "opening_attack", False):
                    gate = "opening_attack_off"
                elif not allowed:
                    gate = "macro_disallows"
                elif army < int(env.macro.attack_at_army):
                    cap = max(int(env.macro.army_cap), int(env.macro.attack_at_army))
                    if army < max(1, int(cap * 0.6)):
                        gate = f"army<{env.macro.attack_at_army} and <60%cap"
                    else:
                        gate = "plateau_fallback_should_fire"
                else:
                    gate = "SHOULD_FIRE"
                rows.append(
                    {
                        "step": step,
                        "tick": env.state.get("tick"),
                        "army": army,
                        "attack_at_army": int(env.macro.attack_at_army),
                        "army_cap": int(env.macro.army_cap),
                        "allows_attack": bool(allowed),
                        "stage_names": list(env.macro.stage_names),
                        "closest_px": round(closest, 1) if closest is not None else None,
                        "gate": gate,
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
        print(json.dumps(row, ensure_ascii=False))
    (ROOT / ".runtime" / "attack_gate_trace.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
