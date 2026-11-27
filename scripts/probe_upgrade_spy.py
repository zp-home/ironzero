"""Spy on the exact payload that carries a curriculum upgrade to the bridge.

Rounds 25-26 show the upgrade stage charging 1400 credits and latching
``_upgrade_in_flight``, yet the bridge never logs ``upgrade dispatched`` or
``upgrade rejected`` for that building, and the extractor stays at tech_level 1.

Both of those log lines sit inside ``upgradeBuilding``, after its early guards:
the building must resolve, belong to the player, report a real
``getBuildActionId``, and expose a matching ``SpecialAction``. A silent exit
therefore means one of those guards fired -- or the order never reached Java at
all.

This wraps ``stepping.send_step`` so the *decoded* action list that actually goes
on the wire is captured, together with the outcome list the bridge sent back.
That distinguishes "never sent" from "sent and rejected" without guessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import ironzero.rwpp.stepping as stepping
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.strategy import RuleStrategy

CAPTURED: list[tuple[int, int, list, list]] = []
_orig_send = stepping.send_step


def spy_send(env, action, extra_actions=None):
    acts, disc = _orig_send(env, action, extra_actions)
    CAPTURED.append(
        (int(getattr(env, "episode_steps", 0)), int(getattr(env, "last_tick", 0)),
         list(acts), list(extra_actions or []))
    )
    return acts, disc


def main() -> int:
    stepping.send_step = spy_send
    import ironzero.rwpp.env as env_mod

    env_mod.send_step = spy_send

    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=4000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_upgrades=2,
        economy_credit_reserve=700.0,
    )
    env = RwppDuelEnv(game, port=31670, seed=71001, episode_prefix="spyupg",
                      strategy=RuleStrategy())
    noop = np.zeros(5, dtype=np.int64)
    sent_upgrade = 0
    try:
        env.reset()
        for step in range(400):
            _obs, _r, term, trunc, info = env.step(noop)
            if CAPTURED:
                acts = CAPTURED[-1][2]
                ups = [a for a in acts if isinstance(a, dict)
                       and a.get("kind") == "upgrade_building"]
                if ups:
                    sent_upgrade += 1
                    print(f"SENT step={step} tick={CAPTURED[-1][1]} "
                          f"upgrade={ups} all_acts={acts}", flush=True)
            outcomes = getattr(env, "last_action_outcomes", None)
            if outcomes and any(o for o in outcomes):
                print(f"OUTCOME step={step} tick={info.get('tick')} {outcomes}", flush=True)
            if term or trunc:
                break
        own = env.state.get("own_entities", [])
        print("FINAL extractors:", [
            (u.get("entity_id"), u.get("unit_type"), u.get("tech_level"),
             u.get("build_progress"), u.get("queue_length"))
            for u in own if "extractor" in str(u.get("unit_type"))
        ], flush=True)
        print(f"upgrade orders actually sent: {sent_upgrade}", flush=True)
        print(f"upgrades_issued counter: {env._upgrades_issued}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
