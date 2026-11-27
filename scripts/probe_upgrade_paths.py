"""Does an extractor upgrade actually complete to tech_level 2?

Round 19-20 concluded the upgrade queue item was "discarded": the bank dropped by
exactly the 1400 price but the extractor stayed extractorT1 / tech_level 1, and
``_send_direct_order`` produced no progress at all.

This probe isolates the two dispatch paths and watches ``queue_progress``:

  * ``env._send_direct_order({kind: upgrade_building, ...})`` -- the path used by
    the earlier "manual" experiments;
  * ``env.step(action)`` with kind=``upgrade_building`` -- the path the action
    space and therefore the policy actually use.

The earlier conclusion was drawn from the FIRST path only. If the second works,
the upgrade system was never broken and the real problem was elsewhere.
"""
from __future__ import annotations

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


def make_env(port: int) -> RwppDuelEnv:
    game = RwppGameConfig(
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_name="Lake",
        map_width_px=2600.0,
        map_height_px=2600.0,
        episode_ticks=12000,
        enemy_ai_difficulty=1,
        unlock_ai=True,
        win_mode="objective",
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        economy_max_extractors=1,
        economy_max_factories=0,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=2000.0,
    )
    return RwppDuelEnv(
        game, port=port, seed=71001, episode_prefix="upglong",
        strategy=RushStrategy(attack_at=8, max_factories=0, max_extractors=1,
                              reserve=2000.0, army_cap=0, max_upgrades=0),
    )


def snapshot(env, eid: int) -> str:
    own = env.state.get("own_entities") or []
    rows = [
        (u.get("unit_type"), u.get("tech_level"), u.get("queue_length"),
         round(float(u.get("queue_progress") or 0.0), 3))
        for u in own if u.get("entity_id") == eid
    ]
    gs = env.state.get("global_state") or {}
    return f"funds={gs.get('funds')} income={gs.get('income')} extractor={rows}"


def main() -> int:
    noop = np.array([9, 0, 0, 0, 0], dtype=np.int64)
    upgrade = np.array(
        [ACTION_KINDS.index("upgrade_building"), 4, 4, 4, 0], dtype=np.int64
    )

    for path in ("direct_order", "step_action"):
        env = make_env(31400 if path == "direct_order" else 31410)
        fired = False
        try:
            env.reset(seed=71001)
            for step in range(250):
                _o, _r, t, tr, _info = env.step(noop)
                own = env.state.get("own_entities") or []
                ready = [
                    u for u in own
                    if "xtractor" in str(u.get("unit_type", "")).lower()
                    and u.get("is_operational")
                ]
                if ready and not fired:
                    eid = int(ready[0]["entity_id"])
                    print(f"[{path}] fire at step={step} {snapshot(env, eid)}", flush=True)
                    if path == "direct_order":
                        env._send_direct_order(
                            {"kind": "upgrade_building", "building_entity_id": eid},
                            {"kind": "upgrade_building", "building_entity_id": eid},
                        )
                    else:
                        env.step(upgrade)
                    fired = True
                    for offset in range(400):
                        _o, _r, t3, tr3, _i3 = env.step(noop)
                        if offset % 20 == 0:
                            print(f"[{path}]   +{offset:3d} {snapshot(env, eid)}", flush=True)
                        if t3 or tr3:
                            break
                    break
                if t or tr:
                    break
        finally:
            env.close()
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
