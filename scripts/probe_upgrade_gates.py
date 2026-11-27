"""Instrument why _issue_upgrades never fires even when enabled.

probe_upgrades showed max_upgrades=2/4 still produced upg_orders=0 and tech_level=1
on all entities. economy_upgrade defaults True. So a gate inside _issue_upgrades
is refusing every decision. This replicates the gates in order and reports the
first refusal, and separately checks what the bridge reports as upgradable.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.observation.menus import upgradable_buildings
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _is_operational
from scripts.probe_lake_winrate import RushStrategy


def main() -> int:
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
        max_entities=16,
        units_per_team=16,
        economy_army_cap=56,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=4,
        economy_max_builders=2,
        economy_credit_reserve=700.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=29250, seed=71000, episode_prefix="upgdiag",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    gates: Counter[str] = Counter()
    orig = env._issue_upgrades

    def wrap():
        out = orig()
        state = env.state
        own = state.get("own_entities") or []
        upgradable, _ = upgradable_buildings(state)
        if not upgradable:
            gates["no_upgradable_reported"] += 1
        elif any(not _is_operational(u) for u in upgradable):
            gates["upgradable_not_operational"] += 1
        elif any(u.get("is_building") and not _is_operational(u) for u in own):
            gates["builder_busy"] += 1
        else:
            up_ids = {u.get("entity_id") for u in upgradable}
            print(f"  UPGRADABLE AVAILABLE ids={up_ids} out={bool(out)} "
                  f"issued={env._upgrades_issued} funds="
                  f"{(state.get('global_state') or {}).get('funds')}", flush=True)
            gates["would_fire"] += 1
        return out

    env._issue_upgrades = wrap
    try:
        env.reset(seed=71000)
        for step in range(1200):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            if step % 300 == 0:
                own = env.state.get("own_entities") or []
                upg, _ = upgradable_buildings(env.state)
                print(f"step={step} tick={env.state.get('tick')} "
                      f"upgradable={len(upg)} issued={env._upgrades_issued} "
                      f"funds={(env.state.get('global_state') or {}).get('funds')}", flush=True)
            if t or tr:
                break
        print("\ngates:", dict(gates.most_common()), flush=True)
        own = env.state.get("own_entities") or []
        for u in own:
            if u.get("is_building"):
                print(f"  {u.get('unit_type'):20s} id={u.get('entity_id')} "
                      f"tech={u.get('tech_level')} can_upgrade={u.get('can_upgrade')} "
                      f"operational={u.get('is_operational')}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
