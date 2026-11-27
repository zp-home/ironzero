"""Instrument every gate of _issue_late_expansion to find why only 1 extractor.

Income stuck at 26.0 == one T1 extractor for the whole episode in every config
tried so far. The expansion stage is the only thing that raises the income
ceiling, so this probe re-implements its gate order and records the FIRST gate
that refuses, per decision.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories, _is_operational
from scripts.probe_lake_winrate import RushStrategy


def gate_reason(env) -> str:
    state = env.state
    own = state.get("own_entities") or []
    if not _combat_factories(state):
        return "no_combat_factory"
    slots = state.get("resource_slots") or []
    free = [i for i, s in enumerate(slots) if isinstance(s, dict) and not s.get("occupied")]
    if not free:
        return "no_free_pool"
    busy = [
        u for u in own
        if u.get("is_building")
        and isinstance(u.get("build_progress"), (int, float))
        and 0.0 < float(u.get("build_progress")) < 1.0
        and not _is_operational(u)
    ]
    if busy:
        names = Counter(str(u.get("unit_type")) for u in busy)
        return f"unfinished_structure:{dict(names)}"
    if not any(u.get("is_builder") and float(u.get("hp") or 0) > 0 for u in own):
        return "no_builder"
    funds = float((state.get("global_state") or {}).get("funds") or 0.0)
    if funds < 700.0:
        return f"poor:{int(funds)}"
    occupied = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
    if occupied >= int(env.macro.max_extractors):
        return "cap_reached"
    return "SHOULD_FIRE"


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
        frame_ms=0,
        physics_ms=16,
        lockstep=True,
        max_entities=16,
        units_per_team=16,
        economy_army_cap=40,
        economy_army_target=24,
        economy_max_extractors=6,
        economy_max_factories=2,
        economy_max_turrets=0,
        economy_max_upgrades=0,
        economy_credit_reserve=300.0,
        economy_production_queue_depth=4,
    )
    env = RwppDuelEnv(game, port=27990, seed=71000, episode_prefix="expdiag",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=300.0, army_cap=40))
    reasons: Counter[str] = Counter()
    expansions = 0
    prev_occ = None
    try:
        env.reset(seed=71000)
        for step in range(19674 // 10 + 60):
            _o, _r, t, tr, info = env.step(np.array([9, 0, 0, 0, 0]))
            reasons[gate_reason(env)] += 1
            slots = env.state.get("resource_slots") or []
            occ = sum(1 for s in slots if isinstance(s, dict) and s.get("occupied"))
            if prev_occ is not None and occ > prev_occ:
                expansions += 1
                print(f"  step={step} tick={env.state.get('tick')} extractors {prev_occ}->{occ}", flush=True)
            prev_occ = occ
            if t or tr:
                print(f"RESULT={info.get('result')} tick={env.state.get('tick')}", flush=True)
                break
        print(f"\nextractor additions: {expansions}", flush=True)
        print("final income:", (env.state.get("global_state") or {}).get("income"), flush=True)
        print("\ngate reasons (top 12):", flush=True)
        for r, c in reasons.most_common(12):
            print(f"  {c:6d}  {r}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
