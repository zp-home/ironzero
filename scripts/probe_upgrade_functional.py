"""Functional test: does the extractor tier upgrade actually raise income?

Rounds 19-20 concluded "the upgrade is charged but never performed" from the
``_send_direct_order`` path. This probe shows that path does nothing at all
(no charge, no queue progress), while ``env.step`` with kind=upgrade_building
charges exactly 1400 and advances ``queue_progress`` to ~0.95.

What is still unproven is the *outcome*: does the tier actually go up? The
honest measurement is income, not ``tech_level`` -- a T2 extractor earns 18/s
against 8/s for T1 (MicroDuelConfig), so a completed upgrade is a visible jump
in the income rate.

This runs two arms with identical seeds and economy, differing only in whether
the upgrade is dispatched, and reports the income trajectory and the extractor
entity types at the end.
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
        episode_ticks=14000,
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
        game, port=port, seed=71001, episode_prefix="upgfn",
        strategy=RushStrategy(attack_at=8, max_factories=0, max_extractors=1,
                              reserve=2000.0, army_cap=0, max_upgrades=0),
    )


def run(port: int, do_upgrade: bool) -> dict:
    noop = np.array([9, 0, 0, 0, 0], dtype=np.int64)
    upgrade = np.array(
        [ACTION_KINDS.index("upgrade_building"), 4, 4, 4, 0], dtype=np.int64
    )
    env = make_env(port)
    incomes: list[float] = []
    fired = False
    try:
        env.reset(seed=71001)
        for step in range(1400):
            action = noop
            own = env.state.get("own_entities") or []
            ready = [
                u for u in own
                if "xtractor" in str(u.get("unit_type", "")).lower()
                and u.get("is_operational") and u.get("can_upgrade")
            ]
            # Fire once, as soon as an upgradable extractor exists and the bank
            # can pay for it (the engine needs the full 1400 at dispatch time).
            gs = env.state.get("global_state") or {}
            funds = float(gs.get("funds") or 0.0)
            if do_upgrade and not fired and ready and funds >= 1400.0:
                action = upgrade
                fired = True
            _o, _r, t, tr, _info = env.step(action)
            gs = env.state.get("global_state") or {}
            incomes.append(float(gs.get("income") or 0.0))
            if t or tr:
                break
        own = env.state.get("own_entities") or []
        types = sorted(
            (str(u.get("unit_type")), u.get("tech_level"))
            for u in own if "xtractor" in str(u.get("unit_type", "")).lower()
        )
    finally:
        env.close()
    return {
        "fired": fired,
        "peak_income": max(incomes) if incomes else 0.0,
        "final_income": incomes[-1] if incomes else 0.0,
        "extractor_types": types,
        "income_tail": incomes[-5:],
    }


def main() -> int:
    for do_upgrade in (False, True):
        result = run(31420 if not do_upgrade else 31430, do_upgrade)
        label = "with upgrade" if do_upgrade else "control      "
        print(f"{label}: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
