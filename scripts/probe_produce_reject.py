"""Why does the policy get PRODUCE_UNAVAILABLE 1233 times per update?

That bucket dominated every rejection in the round-24 training run:
  PRODUCE_UNAVAILABLE=1233  BUILD_UNAVAILABLE=30  UNIT_NOT_OWNED=25

Every invalid action pays ``invalid_penalty``, so this is both wasted sample
budget and a misleading gradient: the mask says "this is legal" and the engine
says no. This probe walks the exact mismatch -- which (factory, unit) pairs the
mask advertises versus which ``produceUnit`` accepts -- and prints the first few
disagreements with the engine's reason.

It drives the env with the *policy's* own action distribution (uniform over the
unmasked legal set) so the sample of attempted produce actions matches what
training actually sees.
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
from ironzero.rwpp.env import RwppDuelEnv, _combat_factories
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
    env = RwppDuelEnv(game, port=30600, seed=71000, episode_prefix="prodrej",
                      strategy=RushStrategy(attack_at=8, max_factories=2,
                                            max_extractors=6, reserve=700.0,
                                            army_cap=56))
    rng = np.random.default_rng(7)
    masks_seen: Counter[str] = Counter()
    advertised: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    rejected_when: Counter[str] = Counter()
    try:
        env.reset(seed=71000)
        for step in range(1200):
            masks = env.action_masks()
            produce_mask = masks.get("kind")
            legal = np.flatnonzero(produce_mask) if produce_mask is not None else np.array([9])
            kind = int(rng.choice(legal))
            action = np.array([kind, 0, 0, 0, 0], dtype=np.int64)
            masks_seen[str(kind)] += 1
            snapshot = None
            if kind == 7:  # produce_unit
                facs = _combat_factories(env.state)
                for entry in facs:
                    for name in (entry.get("can_produce") or []):
                        advertised[str(name)] += 1
                # What the bank and queue looked like when the action was chosen.
                gs = env.state.get("global_state") or {}
                snapshot = (
                    f"funds={round(float(gs.get('funds') or 0))} "
                    f"facs={len(facs)} "
                    f"q={sum(int(f.get('queue_length') or 0) for f in facs)}"
                )
            before = len(env.receipts)
            _o, _r, t, tr, info = env.step(action)
            for r in env.receipts[before:]:
                if r.get("accepted", True):
                    continue
                reason = str(r.get("rejected_reason") or "UNKNOWN")
                rejected[reason] += 1
                if kind == 7 and snapshot:
                    rejected_when[f"{reason} | {snapshot}"] += 1
            if t or tr:
                break
    finally:
        env.close()
    print("kind samples:", dict(masks_seen.most_common()), flush=True)
    print("advertised units:", dict(advertised.most_common(8)), flush=True)
    print("rejected reasons:", dict(rejected.most_common(8)), flush=True)
    print("produce rejections in context:", flush=True)
    for k, v in rejected_when.most_common(8):
        print(f"   {v:4d} x {k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
