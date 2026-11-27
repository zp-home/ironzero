"""Validate that the bridge's observation JSON still parses after adding is_built.

The economy collapsed to a lone command centre after the bridge change adding
``is_built`` to factories. That is the signature of a malformed observation: if
``factoriesJson`` emits invalid JSON, the whole payload may be dropped or fields
shift. This checks that the running bridge produces a parseable observation with
sensible factory entries.

Usage::

    python scripts/probe_obs_valid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ironzero.rwpp.config import RwppGameConfig  # noqa: E402
from ironzero.rwpp.env import RwppDuelEnv  # noqa: E402
from ironzero.strategy.rule_strategy import RuleStrategy  # noqa: E402


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
    env = RwppDuelEnv(game, port=24_600, seed=24_600, episode_prefix="obsval", strategy=RuleStrategy())
    try:
        obs, info = env.reset()
        print("obs shape:", obs.shape if hasattr(obs, "shape") else type(obs))
        print("own:", len(env.state.get("own_entities") or []))
        for entry in env.state.get("factories") or []:
            if isinstance(entry, dict):
                print(
                    {
                        "id": entry.get("entity_id"),
                        "type": entry.get("unit_type"),
                        "is_built": entry.get("is_built"),
                        "menu": entry.get("can_produce"),
                    }
                )
        for step in range(500):
            _o, _r, term, trunc, _i = env.step(np.array([9, 0, 0, 0, 0]))
            if term or trunc:
                break
        print("after 500 decisions: own =", len(env.state.get("own_entities") or []))
        for entry in env.state.get("factories") or []:
            if isinstance(entry, dict):
                print(
                    {
                        "id": entry.get("entity_id"),
                        "type": entry.get("unit_type"),
                        "is_built": entry.get("is_built"),
                        "menu": entry.get("can_produce"),
                    }
                )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
