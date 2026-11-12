# SPDX-License-Identifier: Apache-2.0
"""Optional strategic layer (LLM or rule based) for :class:`RwppDuelEnv`.

The strategy layer only nudges reward shaping: it produces a directive whose
target point is turned into a potential term. It never emits low-level actions
(doc 12.4 / stage 5), and any failure inside it must degrade to "no shaping"
instead of breaking training.

These are free functions taking the environment as ``env`` so they can live
outside the (already long) ``env.py``; :class:`RwppDuelEnv` still exposes the
historical method names ``_strategy_summary`` / ``_refresh_strategy`` /
``_strategy_potential`` as thin delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RwppDuelEnv


def strategy_summary(env: "RwppDuelEnv") -> dict[str, Any]:
    own = env.state.get("own_entities", [])
    enemies = env.state.get("visible_enemy_entities", [])
    return {
        "tick": int(env.state.get("tick", 0)),
        "own_units": [
            {
                "x": float(unit.get("x") or 0.0),
                "y": float(unit.get("y") or 0.0),
                "hp": float(unit.get("hp") or 0.0),
                "max_hp": float(unit.get("max_hp") or 0.0),
            }
            for unit in own
        ],
        "enemy_units": [
            {
                "x": float(unit.get("x") or 0.0),
                "y": float(unit.get("y") or 0.0),
                "hp": float(unit.get("hp") or 0.0),
                "max_hp": float(unit.get("max_hp") or 0.0),
            }
            for unit in enemies
        ],
        "map": {"width": float(env.micro.map_width), "height": float(env.micro.map_height)},
    }


def refresh_strategy(env: "RwppDuelEnv") -> None:
    if env.strategy is None:
        return
    propose = getattr(env.strategy, "propose", None)
    if propose is None:
        return
    try:
        env.directive = propose(env._strategy_summary())
    except Exception:  # noqa: BLE001 - strategy must never break training
        env.directive = None


def strategy_potential(env: "RwppDuelEnv") -> float | None:
    """Progress towards the directive's target point, normalised to map size."""

    directive = env.directive
    if directive is None:
        return None
    target = None
    region = getattr(directive, "attack_region", None)
    rally = getattr(directive, "rally_point", None)
    intent = str(getattr(directive, "intent", ""))
    if intent in {"push", "harass"} and region:
        target = (float(region[0]), float(region[1]))
    elif rally:
        target = (float(rally[0]), float(rally[1]))
    elif region:
        target = (float(region[0]), float(region[1]))
    own = env.state.get("own_entities", [])
    if target is None or not own:
        return None
    cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
    cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
    diagonal = float(np.hypot(env.micro.map_width, env.micro.map_height)) or 1.0
    return -float(np.hypot(target[0] - cx, target[1] - cy)) / diagonal


__all__ = ["refresh_strategy", "strategy_potential", "strategy_summary"]
