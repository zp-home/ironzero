# SPDX-License-Identifier: Apache-2.0
"""Optional strategic layer (LLM or rule based) for :class:`RwppDuelEnv`.

The strategy layer owns two things and neither of them is a unit order:

* a **directive** whose target point becomes a potential term in the reward
  (doc 12.4 / stage 5 -- the commander never emits low-level actions);
* a **macro plan** (:class:`~ironzero.strategy.directive.MacroPlan`) that the
  build stages execute: which economy goals run, in what priority, within what
  budgets and when the march begins.

Before the macro plan existed the second half was hard-coded in ``env.py`` and no
directive could change it, so "the commander" could not touch the economy at all.
The plan is resolved through :mod:`ironzero.rwpp.macro`, which also guarantees the
no-directive case is bit-identical to the configured behaviour.

These are free functions taking the environment as ``env`` so they can live
outside the (already long) ``env.py``; :class:`RwppDuelEnv` still exposes the
historical method names ``_strategy_summary`` / ``_refresh_strategy`` /
``_strategy_potential`` as thin delegates. Any failure inside them must degrade to
"no shaping, default economy" instead of breaking training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .macro import economy_summary, resolve_macro

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RwppDuelEnv


def strategy_summary(env: "RwppDuelEnv") -> dict[str, Any]:
    """Everything the strategist is allowed to see, as a plain mapping.

    Deliberately *not* the observation tensor: the commander reasons about
    aggregates (counts, a bank, a centroid), and the two consumers want different
    things. Note what is **absent**: no entity ids, no target slots, no action
    indices. A strategist that can name an entity id can start emitting unit
    orders, which is the one thing doc 12.4 forbids.

    ``economy`` is the macro block. It is what makes an economy plan possible at
    all -- without it the strategist had unit positions and nothing else, so
    "plan the build order" was guesswork.
    """

    own = env.state.get("own_entities", [])
    enemies = env.state.get("visible_enemy_entities", [])
    macro_state = economy_summary(env.state)
    env.macro_state = macro_state
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
        "economy": macro_state,
    }


def refresh_strategy(env: "RwppDuelEnv") -> None:
    """Ask the strategist for a new directive and resolve its macro plan.

    Two independent failures are contained here:

    * a strategist that raises must not break the episode -- the directive is
      cleared, which makes the reward term 0 and (via :func:`resolve_macro`) the
      economy plan the configured default;
    * a directive without a usable macro plan (an old object, a broken payload)
      resolves to the same configured default rather than an exception.
    """

    if env.strategy is not None:
        propose = getattr(env.strategy, "propose", None)
        if propose is not None:
            try:
                env.directive = propose(env._strategy_summary())
            except Exception:  # noqa: BLE001 - strategy must never break training
                env.directive = None
    env.macro = resolve_macro(env.game, env.directive)


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
