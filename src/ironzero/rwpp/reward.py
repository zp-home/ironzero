# SPDX-License-Identifier: Apache-2.0
"""Hp / distance potential terms used by the real-game reward shaping.

These are pure functions over a bridge state dict. They are the *input* to the
reward decomposition in :mod:`ironzero.rwpp.env`; the accumulation order of the
final reward lives there and must not change (floating-point sums are order
sensitive and the reward assertions in the tests depend on it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig


def hp_potential(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> float:
    """Normalised hp advantage in ``[-1, 1]`` (own minus enemy).

    Dead units disappear from the bridge stream, so the reference totals from
    the first observation of the episode are used when available; otherwise the
    alive-unit maximum is the denominator.
    """

    own, enemy = hp_fractions(state, own_max_ref, enemy_max_ref)
    return own - enemy


def hp_fractions(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> tuple[float, float]:
    """Return ``(own_hp_frac, enemy_hp_frac)`` from a raw state."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    own_hp = sum(float(unit.get("hp") or 0.0) for unit in own)
    enemy_hp = sum(float(unit.get("hp") or 0.0) for unit in enemies)
    own_max = sum(float(unit.get("max_hp") or 0.0) for unit in own) or 1.0
    enemy_max = sum(float(unit.get("max_hp") or 0.0) for unit in enemies) or 1.0
    own_ref = float(own_max_ref) if own_max_ref else own_max
    enemy_ref = float(enemy_max_ref) if enemy_max_ref else enemy_max
    # The AI may field reinforcements, so fractions are clamped to [0, 1].
    return (
        max(0.0, min(1.0, own_hp / max(1.0, own_ref))),
        max(0.0, min(1.0, enemy_hp / max(1.0, enemy_ref))),
    )


def army_hp_reference(
    state: dict[str, Any],
    config: MicroDuelConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """Episode-start army max-hp totals used to normalise hp fractions.

    This must run on the **ground-truth** state, never on the fogged policy
    view: the bridge streams every unit at reset, so the first frame carries the
    full army value even when ``vision_radius`` hides the enemies from the
    policy. Summing the visible enemy list instead would collapse the reference
    to a single unit (or to the configured fallback) at the start of a fogged
    episode and make every later hp fraction read as near-zero damage.

    ``max_hp`` falls back to ``config.unit_hp`` per entity; if the stream is
    empty the configured ``units_per_team * unit_hp`` is used.
    """

    expected = float(max(1, int(config.units_per_team)) * float(config.unit_hp))
    own_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("own_entities", [])
    )
    enemy_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("visible_enemy_entities", [])
    )
    own_ref = own_total if own_total > 0.0 else expected
    enemy_ref = enemy_total if enemy_total > 0.0 else expected
    return float(own_ref), float(enemy_ref)


def army_distance_norm(state: dict[str, Any], map_diagonal: float) -> float:
    """Normalised distance between the two army centroids (0 = same spot)."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    if not own or not enemies:
        return 0.0
    own_cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
    own_cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
    enemy_cx = sum(float(unit.get("x") or 0.0) for unit in enemies) / len(enemies)
    enemy_cy = sum(float(unit.get("y") or 0.0) for unit in enemies) / len(enemies)
    diagonal = max(1.0, float(map_diagonal))
    return min(1.0, float(np.hypot(enemy_cx - own_cx, enemy_cy - own_cy)) / diagonal)


__all__ = [
    "army_distance_norm",
    "army_hp_reference",
    "hp_fractions",
    "hp_potential",
]
