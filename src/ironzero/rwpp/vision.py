# SPDX-License-Identifier: Apache-2.0
"""Observation-level fog of war for the real-game environment."""

from __future__ import annotations

from typing import Any


def apply_vision(state: dict[str, Any], radius: float) -> dict[str, Any]:
    """Observation-level fog of war: keep only enemies in sight of our units.

    The native bridge streams every unit; this filter makes the policy's view
    match what a human player would see, which is what ``vision_radius`` in the
    micro config describes. ``radius <= 0`` disables fog (full information).
    """

    if radius <= 0.0:
        return state
    own = state.get("own_entities", [])
    if not own:
        return state
    limit = float(radius) ** 2
    visible: list[dict[str, Any]] = []
    for enemy in state.get("visible_enemy_entities", []):
        ex = float(enemy.get("x") or 0.0)
        ey = float(enemy.get("y") or 0.0)
        for unit in own:
            dx = ex - float(unit.get("x") or 0.0)
            dy = ey - float(unit.get("y") or 0.0)
            if dx * dx + dy * dy <= limit:
                visible.append(enemy)
                break
    state["visible_enemy_entities"] = visible
    state["target_ids"] = [int(enemy.get("entity_id", -1)) for enemy in visible]
    return state


__all__ = ["apply_vision"]
