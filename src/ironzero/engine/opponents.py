# SPDX-License-Identifier: Apache-2.0
"""OpponentMixin: the scripted / idle / random adversary driver.

``scripted_actions()`` is also used as the team-0 helper for the "scripted"
opponent, so both teams share one focus-fire implementation.
"""

from __future__ import annotations

from typing import Any

from .constants import ACTION_KINDS


class OpponentMixin:
    """Scripted opponent policies."""

    def scripted_actions(self, team: int = 0) -> list[dict[str, Any]]:
        """Deterministic "attack the nearest enemy" orders for ``team``.

        Under fog of war the scripted AI cannot target what it cannot see, so it
        advances toward the nearest enemy position with an attack-move order and
        switches to an explicit attack once the target is visible.
        """

        if self.done:
            return []
        own = self.alive_units(team)
        enemies = self.alive_units(1 - team)
        if not own or not enemies:
            return []
        actions: list[dict[str, Any]] = []
        fog = self.config.vision_radius > 0.0
        for group in sorted({unit.group for unit in own}):
            members = [unit for unit in own if unit.group == group]
            anchor = members[0]
            nearest = min(enemies, key=lambda target: self._distance(anchor, target))
            if fog and not self._visible_to(nearest, team):
                actions.append(
                    {
                        "kind": "attack_move",
                        "unit_group": group,
                        "x": nearest.x,
                        "y": nearest.y,
                    }
                )
            else:
                actions.append(
                    {
                        "kind": "attack_target",
                        "unit_group": group,
                        "target_entity_id": nearest.entity_id,
                    }
                )
        return actions

    def _opponent_actions(self, team: int = 0) -> list[dict[str, Any]]:
        if self.done:
            return []
        enemy_team = 1 - team
        own = self.alive_units(enemy_team)
        if not own:
            return []
        targets = self.alive_units(team)
        if not targets:
            return []
        if self.opponent == "idle":
            return []
        if self.opponent == "random":
            actions: list[dict[str, Any]] = []
            for group in sorted({u.group for u in own}):
                member = next(u for u in own if u.group == group)
                kind = self._rng.choice(ACTION_KINDS)
                if kind in {"move_group", "attack_move"}:
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "x": self._rng.uniform(0.0, self.config.map_width),
                            "y": self._rng.uniform(0.0, self.config.map_height),
                        }
                    )
                elif kind == "attack_target":
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "target_entity_id": self._rng.choice(targets).entity_id,
                        }
                    )
                else:
                    actions.append({"kind": kind, "unit_group": group})
                _ = member
            return actions
        # scripted aggressor: advance/target exactly like scripted_actions()
        return self.scripted_actions(enemy_team)
