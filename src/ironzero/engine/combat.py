# SPDX-License-Identifier: Apache-2.0
"""CombatMixin: targeting, fog of war, damage resolution and win/loss checks."""

from __future__ import annotations

import math

from .constants import EXTRACTOR_TYPE_ID, LAND_FACTORY_TYPE_ID
from .entities import Unit


class CombatMixin:
    """Target selection, visibility, damage and terminal-state detection."""

    def _select_target(self, unit: Unit) -> Unit | None:
        if unit.is_building and unit.entity_type_id in {EXTRACTOR_TYPE_ID, LAND_FACTORY_TYPE_ID}:
            return None
        if unit.order == 1:  # plain move never auto-engages
            return None
        if unit.order == 3 and unit.target_id is not None:
            target = self.unit_by_id(unit.target_id)
            if target is not None and target.alive and self._visible_to(target, unit.team):
                return target
            unit.order = 0
            unit.target_id = None
            self._reset_unit_path(unit)
        best: Unit | None = None
        best_dist = math.inf
        for enemy in self.alive_units(1 - unit.team):
            if not self._visible_to(enemy, unit.team):
                continue
            dist = self._distance(unit, enemy)
            if dist < best_dist:
                best, best_dist = enemy, dist
        return best

    def _visible_to(self, target: Unit, team: int) -> bool:
        """Fog of war check: is ``target`` within vision of any unit of ``team``?"""

        radius = self.config.vision_radius
        if radius <= 0.0:
            return True
        return any(self._distance(observer, target) <= radius for observer in self.alive_units(team))

    def _damage(self, target: Unit, amount: float, source: Unit) -> None:
        if not target.alive:
            return
        actual = min(target.hp, amount)
        target.hp -= actual
        self.stats["damage_dealt"][source.team] += actual
        self.stats["damage_taken"][target.team] += actual
        self.events.append(
            {
                "type": "shot",
                "source": source.entity_id,
                "target": target.entity_id,
                "team": source.team,
                "amount": round(actual, 3),
                "tick": self.tick,
            }
        )
        if target.hp <= 0.0:
            target.alive = False
            target.hp = 0.0
            self.stats["kills"][source.team] += 1
            self.stats["losses"][target.team] += 1
            if getattr(target, "is_building", False) and hasattr(self, "economic_events"):
                self.economic_events[target.team]["building_destroyed"] += 1
            self.events.append(
                {
                    "type": "kill",
                    "source": source.entity_id,
                    "target": target.entity_id,
                    "team": source.team,
                    "x": round(target.x, 3),
                    "y": round(target.y, 3),
                    "tick": self.tick,
                }
            )

    def _check_done(self) -> None:
        cfg = self.config
        if self.done:
            return
        if not self.alive_units(1):
            self.done = True
            self.result = "victory"
        elif not self.alive_units(0):
            self.done = True
            self.result = "defeat"
        elif self.tick >= cfg.episode_ticks:
            self.done = True
            self.result = "timeout"
