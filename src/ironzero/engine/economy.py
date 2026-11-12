# SPDX-License-Identifier: Apache-2.0
"""EconomyMixin: passive income, extractor income and factory production.

The economy only ticks when ``config.enable_economy`` is set; arena-mode games
without economy therefore pay exactly zero cost for this mixin.
"""

from __future__ import annotations

from .constants import EXTRACTOR_TYPE_ID, LAND_FACTORY_TYPE_ID, TANK_TYPE_ID
from .entities import Unit


class EconomyMixin:
    """Income accrual, production queues and the income observation value."""

    def _economic_tick(self) -> None:
        """Advance funds, extractor income and factory production by one tick."""

        cfg = self.config
        for team in range(cfg.teams):
            self.funds[team] += cfg.base_income_per_second / cfg.tick_rate
        for unit in list(self.units):
            if not unit.alive or not unit.is_building:
                continue
            if unit.entity_type_id == EXTRACTOR_TYPE_ID and unit.build_progress >= 1.0:
                inc = (
                    cfg.extractor_income_per_second
                    if unit.extractor_level <= 1
                    else cfg.extractor_t2_income_per_second
                )
                self.funds[unit.team] += inc / cfg.tick_rate
            elif unit.entity_type_id == LAND_FACTORY_TYPE_ID and unit.build_progress >= 1.0:
                if unit.production_queue:
                    unit.production_timer += 1
                    if unit.production_timer >= 60:
                        unit.production_timer = 0
                        unit.production_queue.pop(0)
                        new_id = max((u.entity_id for u in self.units), default=0) + 1
                        offset_x = 35.0 if unit.team == 0 else -35.0
                        new_unit = Unit(
                            entity_id=new_id,
                            team=unit.team,
                            group=0,
                            x=unit.x + offset_x,
                            y=unit.y,
                            max_hp=cfg.unit_hp,
                            hp=cfg.unit_hp,
                            entity_type_id=TANK_TYPE_ID,
                        )
                        self.units.append(new_unit)
                        if hasattr(self, "economic_events"):
                            self.economic_events[unit.team]["unit_produced"] += 1

    def _calculate_income(self, team: int) -> float:
        if not self.config.enable_economy:
            return 0.0
        income = self.config.base_income_per_second
        for u in self.alive_units(team):
            if u.is_building and u.entity_type_id == EXTRACTOR_TYPE_ID and u.build_progress >= 1.0:
                income += (
                    self.config.extractor_income_per_second
                    if u.extractor_level <= 1
                    else self.config.extractor_t2_income_per_second
                )
        return income
