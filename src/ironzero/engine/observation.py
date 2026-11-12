# SPDX-License-Identifier: Apache-2.0
"""ObservationMixin: the ``ironzero/1``-shaped raw payload handed to the bridge."""

from __future__ import annotations

from typing import Any

from .constants import BUILDER_TYPE_ID, LAND_FACTORY_TYPE_ID, ORDER_NAMES
from .entities import Unit


class ObservationMixin:
    """Observation payload and per-entity payload building."""

    def observation(self, team: int = 0) -> dict[str, Any]:
        cfg = self.config
        own = self.alive_units(team)
        enemies = self.alive_units(1 - team)
        if cfg.vision_radius > 0.0:
            enemies = [unit for unit in enemies if self._visible_to(unit, team)]
        own_hp = sum(u.hp for u in own)
        enemy_hp = sum(u.hp for u in enemies)
        own_buildings = [u for u in own if u.is_building]
        return {
            "tick": self.tick,
            "game_time": self.tick / cfg.tick_rate,
            "controlled_team": team,
            "fog": cfg.vision_radius > 0.0,
            "vision_radius": cfg.vision_radius,
            "global_state": {
                "unit_count": len([u for u in own if not u.is_building]),
                "building_count": len(own_buildings),
                "army_value": round(own_hp, 3),
                "visible_enemy_value": round(enemy_hp, 3),
                "kills": self.stats["kills"][team],
                "losses": self.stats["losses"][team],
                "income": round(self._calculate_income(team), 3),
                "funds": round(self.funds.get(team, 0.0), 3),
            },
            "own_entities": [self._entity_payload(u) for u in own],
            "visible_enemy_entities": [self._entity_payload(u) for u in enemies],
            "remembered_enemies": [],
            "local_maps": {},
            "resource_slots": [
                {
                    "slot_id": s.slot_id,
                    "x": s.x,
                    "y": s.y,
                    "occupied": s.occupied,
                    "building_entity_id": s.building_entity_id,
                    "owner_team": s.owner_team,
                    "extractor_level": s.extractor_level,
                }
                for s in self.resource_slots
            ],
            "target_ids": [u.entity_id for u in enemies],
            "tick_interval": cfg.decision_interval,
        }

    def _entity_payload(self, unit: Unit) -> dict[str, Any]:
        # ``is_builder`` / ``is_factory`` are derived from the mock engine's own
        # type table (builder=2, extractor=10, landFactory=11, turret=12). They
        # are reported explicitly because the action mask trusts only these
        # capability flags: inferring "is a builder" from a numeric type id is
        # exactly the guess that made the real bridge open the economy with no
        # builder on the map (see ``observation/masks.py``).
        is_builder = unit.entity_type_id == BUILDER_TYPE_ID
        is_factory = unit.is_building and unit.entity_type_id == LAND_FACTORY_TYPE_ID
        return {
            "entity_id": unit.entity_id,
            "entity_type_id": unit.entity_type_id,
            "team_id": unit.team,
            "x": round(unit.x, 4),
            "y": round(unit.y, 4),
            "hp": round(unit.hp, 4),
            "max_hp": unit.max_hp,
            "is_building": unit.is_building,
            "is_builder": is_builder,
            "is_factory": is_factory,
            "is_visible": True,
            "is_alive": unit.alive,
            "order_type": ORDER_NAMES[unit.order],
            "attack_target_id": unit.target_id,
            "group_id": unit.group,
            "cooldown_frac": round(unit.cooldown / self.config.attack_cooldown_ticks, 4),
            "build_progress": unit.build_progress,
            "production_queue_length": len(unit.production_queue) if unit.is_building else 0,
        }
