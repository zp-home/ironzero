# SPDX-License-Identifier: Apache-2.0
"""ActionMixin: validation and execution of one decision's actions.

``apply_actions()`` turns a list of raw action dicts into receipts, delegating
each one to ``_apply_one()`` which either mutates world state and returns
``None`` or returns one of the rejection reason constants.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .constants import (
    ACTION_KINDS,
    ACTION_MASKED,
    EXTRACTOR_TYPE_ID,
    INSUFFICIENT_FUNDS,
    INVALID_MESSAGE,
    LAND_FACTORY_TYPE_ID,
    POSITION_BLOCKED,
    STALE_TICK,
    TARGET_NOT_VISIBLE,
    TURRET_TYPE_ID,
    UNIT_NOT_OWNED,
    UNKNOWN_UNIT,
)
from .entities import Unit


class ActionMixin:
    """Decision-level action application."""

    def apply_actions(
        self,
        actions: Sequence[dict[str, Any]],
        team: int = 0,
    ) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        if self.done:
            return [
                {"accepted": False, "rejected_reason": STALE_TICK, "action_index": i}
                for i, _ in enumerate(actions)
            ]
        for index, action in enumerate(actions[:64]):
            reason = self._apply_one(action, team)
            receipts.append(
                {
                    "action_index": index,
                    "kind": action.get("kind"),
                    "accepted": reason is None,
                    "rejected_reason": reason,
                }
            )
        return receipts

    def _apply_one(self, action: dict[str, Any], team: int) -> str | None:
        kind = action.get("kind")
        if kind not in ACTION_KINDS:
            return INVALID_MESSAGE

        if kind == "build_extractor":
            slot_id = action.get("slot_id")
            if not isinstance(slot_id, int) or not (0 <= slot_id < len(self.resource_slots)):
                return INVALID_MESSAGE
            slot = self.resource_slots[slot_id]
            if slot.occupied:
                return ACTION_MASKED
            if self.funds.get(team, 0.0) < self.config.extractor_cost:
                return INSUFFICIENT_FUNDS
            self.funds[team] -= self.config.extractor_cost
            new_id = max((u.entity_id for u in self.units), default=0) + 1
            extractor = Unit(
                entity_id=new_id,
                team=team,
                group=0,
                x=slot.x,
                y=slot.y,
                max_hp=600.0,
                hp=600.0,
                is_building=True,
                entity_type_id=EXTRACTOR_TYPE_ID,
                build_progress=1.0,
                extractor_level=1,
            )
            self.units.append(extractor)
            slot.occupied = True
            slot.building_entity_id = new_id
            slot.owner_team = team
            slot.extractor_level = 1
            if hasattr(self, "economic_events"):
                self.economic_events[team]["extractor_built"] += 1
            return None

        if kind == "build_structure":
            stype = action.get("structure_type", "land_factory")
            cost = self.config.factory_cost if stype == "land_factory" else self.config.turret_cost
            if self.funds.get(team, 0.0) < cost:
                return INSUFFICIENT_FUNDS
            x = float(action.get("x", math.nan))
            y = float(action.get("y", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                return INVALID_MESSAGE
            if not (0.0 <= x <= self.config.map_width and 0.0 <= y <= self.config.map_height):
                return POSITION_BLOCKED
            self.funds[team] -= cost
            new_id = max((u.entity_id for u in self.units), default=0) + 1
            type_id = LAND_FACTORY_TYPE_ID if stype == "land_factory" else TURRET_TYPE_ID
            hp = 1200.0 if stype == "land_factory" else 450.0
            building = Unit(
                entity_id=new_id,
                team=team,
                group=0,
                x=x,
                y=y,
                max_hp=hp,
                hp=hp,
                is_building=True,
                entity_type_id=type_id,
                build_progress=1.0,
            )
            self.units.append(building)
            if hasattr(self, "economic_events"):
                self.economic_events[team]["factory_built"] += 1
            return None

        if kind == "produce_unit":
            unit_type = action.get("unit_type", "tank")
            cost = self.config.tank_cost
            if self.funds.get(team, 0.0) < cost:
                return INSUFFICIENT_FUNDS
            factory_id = action.get("factory_entity_id")
            factory: Unit | None = None
            if factory_id is not None:
                u = self.unit_by_id(factory_id)
                if u is not None and u.alive and u.team == team and u.entity_type_id == LAND_FACTORY_TYPE_ID:
                    factory = u
            if factory is None:
                for u in self.alive_units(team):
                    if u.is_building and u.entity_type_id == LAND_FACTORY_TYPE_ID:
                        factory = u
                        break
            if factory is None:
                return ACTION_MASKED
            self.funds[team] -= cost
            factory.production_queue.append(unit_type)
            return None

        if kind == "upgrade_building":
            building_id = action.get("building_entity_id")
            building = self.unit_by_id(building_id) if isinstance(building_id, int) else None
            if building is None or not building.alive or building.team != team or not building.is_building:
                return INVALID_MESSAGE
            if building.entity_type_id == EXTRACTOR_TYPE_ID and building.extractor_level == 1:
                cost = self.config.extractor_upgrade_cost
                if self.funds.get(team, 0.0) < cost:
                    return INSUFFICIENT_FUNDS
                self.funds[team] -= cost
                building.extractor_level = 2
                for slot in self.resource_slots:
                    if slot.building_entity_id == building.entity_id:
                        slot.extractor_level = 2
                if hasattr(self, "economic_events"):
                    self.economic_events[team]["extractor_upgraded"] += 1
                return None
            return ACTION_MASKED

        if kind == "repair_target":
            target_id = action.get("target_entity_id")
            target = self.unit_by_id(target_id) if isinstance(target_id, int) else None
            if target is None or not target.alive or target.team != team:
                return INVALID_MESSAGE
            target.hp = min(target.max_hp, target.hp + 50.0)
            return None

        group = action.get("unit_group")
        if not isinstance(group, int) or not 0 <= group < self.config.groups:
            return UNKNOWN_UNIT
        members = [u for u in self.alive_units(team) if u.group == group]
        if not members:
            return UNIT_NOT_OWNED

        if kind in {"move_group", "attack_move"}:
            x = float(action.get("x", math.nan))
            y = float(action.get("y", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                return INVALID_MESSAGE
            if not (0.0 <= x <= self.config.map_width and 0.0 <= y <= self.config.map_height):
                return POSITION_BLOCKED
            if self.terrain is not None:
                # A target inside water means "move to the nearest shore".
                x, y = self.terrain.nearest_passable_world(x, y)
            order = 1 if kind == "move_group" else 2
            for member in members:
                member.order = order
                member.tx = x
                member.ty = y
                member.target_id = None
            if self.terrain is not None:
                self._assign_group_path(members, (x, y))
            return None

        if kind == "attack_target":
            target_id = action.get("target_entity_id")
            if not isinstance(target_id, int) or target_id < 0:
                return INVALID_MESSAGE
            target = self.unit_by_id(target_id)
            if target is None or not target.alive or target.team == team:
                return TARGET_NOT_VISIBLE
            for member in members:
                member.order = 3
                member.target_id = target_id
            if self.terrain is not None:
                self._assign_group_path(members, (target.x, target.y))
            return None

        if kind == "stop":
            for member in members:
                member.order = 0
                member.target_id = None
                member.tx = member.x
                member.ty = member.y
                self._reset_unit_path(member)
            return None

        # hold
        for member in members:
            member.order = 4
            member.target_id = None
            member.tx = member.x
            member.ty = member.y
            self._reset_unit_path(member)
        return None
