# SPDX-License-Identifier: Apache-2.0
"""Macroeconomic and production strategy controller for IronZero.

This controller executes high-level RTS build orders:
1. Occupy resource slots with extractors whenever funds permit.
2. Establish land factories for military production.
3. Keep production queues busy churning out combat units.
4. Upgrade extractors to T2 for compound economy expansion.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ironzero.mock_engine import EXTRACTOR_TYPE_ID, LAND_FACTORY_TYPE_ID


class MacroController:
    """Deterministic build-order and production manager."""

    def __init__(
        self,
        *,
        team: int = 0,
        extractor_cost: float = 700.0,
        extractor_upgrade_cost: float = 1200.0,
        factory_cost: float = 800.0,
        tank_cost: float = 250.0,
        max_factories: int = 2,
    ) -> None:
        self.team = team
        self.extractor_cost = extractor_cost
        self.extractor_upgrade_cost = extractor_upgrade_cost
        self.factory_cost = factory_cost
        self.tank_cost = tank_cost
        self.max_factories = max_factories
        self.stats = {
            "extractors_built": 0,
            "extractors_upgraded": 0,
            "factories_built": 0,
            "units_queued": 0,
        }

    def reset_stats(self) -> None:
        self.stats = {
            "extractors_built": 0,
            "extractors_upgraded": 0,
            "factories_built": 0,
            "units_queued": 0,
        }

    def get_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def plan_macro_actions(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Generate macro actions (build, upgrade, produce) from current state."""
        actions: list[dict[str, Any]] = []
        global_state = observation.get("global_state", {})
        funds = float(global_state.get("funds", 0.0))
        resource_slots = observation.get("resource_slots", [])
        own_entities = observation.get("own_entities", [])

        # 1. 优先抢占空置矿点建造提取器
        unclaimed_slots = [
            s for s in resource_slots
            if not s.get("occupied") or s.get("owner_team") is None
        ]
        if unclaimed_slots and funds >= self.extractor_cost:
            target_slot = unclaimed_slots[0]
            actions.append({
                "kind": "build_extractor",
                "slot_id": int(target_slot["slot_id"]),
            })
            funds -= self.extractor_cost
            self.stats["extractors_built"] += 1

        # 2. 统计现有工厂数量，若未达上限且资金充足则建造陆军工厂
        factories = [
            u for u in own_entities
            if u.get("is_building") and u.get("entity_type_id") == LAND_FACTORY_TYPE_ID
        ]
        if len(factories) < self.max_factories and funds >= self.factory_cost:
            # 依队伍在己方后方选址建造
            base_x = 180.0 if self.team == 0 else 820.0
            base_y = 200.0 + len(factories) * 120.0
            actions.append({
                "kind": "build_structure",
                "structure_type": "land_factory",
                "x": base_x,
                "y": base_y,
            })
            funds -= self.factory_cost
            self.stats["factories_built"] += 1

        # 3. 如果已有工厂，保持工厂排队生产坦克
        for factory in factories:
            q_len = int(factory.get("production_queue_length", 0))
            if q_len < 2 and funds >= self.tank_cost:
                actions.append({
                    "kind": "produce_unit",
                    "factory_entity_id": factory.get("entity_id"),
                    "unit_type": "tank",
                })
                funds -= self.tank_cost
                self.stats["units_queued"] += 1

        # 4. 若资金依然富余，将现有 T1 提取器升级为 T2
        if funds >= self.extractor_upgrade_cost:
            t1_extractors = [
                u for u in own_entities
                if u.get("is_building")
                and u.get("entity_type_id") == EXTRACTOR_TYPE_ID
                and u.get("extractor_level", 1) < 2
            ]
            if t1_extractors:
                actions.append({
                    "kind": "upgrade_building",
                    "building_entity_id": t1_extractors[0].get("entity_id"),
                })
                funds -= self.extractor_upgrade_cost
                self.stats["extractors_upgraded"] += 1

        return actions
