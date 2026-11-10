# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from ironzero.micro_config import MicroDuelConfig
from ironzero.mock_engine import (
    EXTRACTOR_TYPE_ID,
    LAND_FACTORY_TYPE_ID,
    TANK_TYPE_ID,
    MicroDuelEngine,
)
from ironzero.protocol import (
    ActionKind,
    BuildExtractorAction,
    BuildStructureAction,
    ProduceUnitAction,
    UpgradeBuildingAction,
    dump_message,
    parse_message,
)


class EconomySystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MicroDuelConfig(
            enable_economy=True,
            starting_funds=2000.0,
            base_income_per_second=6.0,
            extractor_income_per_second=12.0,
            extractor_t2_income_per_second=24.0,
            extractor_cost=700.0,
            extractor_upgrade_cost=1200.0,
            factory_cost=800.0,
            tank_cost=250.0,
            decision_interval=30,
        )
        self.engine = MicroDuelEngine(config=self.config, opponent="idle")

    def test_economy_initialization(self) -> None:
        obs = self.engine.observation(0)
        global_state = obs["global_state"]
        self.assertEqual(global_state["funds"], 2000.0)
        self.assertGreater(global_state["income"], 0.0)
        self.assertEqual(len(obs["resource_slots"]), 4)
        for slot in obs["resource_slots"]:
            self.assertFalse(slot["occupied"])
            self.assertIsNone(slot["building_entity_id"])

    def test_passive_income_growth(self) -> None:
        initial_funds = self.engine.funds[0]
        # 推进 60 ticks (1 秒)
        self.engine.advance(60)
        # 1 秒后资金应增加 base_income_per_second (6.0)
        self.assertAlmostEqual(self.engine.funds[0], initial_funds + 6.0, delta=0.5)

    def test_build_extractor_lifecycle(self) -> None:
        # 1. 建造第一个矿井
        receipts = self.engine.apply_actions(
            [{"kind": "build_extractor", "slot_id": 0}],
            team=0,
        )
        self.assertTrue(receipts[0]["accepted"])
        # 资金扣除 700 (2000 -> 1300)
        self.assertAlmostEqual(self.engine.funds[0], 1300.0, delta=0.5)

        # 检查 slot 状态
        obs = self.engine.observation(0)
        slot0 = obs["resource_slots"][0]
        self.assertTrue(slot0["occupied"])
        self.assertEqual(slot0["owner_team"], 0)
        self.assertEqual(slot0["extractor_level"], 1)

        extractor_id = slot0["building_entity_id"]
        self.assertIsNotNone(extractor_id)
        extractor = self.engine.unit_by_id(extractor_id)
        self.assertIsNotNone(extractor)
        self.assertTrue(extractor.is_building)
        self.assertEqual(extractor.entity_type_id, EXTRACTOR_TYPE_ID)

        # 重复建造同一矿位应当被拒绝 (ACTION_MASKED)
        duplicate_receipts = self.engine.apply_actions(
            [{"kind": "build_extractor", "slot_id": 0}],
            team=0,
        )
        self.assertFalse(duplicate_receipts[0]["accepted"])
        self.assertEqual(duplicate_receipts[0]["rejected_reason"], "ACTION_MASKED")

        # 2. 升级矿井为 T2
        upgrade_receipts = self.engine.apply_actions(
            [{"kind": "upgrade_building", "building_entity_id": extractor_id}],
            team=0,
        )
        self.assertTrue(upgrade_receipts[0]["accepted"])
        # 扣除 1200 (1300 -> 100)
        self.assertAlmostEqual(self.engine.funds[0], 100.0, delta=0.5)
        self.assertEqual(extractor.extractor_level, 2)
        obs_t2 = self.engine.observation(0)
        self.assertEqual(obs_t2["resource_slots"][0]["extractor_level"], 2)

    def test_insufficient_funds_rejection(self) -> None:
        # 清空资金
        self.engine.funds[0] = 50.0
        receipts = self.engine.apply_actions(
            [{"kind": "build_extractor", "slot_id": 1}],
            team=0,
        )
        self.assertFalse(receipts[0]["accepted"])
        self.assertEqual(receipts[0]["rejected_reason"], "INSUFFICIENT_FUNDS")

    def test_factory_production_pipeline(self) -> None:
        # 1. 建造陆军工厂
        receipts = self.engine.apply_actions(
            [
                {
                    "kind": "build_structure",
                    "structure_type": "land_factory",
                    "x": 200.0,
                    "y": 200.0,
                }
            ],
            team=0,
        )
        self.assertTrue(receipts[0]["accepted"])
        # 扣除 800 (2000 -> 1200)
        self.assertAlmostEqual(self.engine.funds[0], 1200.0, delta=0.5)

        obs = self.engine.observation(0)
        factories = [
            u for u in obs["own_entities"]
            if u["is_building"] and u["entity_type_id"] == LAND_FACTORY_TYPE_ID
        ]
        self.assertEqual(len(factories), 1)
        factory_id = factories[0]["entity_id"]

        # 2. 排队生产坦克
        initial_tanks = len(
            [u for u in self.engine.alive_units(0) if not u.is_building]
        )
        produce_receipts = self.engine.apply_actions(
            [
                {
                    "kind": "produce_unit",
                    "factory_entity_id": factory_id,
                    "unit_type": "tank",
                }
            ],
            team=0,
        )
        self.assertTrue(produce_receipts[0]["accepted"])
        # 扣除 250 (1200 -> 950)
        self.assertAlmostEqual(self.engine.funds[0], 950.0, delta=0.5)

        # 推进 65 ticks 超过生产周期 (60 ticks)
        self.engine.advance(65)

        # 检查新坦克是否顺利下线
        current_tanks = len(
            [u for u in self.engine.alive_units(0) if not u.is_building]
        )
        self.assertEqual(current_tanks, initial_tanks + 1)

    def test_protocol_macro_actions_roundtrip(self) -> None:
        # 测试协议对宏观动作的序列化与解析支持
        step_payload = {
            "protocol": "ironzero/1",
            "type": "step",
            "message_id": "m-eco-1",
            "episode_id": "ep-1",
            "tick": 100,
            "decision_id": "d-1",
            "actions": [
                {"kind": "build_extractor", "slot_id": 2},
                {
                    "kind": "build_structure",
                    "structure_type": "land_factory",
                    "x": 150.0,
                    "y": 150.0,
                },
                {"kind": "produce_unit", "unit_type": "tank"},
                {"kind": "upgrade_building", "building_entity_id": 10},
            ],
        }
        parsed = parse_message(step_payload)
        self.assertEqual(len(parsed.actions), 4)
        self.assertIsInstance(parsed.actions[0], BuildExtractorAction)
        self.assertIsInstance(parsed.actions[1], BuildStructureAction)
        self.assertIsInstance(parsed.actions[2], ProduceUnitAction)
        self.assertIsInstance(parsed.actions[3], UpgradeBuildingAction)


if __name__ == "__main__":
    unittest.main()
