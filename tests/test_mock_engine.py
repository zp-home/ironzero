# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic micro-duel engine."""

from __future__ import annotations

import math
import unittest

from ironzero.micro_config import MicroDuelConfig
from ironzero.mock_engine import (
    MicroDuelEngine,
    POSITION_BLOCKED,
    TARGET_NOT_VISIBLE,
    UNKNOWN_UNIT,
)


def run_scripted_fight(engine: MicroDuelEngine, decisions: int = 240) -> None:
    while not engine.done and decisions > 0:
        decisions -= 1
        enemies = engine.alive_units(1)
        if not enemies:
            break
        target = enemies[0].entity_id
        actions = [
            {"kind": "attack_target", "unit_group": group, "target_entity_id": target}
            for group in range(engine.config.groups)
        ]
        engine.step_decision(actions)


class MicroDuelEngineTests(unittest.TestCase):
    def test_reset_is_deterministic_for_a_seed(self) -> None:
        first = MicroDuelEngine()
        second = MicroDuelEngine()
        first.reset(seed=123)
        second.reset(seed=123)
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_different_seeds_produce_different_layouts(self) -> None:
        first = MicroDuelEngine()
        second = MicroDuelEngine()
        first.reset(seed=1)
        second.reset(seed=2)
        self.assertNotEqual(first.snapshot(), second.snapshot())

    def test_fight_terminates_and_rewards_stay_finite(self) -> None:
        engine = MicroDuelEngine()
        engine.reset(seed=7)
        run_scripted_fight(engine)
        self.assertTrue(engine.done)
        self.assertIn(engine.result, {"victory", "defeat", "timeout"})
        eval_result = engine.evaluate(0)
        total_hp = engine.config.units_per_team * engine.config.unit_hp
        self.assertGreaterEqual(eval_result["own_hp_frac"], 0.0)
        self.assertLessEqual(eval_result["enemy_hp_frac"], 1.0)
        # Damage accounting must conserve hit points exactly (no ghost damage).
        self.assertAlmostEqual(
            eval_result["damage_dealt"],
            total_hp - eval_result["enemy_hp_frac"] * total_hp,
            delta=1e-2,
        )
        self.assertAlmostEqual(
            eval_result["damage_taken"],
            total_hp - eval_result["own_hp_frac"] * total_hp,
            delta=1e-2,
        )
        self.assertTrue(math.isfinite(engine._pending_reward))

    def test_controlled_focus_fire_deals_damage(self) -> None:
        engine = MicroDuelEngine(opponent="idle")
        engine.reset(seed=5)
        decisions = 0
        while not engine.done and decisions < 400:
            enemies = engine.alive_units(1)
            if not enemies:
                break
            target_id = enemies[0].entity_id
            actions = [
                {"kind": "attack_target", "unit_group": group, "target_entity_id": target_id}
                for group in range(engine.config.groups)
            ]
            engine.step_decision(actions)
            decisions += 1
        self.assertTrue(engine.done)
        self.assertIn(engine.result, {"victory", "defeat", "timeout"})
        evaluation = engine.evaluate(0)
        self.assertGreater(evaluation["damage_dealt"], 0.0)
        self.assertGreater(engine.stats["kills"][0], 0)

    def test_passive_team_still_terminates_against_scripted_opponent(self) -> None:
        engine = MicroDuelEngine(opponent="scripted")
        engine.reset(seed=5)
        decisions = 0
        while not engine.done and decisions < 400:
            engine.step_decision([])
            decisions += 1
        self.assertTrue(engine.done)
        self.assertIn(engine.result, {"victory", "defeat", "timeout"})

    def test_action_validation_rejects_bad_actions(self) -> None:
        engine = MicroDuelEngine()
        engine.reset(seed=0)
        receipts = engine.apply_actions(
            [
                {"kind": "move_group", "unit_group": 99, "x": 10.0, "y": 10.0},
                {"kind": "attack_target", "unit_group": 0, "target_entity_id": 99},
                {"kind": "move_group", "unit_group": 0, "x": -5.0, "y": 10.0},
                {"kind": "stop", "unit_group": 0},
            ]
        )
        reasons = [receipt["rejected_reason"] for receipt in receipts]
        self.assertEqual(reasons[0], UNKNOWN_UNIT)
        self.assertEqual(reasons[1], TARGET_NOT_VISIBLE)
        self.assertEqual(reasons[2], POSITION_BLOCKED)
        self.assertIsNone(reasons[3])

    def test_actions_move_the_units(self) -> None:
        engine = MicroDuelEngine(opponent="idle")
        engine.reset(seed=0)
        before = engine.observation()
        start = {u["entity_id"]: (u["x"], u["y"]) for u in before["own_entities"]}
        engine.step_decision(
            [{"kind": "move_group", "unit_group": 0, "x": 700.0, "y": 500.0}]
        )
        after = engine.observation()
        moved = [
            u["entity_id"]
            for u in after["own_entities"]
            if (u["x"], u["y"]) != start.get(u["entity_id"])
        ]
        self.assertTrue(moved)
        for entity in after["own_entities"]:
            self.assertLessEqual(entity["x"], engine.config.map_width)
            self.assertGreaterEqual(entity["x"], 0.0)

    def test_custom_config_keeps_shapes_consistent(self) -> None:
        config = MicroDuelConfig(units_per_team=4, groups=2, decision_interval=10)
        engine = MicroDuelEngine(config=config)
        engine.reset(seed=3)
        self.assertEqual(len(engine.units), 8)
        self.assertEqual(engine.config.action_nvec[1], 3)  # 2 groups + broadcast slot
        self.assertEqual(engine.config.action_nvec[0], 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
