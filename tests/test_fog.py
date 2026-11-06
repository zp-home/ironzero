# SPDX-License-Identifier: Apache-2.0
"""Fog-of-war tests: visibility filtering in the reference engine."""

from __future__ import annotations

from dataclasses import replace
import unittest

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.mock_engine import MicroDuelEngine


class FogOfWarTests(unittest.TestCase):
    def test_default_config_has_no_fog(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.vision_radius, 0.0)
        engine = MicroDuelEngine()
        engine.reset(seed=1)
        state = engine.observation()
        self.assertFalse(state["fog"])
        self.assertEqual(len(state["visible_enemy_entities"]), 8)

    def test_enemies_out_of_vision_are_hidden(self) -> None:
        config = replace(DEFAULT_CONFIG, vision_radius=300.0)
        engine = MicroDuelEngine(config=config, opponent="scripted")
        engine.reset(seed=0)
        state = engine.observation()
        self.assertTrue(state["fog"])
        # Spawns are ~560 px apart, well outside a 300 px vision radius.
        self.assertEqual(state["visible_enemy_entities"], [])
        self.assertEqual(state["target_ids"], [])
        self.assertEqual(len(state["own_entities"]), 8)

    def test_enemies_become_visible_when_in_range(self) -> None:
        config = replace(DEFAULT_CONFIG, vision_radius=300.0)
        engine = MicroDuelEngine(config=config, opponent="scripted")
        engine.reset(seed=0)
        visible = 0
        advance = [
            {"kind": "attack_move", "unit_group": group, "x": 760.0, "y": 500.0}
            for group in range(DEFAULT_CONFIG.groups)
        ]
        for _ in range(120):
            if engine.done:
                break
            engine.step_decision(advance)
            state = engine.observation()
            visible = len(state["visible_enemy_entities"])
            if visible:
                # Everything reported as visible must be inside the vision radius.
                own = engine.alive_units(0)
                for payload in state["visible_enemy_entities"]:
                    distance = min(
                        (payload["x"] - unit.x) ** 2 + (payload["y"] - unit.y) ** 2 for unit in own
                    ) ** 0.5
                    self.assertLessEqual(distance, 300.0 + 1e-6)
                break
        self.assertGreater(visible, 0)

    def test_scripted_ai_advances_toward_invisible_enemies(self) -> None:
        config = replace(DEFAULT_CONFIG, vision_radius=300.0)
        engine = MicroDuelEngine(config=config, opponent="scripted")
        engine.reset(seed=0)
        # Enemies are outside vision: the scripted AI advances instead of idling.
        orders = engine._opponent_actions(team=0)
        self.assertTrue(orders)
        self.assertEqual({order["kind"] for order in orders}, {"attack_move"})
        advance = [
            {"kind": "attack_move", "unit_group": group, "x": 760.0, "y": 500.0}
            for group in range(DEFAULT_CONFIG.groups)
        ]
        for _ in range(140):
            if engine.done:
                break
            engine.step_decision(advance)
            orders = engine._opponent_actions(team=0)
            if orders and all(order["kind"] == "attack_target" for order in orders):
                break
        self.assertTrue(orders)
        self.assertEqual({order["kind"] for order in orders}, {"attack_target"})

    def test_fog_does_not_change_default_arena_behavior(self) -> None:
        plain = MicroDuelEngine(opponent="scripted")
        plain.reset(seed=7)
        fogged = MicroDuelEngine(config=replace(DEFAULT_CONFIG, vision_radius=0.0), opponent="scripted")
        fogged.reset(seed=7)
        for _ in range(10):
            actions = [{"kind": "attack_move", "unit_group": 0, "x": 700.0, "y": 500.0}]
            plain.step_decision(actions)
            fogged.step_decision(actions)
        self.assertEqual(plain.snapshot(), fogged.snapshot())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
