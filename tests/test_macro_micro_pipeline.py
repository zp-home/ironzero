# SPDX-License-Identifier: Apache-2.0
"""Test hierarchical Macro+Micro end-to-end training and evaluation loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ironzero.micro_config import MicroDuelConfig
from ironzero.strategy.macro_controller import MacroController
from ironzero.training.evaluate_macro_micro import evaluate_macro_micro_policy
from ironzero.training.ppo import load_policy
from ironzero.training.train_macro_micro import train_macro_micro


class MacroMicroPipelineTests(unittest.TestCase):
    def test_macro_controller_plans_expected_actions(self) -> None:
        macro = MacroController(
            team=0,
            extractor_cost=700.0,
            factory_cost=800.0,
            tank_cost=250.0,
            max_factories=1,
        )
        fake_obs = {
            "global_state": {"funds": 2000.0},
            "resource_slots": [
                {"slot_id": 0, "occupied": False, "owner_team": None},
                {"slot_id": 1, "occupied": False, "owner_team": None},
            ],
            "own_entities": [],
        }
        actions = macro.plan_macro_actions(fake_obs)
        self.assertGreaterEqual(len(actions), 2)
        kinds = [a["kind"] for a in actions]
        self.assertIn("build_extractor", kinds)
        self.assertIn("build_structure", kinds)

    def test_train_and_evaluate_macro_micro_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = train_macro_micro(
                total_updates=2,
                num_envs=2,
                rollout_steps=16,
                out_dir=tmp_dir,
            )
            checkpoint_path = Path(tmp_dir) / "checkpoint.pt"
            self.assertTrue(checkpoint_path.exists())

            policy, _ = load_policy(checkpoint_path, device="cpu")
            cfg = MicroDuelConfig(
                enable_economy=True,
                starting_funds=1000.0,
                episode_ticks=120,
                decision_interval=30,
            )
            summary = evaluate_macro_micro_policy(
                policy,
                episodes=2,
                config=cfg,
                device="cpu",
            )
            self.assertEqual(summary["episodes"], 2)
            self.assertIn("mean_return", summary)
            self.assertIn("mean_decisions", summary)


if __name__ == "__main__":
    unittest.main()
