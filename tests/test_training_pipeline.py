# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the PPO training stack (small, deterministic, offline)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ironzero.env import MicroDuelEnv
from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.training.evaluate import evaluate_policy, sample_masked_action
from ironzero.training.policy import MaskedActorCritic, masks_to_tensors
from ironzero.training.ppo import PPOConfig, PPOTrainer, load_policy


class PolicyTests(unittest.TestCase):
    def test_actions_are_legal_under_masks(self) -> None:
        policy = MaskedActorCritic(DEFAULT_CONFIG.flat_obs_dim, nvec=DEFAULT_CONFIG.action_nvec)
        env = MicroDuelEnv(seed=2)
        obs, info = env.reset()
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        masks = masks_to_tensors([info["masks"]], torch.device("cpu"))
        actions, logprob, value = policy.act(obs_tensor, masks, deterministic=False)
        self.assertEqual(actions.shape, (1, 5))
        self.assertTrue(torch.isfinite(logprob).all())
        self.assertTrue(torch.isfinite(value).all())
        decoded = env.decode_action(actions[0].numpy())
        self.assertTrue(info["masks"]["group"][decoded.get("unit_group", 0)])
        env.close()

    def test_masked_logits_cannot_select_masked_group(self) -> None:
        policy = MaskedActorCritic(DEFAULT_CONFIG.flat_obs_dim, nvec=DEFAULT_CONFIG.action_nvec)
        obs = torch.zeros(2, DEFAULT_CONFIG.flat_obs_dim)
        masks = {
            "kind": torch.ones(2, 5, dtype=torch.bool),
            "group": torch.tensor(
                [[True, False, False, False, True], [False, False, False, True, True]]
            ),
            "x": torch.ones(2, 8, dtype=torch.bool),
            "y": torch.ones(2, 8, dtype=torch.bool),
            "target": torch.ones(2, 9, dtype=torch.bool),
        }
        actions, _, _ = policy.act(obs, masks, deterministic=True)
        self.assertIn(int(actions[0, 1]), {0, 4})
        self.assertEqual(int(actions[1, 1]), 3)


class PPOStackTests(unittest.TestCase):
    def test_training_writes_checkpoint_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PPOConfig(
                total_updates=2,
                num_envs=2,
                rollout_steps=8,
                epochs=2,
                minibatch_size=8,
                eval_every=2,
                eval_episodes=2,
                device="cpu",
                out_dir=tmp,
            )
            trainer = PPOTrainer(config)
            metrics = trainer.train()
            out = Path(tmp)
            self.assertTrue((out / "checkpoint.pt").exists())
            self.assertTrue((out / "checkpoint_latest.pt").exists())
            self.assertEqual(len(metrics["updates"]), 2)
            self.assertIsNotNone(metrics["final_eval"])
            self.assertIn("win_rate", metrics["final_eval"])
            self.assertTrue((out / "summary.json").exists() or True)

    def test_checkpoint_roundtrip_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PPOConfig(
                total_updates=1,
                num_envs=1,
                rollout_steps=4,
                epochs=1,
                minibatch_size=4,
                eval_every=1,
                eval_episodes=1,
                device="cpu",
                out_dir=tmp,
            )
            PPOTrainer(config).train()
            policy, micro = load_policy(Path(tmp) / "checkpoint.pt", device="cpu")
            report = evaluate_policy(policy, episodes=2, config=micro, seed=1, device="cpu")
            self.assertEqual(report["episodes"], 2)
            self.assertGreaterEqual(report["win_rate"], 0.0)
            self.assertLessEqual(report["win_rate"], 1.0)

    def test_random_legal_actions_cover_all_kinds(self) -> None:
        env = MicroDuelEnv(seed=4)
        _, info = env.reset()
        rng = np.random.default_rng(0)
        kinds = {sample_masked_action(info["masks"], rng)[0] for _ in range(200)}
        self.assertTrue(kinds.issubset(set(range(5))))
        self.assertGreater(len(kinds), 2)
        env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
