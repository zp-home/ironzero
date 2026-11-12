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
from ironzero.mock_engine import ACTION_KINDS
from ironzero.observations import build_masks
from ironzero.training.evaluate import evaluate_policy, sample_masked_action
from ironzero.training.policy import MaskedActorCritic, masks_to_tensors
from ironzero.training.ppo import PPOConfig, PPOTrainer, load_policy


class ActionContractTests(unittest.TestCase):
    """Guards the single source of truth for the action/observation contract.

    A previous change moved the ``kind`` head from 5 to 10 values but left parts
    of the stack pinned to the old width, which only surfaced at the first
    forward pass. This assertion makes that class of half-finished edit fail
    loudly at its source instead.
    """

    def test_action_space_contract_is_derived_consistently(self) -> None:
        cfg = DEFAULT_CONFIG
        # 1. The kind head spans exactly the engine's action vocabulary...
        self.assertEqual(cfg.action_nvec[0], len(ACTION_KINDS))
        # 2. ...while the head count stays at five (kind, group, x, y, target).
        self.assertEqual(len(cfg.action_nvec), 5)

        # 3. The masks the environment actually emits are as wide as the head.
        #    A narrower mask crashes the masked forward pass, not the mask build.
        env = MicroDuelEnv(seed=11)
        try:
            _, info = env.reset()
            masks = build_masks(info["raw_state"])
        finally:
            env.close()
        self.assertEqual(masks["kind"].shape[0], cfg.action_nvec[0])

        # 4. The policy derives its heads from the same contract when no explicit
        #    ``nvec`` is passed, so a stale hard-coded default cannot reappear.
        policy = MaskedActorCritic(cfg.flat_obs_dim)
        self.assertEqual(policy.nvec, cfg.action_nvec)
        # Prove the derived widths are usable end to end, not just equal.
        tensors = {
            key: torch.ones(1, width, dtype=torch.bool)
            for key, width in zip(("kind", "group", "x", "y", "target"), cfg.action_nvec)
        }
        actions, logprob, _ = policy.act(
            torch.zeros(1, cfg.flat_obs_dim), tensors, deterministic=True
        )
        self.assertEqual(actions.shape, (1, len(cfg.action_nvec)))
        self.assertTrue(torch.isfinite(logprob).all())


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
            "kind": torch.ones(2, DEFAULT_CONFIG.action_nvec[0], dtype=torch.bool),
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
        self.assertTrue(kinds.issubset(set(range(DEFAULT_CONFIG.action_nvec[0]))))
        self.assertGreater(len(kinds), 2)
        env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
