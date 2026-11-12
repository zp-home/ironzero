# SPDX-License-Identifier: Apache-2.0
"""Tests for observation encoding, action masks and the gym-style env."""

from __future__ import annotations

import unittest

import numpy as np

from ironzero.env import MicroDuelEnv, MultiDiscreteSpace
from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.mock_engine import ACTION_KINDS
from ironzero.observations import (
    build_masks,
    encode_observation,
    flat_mask_vector,
    masks_as_lists,
)
from ironzero.training.evaluate import sample_masked_action


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MicroDuelEnv(seed=11)
        self.obs, self.info = self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_flat_observation_shape_and_dtype(self) -> None:
        self.assertEqual(self.obs.shape, (DEFAULT_CONFIG.flat_obs_dim,))
        self.assertEqual(self.obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(self.obs)))

    def test_encoder_matches_contract(self) -> None:
        encoded = encode_observation(self.info["raw_state"])
        np.testing.assert_allclose(encoded, self.obs, rtol=0, atol=0)

    def test_mask_shapes_and_minimum_validity(self) -> None:
        masks = build_masks(self.info["raw_state"])
        # The kind head spans every entry of ACTION_KINDS (10 today); the head
        # *count* is still 5. Deriving keeps this test alive across contract bumps.
        self.assertEqual(masks["kind"].shape, (DEFAULT_CONFIG.action_nvec[0],))
        self.assertEqual(masks["group"].shape, (DEFAULT_CONFIG.groups + 1,))
        self.assertEqual(masks["target"].shape, (DEFAULT_CONFIG.target_slots,))
        # The five tactical orders plus ``noop`` are unconditional; the economy
        # kinds are gated on an affordable builder, so the head is no longer
        # all-True. The default mock army is tank-only, so exactly the tactical
        # prefix and the trailing no-op are selectable.
        self.assertTrue(masks["kind"][:5].all())
        self.assertTrue(masks["kind"][-1], "noop must always be selectable")
        tactical = sum(1 for kind in ACTION_KINDS if kind in {
            "move_group", "attack_move", "attack_target", "stop", "hold", "noop"
        })
        self.assertEqual(int(masks["kind"].sum()), tactical)
        self.assertFalse(masks["kind"][5:-1].any())
        self.assertTrue(masks["group"].any())
        self.assertTrue(masks["target"].any())
        flat = flat_mask_vector(masks)
        self.assertEqual(flat.dtype, np.bool_)
        self.assertTrue(flat[:5].all())

    def test_masks_serialize_into_protocol_message(self) -> None:
        from ironzero.mock_bridge import state_to_observation_message

        message = state_to_observation_message(
            self.info["raw_state"], episode_id="ep-1", decision_id="d-1"
        )
        self.assertEqual(len(message.masks.action_mask), DEFAULT_CONFIG.action_nvec[0])
        self.assertEqual(len(message.masks.unit_group_mask), DEFAULT_CONFIG.groups + 1)
        self.assertEqual(len(message.masks.position_mask), DEFAULT_CONFIG.position_grid)
        self.assertEqual(message.tick, 0)

    def test_action_space_nvec(self) -> None:
        space = self.env.action_space
        self.assertIsInstance(space, MultiDiscreteSpace)
        self.assertEqual(tuple(int(v) for v in space.nvec), DEFAULT_CONFIG.action_nvec)
        sample = space.sample()
        self.assertTrue(space.contains(sample))


class EnvironmentTests(unittest.TestCase):
    def test_step_advances_tick_by_decision_interval(self) -> None:
        env = MicroDuelEnv(seed=3)
        _, info = env.reset()
        start_tick = info["raw_state"]["tick"]
        action = sample_masked_action(info["masks"], np.random.default_rng(0))
        _, _, _, _, step_info = env.step(action)
        self.assertEqual(step_info["tick"] - start_tick, DEFAULT_CONFIG.decision_interval)

    def test_attack_target_uses_target_slot_mapping(self) -> None:
        env = MicroDuelEnv(seed=3)
        _, info = env.reset()
        target_ids = info["raw_state"]["target_ids"]
        action = np.array([2, 0, 0, 0, 2], dtype=np.int64)
        decoded = env.decode_action(action)
        self.assertEqual(decoded["kind"], "attack_target")
        self.assertEqual(decoded["target_entity_id"], target_ids[2])

    def test_no_target_slot_falls_back_to_stop(self) -> None:
        env = MicroDuelEnv(seed=3)
        _, info = env.reset()
        action = np.array([2, 0, 0, 0, DEFAULT_CONFIG.target_slots - 1], dtype=np.int64)
        decoded = env.decode_action(action)
        self.assertEqual(decoded["kind"], "stop")

    def test_group_mask_freezes_to_alive_group(self) -> None:
        env = MicroDuelEnv(seed=3)
        _, info = env.reset()
        masks = dict(info["masks"])
        masks["group"] = np.array([False, True, False, False, True])
        env.masks = masks
        self.assertEqual(env._safe_group(0), 1)

    def test_episode_terminates_with_random_legal_actions(self) -> None:
        env = MicroDuelEnv(seed=5)
        rng = np.random.default_rng(5)
        _, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 400:
            action = sample_masked_action(info["masks"], rng)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        self.assertTrue(done)
        self.assertIn(info["result"], {"victory", "defeat", "timeout"})

    def test_broadcast_group_commands_all_groups(self) -> None:
        env = MicroDuelEnv(seed=3)
        env.reset()
        cfg = DEFAULT_CONFIG
        action = np.array([1, cfg.groups, 4, 4, 0], dtype=np.int64)
        actions = env.decode_actions(action)
        self.assertEqual(len(actions), cfg.groups)
        self.assertEqual({item["unit_group"] for item in actions}, set(range(cfg.groups)))
        env.close()

    def test_reset_seed_is_reproducible(self) -> None:
        first = MicroDuelEnv(seed=42)
        second = MicroDuelEnv(seed=42)
        obs_a, _ = first.reset()
        obs_b, _ = second.reset()
        np.testing.assert_allclose(obs_a, obs_b)
        action = np.array([0, 0, 7, 7, 0], dtype=np.int64)
        first.step(action)
        second.step(action)
        np.testing.assert_allclose(first.state["own_entities"][0]["x"], second.state["own_entities"][0]["x"])
        first.close()
        second.close()

    def test_masked_random_actions_are_legal(self) -> None:
        env = MicroDuelEnv(seed=9)
        _, info = env.reset()
        rng = np.random.default_rng(1)
        for _ in range(20):
            action = sample_masked_action(info["masks"], rng)
            decoded = env.decode_action(action)
            self.assertIn(
                decoded["kind"],
                {"move_group", "attack_move", "attack_target", "stop", "hold", "noop"},
            )
            if "unit_group" in decoded:
                self.assertTrue(info["masks"]["group"][decoded["unit_group"]])
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
