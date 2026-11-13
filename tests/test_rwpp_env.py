# SPDX-License-Identifier: Apache-2.0
"""Tests for the real-game RL environment (bridge protocol level, no game needed).

A tiny in-process fake bridge plays the role of ``IronZeroBridge.java`` so the
reset/step loop, observation encoding, action decoding and reward shaping can be
verified deterministically on machines without the Rusted Warfare runtime.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest

import numpy as np

from ironzero.micro_config import MicroDuelConfig
from ironzero.observations import build_masks, encode_observation
from ironzero.rwpp_env import (
    RwppDuelEnv,
    RwppEnvError,
    RwppGameConfig,
    apply_vision,
    decode_action,
    hp_potential,
    observation_to_state,
)
from ironzero.training.vec_rwpp import VecRwppEnv

MAP_PX = 2600.0


def make_micro() -> MicroDuelConfig:
    return MicroDuelConfig(map_width=MAP_PX, map_height=MAP_PX, episode_ticks=30)


def obs_message(
    tick: int,
    own_hp: list[float],
    enemy_hp: list[float],
    *,
    own_xy: tuple[float, float] = (640.0, 2000.0),
    enemy_xy: tuple[float, float] = (640.0, 200.0),
    own_positions: list[tuple[float, float]] | None = None,
    enemy_positions: list[tuple[float, float]] | None = None,
    result: str | None = None,
    terminated: bool = False,
) -> dict:
    def entities(
        hp_values: list[float],
        positions: list[tuple[float, float]] | None,
        base_xy: tuple[float, float],
        team: int,
        base_id: int,
    ) -> list[dict]:
        items = []
        for index, hp in enumerate(hp_values):
            if hp <= 0.0:
                continue
            if positions is not None:
                x, y = positions[index]
            else:
                x, y = base_xy[0] + index * 20.0, base_xy[1]
            items.append(
                {
                    "entity_id": base_id + index,
                    "team_id": team,
                    "x": x,
                    "y": y,
                    "hp": hp,
                    "max_hp": 210.0,
                    "is_alive": True,
                }
            )
        return items

    own = entities(own_hp, own_positions, own_xy, 0, 100)
    enemies = entities(enemy_hp, enemy_positions, enemy_xy, 1, 200)
    return {
        "protocol": "ironzero/1",
        "type": "obs",
        "message_id": f"obs-{tick}",
        "episode_id": "fake",
        "tick": tick,
        "game_time": tick / 60.0,
        "controlled_team": 0,
        "global_state": {"unit_count": len(own) + len(enemies)},
        "own_entities": own,
        "visible_enemy_entities": enemies,
        "reward": 0.0,
        "result": result,
        "terminated": terminated,
        "truncated": False,
    }


# ----------------------------------------------------------------------
# Pure conversion helpers
# ----------------------------------------------------------------------
class RwppConversionTests(unittest.TestCase):
    def test_observation_to_state_encodes_like_mock_engine(self) -> None:
        micro = make_micro()
        message = obs_message(10, [210.0, 210.0], [210.0, 105.0])
        state = observation_to_state(message)
        self.assertEqual(state["tick"], 10)
        self.assertEqual(state["own_entities"][0]["entity_id"], 100)
        self.assertEqual(state["target_ids"], [200, 201])
        obs = encode_observation(state, micro)
        self.assertEqual(obs.shape, (micro.flat_obs_dim,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertGreater(float(obs[3]), 0.0)
        self.assertLessEqual(float(obs[3]), 1.0)
        self.assertGreater(float(obs[4]), 0.0)
        self.assertLessEqual(float(obs[4]), 1.0)
        masks = build_masks(state, micro)
        self.assertTrue(bool(masks["group"][0]))
        self.assertTrue(bool(masks["group"][-1]))
        self.assertTrue(bool(masks["target"][-1]))
        self.assertEqual(int(np.flatnonzero(masks["target"]).size), 3)

    def test_decode_action_heads(self) -> None:
        micro = make_micro()
        state = observation_to_state(obs_message(0, [210.0], [210.0, 210.0]))
        move = decode_action(np.array([0, 2, 7, 7, 0]), state, micro)
        self.assertEqual(move[0]["kind"], "move_group")
        self.assertEqual(move[0]["unit_group"], 2)
        self.assertGreaterEqual(move[0]["x"], 0.0)
        self.assertLessEqual(move[0]["x"], MAP_PX)
        self.assertGreaterEqual(move[0]["y"], 0.0)
        self.assertLessEqual(move[0]["y"], MAP_PX)
        attack_move = decode_action(np.array([1, 4, 0, 0, 0]), state, micro)
        self.assertEqual(attack_move[0]["kind"], "attack_move")
        self.assertEqual(attack_move[0]["unit_group"], 4)  # broadcast slot
        target = decode_action(np.array([2, 1, 0, 0, 1]), state, micro)
        self.assertEqual(target[0]["kind"], "attack_target")
        self.assertEqual(target[0]["unit_group"], 1)
        self.assertEqual(target[0]["target_entity_id"], 201)
        # stale target slot and explicit stop/hold become a group hold order
        for head in (np.array([2, 0, 0, 0, 8]), np.array([3, 3, 0, 0, 0]), np.array([4, 4, 0, 0, 0])):
            hold = decode_action(head, state, micro)
            self.assertEqual(hold[0]["kind"], "hold")
            self.assertIn(hold[0]["unit_group"], (0, 3, 4))

    def test_hp_potential_tracks_damage(self) -> None:
        full = observation_to_state(obs_message(0, [210.0, 210.0], [210.0, 210.0, 210.0]))
        hurt = observation_to_state(obs_message(10, [210.0, 210.0], [210.0, 0.0, 0.0]))
        self.assertAlmostEqual(hp_potential(full), 0.0)
        # episode-start army values: own 2x210, enemy 3x210. Two enemies died, so
        # the enemy side keeps 1/3 of its starting hp while our side is untouched.
        potential = hp_potential(hurt, own_max_ref=420.0, enemy_max_ref=630.0)
        self.assertAlmostEqual(potential, 1.0 - 1.0 / 3.0)

    def test_hp_potential_clamps_reinforcements(self) -> None:
        # The skirmish AI may field extra units: fractions must stay in [0, 1].
        state = observation_to_state(obs_message(10, [210.0], [210.0, 210.0, 210.0, 210.0]))
        potential = hp_potential(state, own_max_ref=210.0, enemy_max_ref=210.0)
        self.assertGreaterEqual(potential, -1.0)
        self.assertLessEqual(potential, 1.0)
        self.assertAlmostEqual(potential, 0.0)  # both sides clamped at 100%

    def test_vision_filter_hides_distant_enemies(self) -> None:
        state = observation_to_state(obs_message(0, [210.0, 210.0], [210.0, 210.0]))
        # own at y=2000, enemies at y=200 -> ~1800 px away, outside a 260 px fog
        filtered = apply_vision(dict(state), 260.0)
        self.assertEqual(filtered["visible_enemy_entities"], [])
        self.assertEqual(filtered["target_ids"], [])
        # the ground truth state is untouched
        self.assertEqual(len(state["visible_enemy_entities"]), 2)
        # without fog nothing is hidden
        visible = apply_vision(dict(state), 0.0)
        self.assertEqual(len(visible["visible_enemy_entities"]), 2)


# ----------------------------------------------------------------------
# Fake bridge + process launcher
# ----------------------------------------------------------------------
class FakeProcess:
    def __init__(self) -> None:
        self.waited = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0

    def kill(self) -> None:
        self.killed = True


class FakeBridge:
    """Accepts game clients and replays a scripted observation/step exchange."""

    def __init__(
        self,
        port: int,
        *,
        steps: int = 3,
        episodes: int = 1,
        eliminate_on_last: bool = False,
        result: str = "timeout",
        result_in_obs: bool = False,
        done_without_obs: bool = False,
        close_after_first_obs: bool = False,
        own_hp: list[float] | None = None,
        enemy_hp: list[float] | None = None,
        enemy_step_hp: list[float] | None = None,
        own_positions: list[tuple[float, float]] | None = None,
        enemy_positions: list[tuple[float, float]] | None = None,
    ) -> None:
        self.port = port
        self.steps = steps
        self.episodes = episodes
        self.eliminate_on_last = eliminate_on_last
        self.result = result
        self.result_in_obs = result_in_obs
        self.done_without_obs = done_without_obs
        self.close_after_first_obs = close_after_first_obs
        self.own_hp = list(own_hp) if own_hp is not None else [210.0, 210.0]
        self.enemy_hp = list(enemy_hp) if enemy_hp is not None else [210.0, 210.0, 210.0]
        self.enemy_step_hp = (
            list(enemy_step_hp) if enemy_step_hp is not None else [120.0, 210.0, 210.0]
        )
        self.own_positions = own_positions
        self.enemy_positions = enemy_positions
        self.process = FakeProcess()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # launcher signature: (game_config, port) -> process handle
    def launcher(self, _game: RwppGameConfig, port: int) -> FakeProcess:
        assert port == self.port, f"bridge {self.port} asked to serve port {port}"
        return self.process

    def _serve(self) -> None:
        for _ in range(self.episodes):
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            try:
                self._serve_client(client)
            finally:
                try:
                    client.close()
                except OSError:
                    pass

    def _serve_client(self, client: socket.socket) -> None:
        reader = client.makefile("r", encoding="utf-8")
        writer = client.makefile("w", encoding="utf-8", newline="\n")

        def send(payload: dict) -> None:
            writer.write(json.dumps(payload) + "\n")
            writer.flush()

        send({"protocol": "ironzero/1", "type": "hello", "message_id": "hello"})
        tick = 0
        send(
            obs_message(
                tick,
                self.own_hp,
                self.enemy_hp,
                own_positions=self.own_positions,
                enemy_positions=self.enemy_positions,
            )
        )
        if self.close_after_first_obs:
            # Simulate a game process that dies right after the first frame.
            return
        for step in range(self.steps):
            line = reader.readline()
            if not line:
                return
            message = json.loads(line)
            if message.get("type") == "shutdown":
                return
            tick += 10
            enemy_hp = list(self.enemy_step_hp)
            if step == self.steps - 1 and self.eliminate_on_last:
                enemy_hp = [0.0] * len(enemy_hp)
            send(
                {
                    "protocol": "ironzero/1",
                    "type": "receipt",
                    "message_id": f"receipt-{tick}",
                    "tick": tick,
                    "decision_id": message.get("decision_id"),
                    "accepted": True,
                    "accepted_count": 1,
                    "rejected_count": 0,
                }
            )
            last = step == self.steps - 1
            terminal = last and self.result_in_obs and self.result != "timeout"
            if not (last and self.done_without_obs):
                send(
                    obs_message(
                        tick,
                        self.own_hp,
                        enemy_hp,
                        own_positions=self.own_positions,
                        enemy_positions=self.enemy_positions,
                        result=self.result if terminal else None,
                        terminated=terminal,
                    )
                )
            if last:
                send(
                    {
                        "protocol": "ironzero/1",
                        "type": "done",
                        "message_id": f"done-{tick}",
                        "tick": tick,
                        "result": self.result,
                        "terminated": self.result != "timeout",
                        "truncated": self.result == "timeout",
                    }
                )
                return


def make_env(bridge: FakeBridge, port: int) -> RwppDuelEnv:
    game = RwppGameConfig(episode_ticks=30)
    return RwppDuelEnv(
        game,
        port=port,
        micro_config=make_micro(),
        launcher=bridge.launcher,
        episode_prefix="test",
    )


class RwppBridgeLoopTests(unittest.TestCase):
    def test_env_reset_and_step_reward_and_truncation(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=3)
        env = make_env(bridge, port)
        try:
            obs, info = env.reset()
            self.assertEqual(obs.shape, (make_micro().flat_obs_dim,))
            self.assertEqual(info["tick"], 0)
            rewards = []
            done = False
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
                rewards.append(reward)
                self.assertEqual(info["actions"][0]["kind"], "attack_target")
                done = terminated or truncated
            self.assertTrue(done)
            self.assertTrue(any(reward != 0.0 for reward in rewards))
            self.assertIn(env.last_result, {"victory", "timeout", "defeat"})
        finally:
            env.close()

    def test_env_terminates_on_elimination(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, eliminate_on_last=True)
        env = make_env(bridge, port)
        try:
            env.reset()
            env.step(np.array([2, 0, 0, 0, 0]))
            _obs, _reward, terminated, _truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertEqual(info["enemy_alive"], 0)
            self.assertTrue(terminated)
            self.assertEqual(info["result"], "victory")
        finally:
            env.close()

    def test_fog_does_not_end_episode_with_hidden_enemies(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, eliminate_on_last=True)
        game = RwppGameConfig(episode_ticks=30, vision_radius=260.0)
        env = RwppDuelEnv(
            game,
            port=port,
            micro_config=make_micro(),
            launcher=bridge.launcher,
            episode_prefix="fog",
        )
        try:
            env.reset()
            _obs, _reward, terminated, _truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertFalse(terminated, "hidden enemies must not count as eliminated")
            self.assertEqual(info["enemy_alive"], 3)
            self.assertEqual(info["visible_enemies"], 0)
            _obs, _reward, terminated, _truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertTrue(terminated)
            self.assertEqual(info["result"], "victory")
        finally:
            env.close()

    def test_env_honors_bridge_victory(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, result="victory", result_in_obs=True)
        env = make_env(bridge, port)
        try:
            env.reset()
            env.step(np.array([2, 0, 0, 0, 0]))
            _obs, reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["result"], "victory")
            self.assertEqual(info["bridge_result"], "victory")
            self.assertGreater(reward, 0.0)
        finally:
            env.close()

    def test_env_honors_bridge_defeat_from_done_message(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, result="defeat", done_without_obs=True)
        env = make_env(bridge, port)
        try:
            env.reset()
            env.step(np.array([2, 0, 0, 0, 0]))
            _obs, reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["result"], "defeat")
            self.assertEqual(info["bridge_result"], "defeat")
            self.assertLess(reward, 0.0)
        finally:
            env.close()

    def test_env_treats_tick_cap_as_timeout(self) -> None:
        """The tick cap truncates; it does not invent a winner.

        The armies are deliberately balanced here: an unbalanced finish is
        adjudicated as victory/defeat on purpose (see
        ``test_timeout_adjudication``), so this test uses 2-vs-2 with equal hp to
        isolate the truncation path.
        """

        port = find_free_port()
        bridge = FakeBridge(
            port,
            steps=2,
            result="timeout",
            done_without_obs=True,
            own_hp=[210.0, 210.0],
            enemy_hp=[210.0, 210.0],
            enemy_step_hp=[210.0, 210.0],
        )
        env = make_env(bridge, port)
        try:
            env.reset()
            env.step(np.array([2, 0, 0, 0, 0]))
            _obs, _reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["result"], "timeout")
        finally:
            env.close()

    def test_env_fails_fast_when_bridge_closes(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, close_after_first_obs=True)
        env = make_env(bridge, port)
        try:
            env.reset()
            started = time.time()
            _obs, _reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            elapsed = time.time() - started
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            # A dropped connection is adjudicated on the army we can still see
            # (2 own vs 3 enemy here -> a defeat), not written off as a draw.
            # It used to report ``timeout`` purely because the stream stopped,
            # which on the real game turned obvious losses into 0-reward draws.
            self.assertEqual(info["result"], "defeat")
            self.assertTrue(info["disconnected"])
            # But a disconnect must never *award a win*: see
            # ``test_a_disconnect_never_awards_a_win``.
            self.assertNotEqual(info["result"], "victory")
            # The old code spun for the full 60 s obs_timeout here.
            self.assertLess(elapsed, 5.0)
        finally:
            env.close()

    def test_env_raises_when_bridge_never_opens(self) -> None:
        game = RwppGameConfig(episode_ticks=10, start_timeout=0.4)
        env = RwppDuelEnv(
            game,
            port=find_free_port(),
            micro_config=make_micro(),
            launcher=lambda _game, _port: FakeProcess(),
        )
        try:
            with self.assertRaises(RwppEnvError):
                env.reset()
        finally:
            env.close()


class RwppEnvFogTests(unittest.TestCase):
    def test_enemy_hp_reference_uses_ground_truth_not_visible_subset(self) -> None:
        # Enemies spawn far away: with a 260 px fog the policy sees none of them.
        # The hp reference must still be the full 3-unit enemy army, otherwise
        # every later enemy_hp_frac would be normalised against nothing.
        port = find_free_port()
        bridge = FakeBridge(port, steps=2)
        game = RwppGameConfig(episode_ticks=30, vision_radius=260.0)
        env = RwppDuelEnv(
            game,
            port=port,
            micro_config=make_micro(),
            launcher=bridge.launcher,
            episode_prefix="fogref",
        )
        try:
            _obs, info = env.reset()
            self.assertEqual(info["visible_enemies"], 0)
            self.assertAlmostEqual(env.own_max_ref, 2 * 210.0)
            self.assertAlmostEqual(env.enemy_max_ref, 3 * 210.0)
            self.assertAlmostEqual(info["max_refs"]["enemy"], 630.0)
            _obs, _reward, _terminated, _truncated, _info = env.step(np.array([1, 4, 0, 0, 0]))
            self.assertAlmostEqual(env.enemy_max_ref, 630.0, msg="reference must not shrink after a step")
        finally:
            env.close()

    def test_fog_target_slot_decodes_against_visible_enemy_list(self) -> None:
        # Only enemy id=201 is near our army; 200/202 are across the map. Target
        # slot 0 refers to the visible list, so the decodes action must target
        # 201 (the ground-truth list would wrongly pick 200).
        port = find_free_port()
        bridge = FakeBridge(
            port,
            steps=2,
            enemy_positions=[(640.0, 200.0), (660.0, 2000.0), (640.0, 200.0)],
        )
        game = RwppGameConfig(episode_ticks=30, vision_radius=260.0)
        env = RwppDuelEnv(
            game,
            port=port,
            micro_config=make_micro(),
            launcher=bridge.launcher,
            episode_prefix="fogdecode",
        )
        try:
            env.reset()
            visible_ids = [
                int(entity["entity_id"])
                for entity in env.obs_state.get("visible_enemy_entities", [])
            ]
            self.assertEqual(visible_ids, [201])
            _obs, _reward, _terminated, _truncated, info = env.step(
                np.array([2, 4, 0, 0, 0])
            )
            self.assertEqual(info["actions"][0]["kind"], "attack_target")
            self.assertEqual(info["actions"][0]["target_entity_id"], 201)
        finally:
            env.close()


class RwppEnvDoneTests(unittest.TestCase):
    def test_done_message_updates_terminal_tick_and_observation(self) -> None:
        port = find_free_port()
        bridge = FakeBridge(port, steps=2, result="timeout", done_without_obs=True)
        env = make_env(bridge, port)
        try:
            obs, info = env.reset()
            self.assertEqual(info["tick"], 0)
            env.step(np.array([2, 0, 0, 0, 0]))
            obs, _reward, terminated, truncated, info = env.step(np.array([2, 0, 0, 0, 0]))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            # The done message carried tick=20; the returned observation/state
            # must describe that terminal frame, not the previous decision.
            self.assertEqual(info["tick"], 20)
            self.assertEqual(env.last_tick, 20)
            self.assertEqual(obs.shape, (make_micro().flat_obs_dim,))
            self.assertEqual(len(env.masks["target"]), make_micro().target_slots)
            self.assertTrue(env.masks["target"][-1])
        finally:
            env.close()


class VecRwppEnvTests(unittest.TestCase):
    def test_vec_env_reset_step_and_auto_reset(self) -> None:
        base = find_free_port_pair()
        bridges = {
            base: FakeBridge(base, steps=2, episodes=3, eliminate_on_last=True),
            base + 1: FakeBridge(base + 1, steps=2, episodes=3, eliminate_on_last=True),
        }

        def launcher(_game: RwppGameConfig, port: int):
            return bridges[port].process

        game = RwppGameConfig(episode_ticks=30)
        vec = VecRwppEnv(
            2,
            game=game,
            port_base=base,
            micro_config=make_micro(),
            launcher=launcher,
        )
        try:
            obs, masks, _infos = vec.reset()
            self.assertEqual(obs.shape, (2, make_micro().flat_obs_dim))
            self.assertEqual(len(masks), 2)
            batch = vec.step(np.zeros((2, 5), dtype=np.int64))
            self.assertEqual(batch.rewards.shape, (2,))
            batch = vec.step(np.zeros((2, 5), dtype=np.int64))
            self.assertTrue(bool(batch.terminated.any() or batch.truncated.any()))
            completed = vec.drain_completed()
            self.assertTrue(completed, "finished episodes must be reported")
            self.assertGreaterEqual(completed[0]["decisions"], 1)
            # auto-reset happened: observations and masks are populated again
            self.assertEqual(batch.observations.shape, (2, make_micro().flat_obs_dim))
            self.assertTrue(all(masks for masks in vec.masks))
        finally:
            vec.close()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def find_free_port_pair() -> int:
    for _ in range(200):
        base = find_free_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", base + 1))
            sock.close()
            return base
        except OSError:
            sock.close()
            continue
    raise RuntimeError("no free consecutive port pair")


if __name__ == "__main__":
    unittest.main()
