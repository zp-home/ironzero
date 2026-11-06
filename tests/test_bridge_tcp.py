# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of the NDJSON bridge against the mock duel adapter."""

from __future__ import annotations

import unittest

import numpy as np

from ironzero.bridge import BridgeClient, BridgeError
from ironzero.mock_bridge import MockDuelBridge
from ironzero.mock_engine import MicroDuelEngine
from ironzero.protocol import ErrorMessage, HeartbeatMessage, ResetRequest, StepRequest
from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.tcp_adapter import TcpDuelAdapter


class BridgeHandshakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MockDuelBridge(port=0).start()
        self.addCleanup(self.server.stop)

    def test_handshake_returns_server_capabilities(self) -> None:
        client = BridgeClient(port=self.server.port)
        client.connect()
        self.addCleanup(client.close)
        hello = client.handshake()
        self.assertIn("ndjson", hello.capabilities)
        self.assertEqual(hello.protocol, "ironzero/1")

    def test_reset_and_step_roundtrip(self) -> None:
        adapter = TcpDuelAdapter(port=self.server.port)
        self.addCleanup(adapter.close)
        state = adapter.reset(episode_id="ep-bridge", seed=4)
        self.assertEqual(state["tick"], 0)
        self.assertEqual(len(state["own_entities"]), 8)
        self.assertEqual(len(state["target_ids"]), 8)
        result = adapter.step(
            [{"kind": "attack_target", "unit_group": 0, "target_entity_id": state["target_ids"][0]}]
        )
        self.assertEqual(result["tick"], DEFAULT_CONFIG.decision_interval)
        self.assertTrue(result["receipts"] and result["receipts"][0]["accepted"])
        self.assertTrue(np.isfinite(result["reward"]))

    def test_bridge_matches_in_process_engine(self) -> None:
        adapter = TcpDuelAdapter(port=self.server.port)
        self.addCleanup(adapter.close)
        state = adapter.reset(episode_id="ep-parity", seed=9)
        engine = MicroDuelEngine(opponent="scripted")
        engine.reset(seed=9)
        engine_actions = [{"kind": "attack_target", "unit_group": 0, "target_entity_id": state["target_ids"][0]}]
        for _ in range(12):
            state = adapter.step(engine_actions)
            engine.step_decision(engine_actions)
            self.assertEqual(state["tick"], engine.tick)
            self.assertEqual(len(state["own_entities"]), len(engine.alive_units(0)))
            for over_bridge, local in zip(state["own_entities"], engine.alive_units(0)):
                self.assertAlmostEqual(over_bridge["x"], local.x, places=3)
                self.assertAlmostEqual(over_bridge["hp"], local.hp, places=3)

    def test_stale_tick_is_rejected(self) -> None:
        client = BridgeClient(port=self.server.port)
        client.connect()
        self.addCleanup(client.close)
        client.handshake()
        episode_id = "ep-stale"
        client.send(
            ResetRequest(
                message_id="m-1",
                episode_id=episode_id,
                seed=1,
                scenario="micro_duel",
                map_id="micro_duel_fixed",
            )
        )
        obs = client.recv()
        self.assertEqual(obs.tick, 0)
        client.send(
            StepRequest(
                message_id="m-2",
                episode_id=episode_id,
                tick=999,
                decision_id="d-bad",
                actions=[{"kind": "stop", "unit_group": 0}],
            )
        )
        response = client.recv()
        self.assertIsInstance(response, ErrorMessage)
        self.assertEqual(response.error_code, "STALE_TICK")

    def test_unknown_scenario_is_rejected(self) -> None:
        client = BridgeClient(port=self.server.port)
        client.connect()
        self.addCleanup(client.close)
        client.handshake()
        client.send(
            ResetRequest(
                message_id="m-3",
                episode_id="ep-bad",
                seed=0,
                scenario="does_not_exist",
                map_id="x",
            )
        )
        response = client.recv()
        self.assertIsInstance(response, ErrorMessage)

    def test_heartbeat_does_not_break_connection(self) -> None:
        adapter = TcpDuelAdapter(port=self.server.port)
        self.addCleanup(adapter.close)
        adapter.reset(episode_id="ep-hb", seed=2)
        adapter.client.send(HeartbeatMessage(message_id="hb-1", last_tick=0))
        state = adapter.step([{"kind": "hold", "unit_group": 1}])
        self.assertEqual(state["tick"], DEFAULT_CONFIG.decision_interval)

    def test_client_reports_closed_bridge(self) -> None:
        client = BridgeClient(port=self.server.port, recv_timeout=1.0)
        client.connect()
        client.close()
        with self.assertRaises(BridgeError):
            client.recv()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
