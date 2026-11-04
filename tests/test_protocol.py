# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest

from rwrl.protocol import (
    ActionMasks,
    ActionKind,
    AttackMoveAction,
    DoneMessage,
    DoneResult,
    Entity,
    ErrorCode,
    GlobalState,
    HelloMessage,
    MoveGroupAction,
    ObservationMessage,
    ProtocolError,
    ReceiptMessage,
    ResetRequest,
    SequenceGuard,
    StepRequest,
    canonical_json,
    dump_message,
    message_hash,
    message_schema,
    parse_message,
)


class ProtocolTests(unittest.TestCase):
    def test_hello_roundtrip(self) -> None:
        message = HelloMessage(
            message_id="m-1",
            adapter_version="0.1.0",
            game_version="test",
            capabilities=["ndjson", "actions"],
        )
        wire = canonical_json(dump_message(message))
        parsed = parse_message(wire)
        self.assertEqual(parsed.type, "hello")
        self.assertEqual(parsed.capabilities, ["ndjson", "actions"])
        self.assertEqual(message_hash(parsed), message_hash(message))

    def test_step_accepts_discriminated_actions(self) -> None:
        message = parse_message(
            {
                "protocol": "rwrl/1",
                "type": "step",
                "message_id": "m-2",
                "episode_id": "e-1",
                "tick": 12,
                "decision_id": "d-1",
                "actions": [
                    {"kind": "move_group", "unit_group": 0, "x": 10.5, "y": 20.0},
                    {"kind": "attack_move", "unit_group": 1, "x": 3, "y": 4},
                ],
            }
        )
        self.assertIsInstance(message, StepRequest)
        self.assertIsInstance(message.actions[0], MoveGroupAction)
        self.assertIsInstance(message.actions[1], AttackMoveAction)

    def test_attack_target_requires_target_id(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_message(
                {
                    "protocol": "rwrl/1",
                    "type": "step",
                    "message_id": "m-3",
                    "episode_id": "e-1",
                    "tick": 12,
                    "decision_id": "d-2",
                    "actions": [{"kind": "attack_target", "unit_group": 0}],
                }
            )

    def test_unknown_action_kind_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_message(
                {
                    "protocol": "rwrl/1",
                    "type": "step",
                    "message_id": "m-4",
                    "episode_id": "e-1",
                    "tick": 12,
                    "decision_id": "d-3",
                    "actions": [{"kind": "teleport", "unit_group": 0}],
                }
            )

    def test_observation_contract(self) -> None:
        observation = ObservationMessage(
            message_id="m-5",
            episode_id="e-1",
            tick=12,
            decision_id="d-4",
            game_time=0.4,
            controlled_team=0,
            visibility_revision=3,
            observation_schema_version="1.0.0",
            global_state=GlobalState(funds=1000, income=20),
            own_entities=[
                Entity(entity_id=1, entity_type_id=7, team_id=0, x=4.0, y=8.0)
            ],
            masks=ActionMasks(
                action_mask=[True, True, False],
                unit_group_mask=[True],
                target_mask=[False],
                position_mask=[[True, False]],
            ),
        )
        dumped = dump_message(observation)
        self.assertEqual(dumped["type"], "obs")
        self.assertEqual(dumped["own_entities"][0]["entity_id"], 1)

    def test_receipt_requires_result_details(self) -> None:
        with self.assertRaises(Exception):
            ReceiptMessage(
                message_id="m-6",
                episode_id="e-1",
                tick=13,
                decision_id="d-4",
                accepted=False,
            )

    def test_sequence_guard_rejects_stale_and_duplicate(self) -> None:
        guard = SequenceGuard("e-1")
        first = StepRequest(
            message_id="m-7",
            episode_id="e-1",
            tick=10,
            decision_id="d-5",
            actions=[MoveGroupAction(kind=ActionKind.MOVE_GROUP, unit_group=0, x=1, y=2)],
        )
        guard.check(first)
        with self.assertRaises(ProtocolError) as duplicate:
            guard.check(first)
        self.assertEqual(duplicate.exception.code, ErrorCode.DUPLICATE_DECISION)

        stale = StepRequest(
            message_id="m-8",
            episode_id="e-1",
            tick=9,
            decision_id="d-6",
            actions=[MoveGroupAction(kind=ActionKind.MOVE_GROUP, unit_group=0, x=1, y=2)],
        )
        with self.assertRaises(ProtocolError) as stale_error:
            guard.check(stale)
        self.assertEqual(stale_error.exception.code, ErrorCode.STALE_TICK)

        other_episode = ResetRequest(
            message_id="m-9",
            episode_id="e-2",
            seed=1,
            scenario="micro",
            map_id="map",
        )
        with self.assertRaises(ProtocolError) as episode_error:
            guard.check(other_episode)
        self.assertEqual(episode_error.exception.code, ErrorCode.STALE_EPISODE)

    def test_done_message(self) -> None:
        message = DoneMessage(
            message_id="m-10",
            episode_id="e-1",
            tick=100,
            result=DoneResult.VICTORY,
        )
        self.assertEqual(dump_message(message)["result"], "victory")

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_message(
                {
                    "protocol": "rwrl/1",
                    "type": "hello",
                    "message_id": "m-11",
                    "adapter_version": "0.1.0",
                    "game_version": "test",
                    "unexpected": True,
                }
            )

    def test_schema_has_discriminator_and_defs(self) -> None:
        schema = message_schema()
        self.assertEqual(schema["discriminator"]["propertyName"], "type")
        self.assertIn("$defs", schema)
        self.assertIn("StepRequest", schema["$defs"])
        self.assertIn("MoveGroupAction", schema["$defs"])
        self.assertEqual(schema["x-schema-version"], "1.0.0")

    def test_invalid_json_is_protocol_error(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_message("{not json")

    def test_canonical_json_is_stable(self) -> None:
        first = canonical_json({"b": 2, "a": 1})
        second = canonical_json({"a": 1, "b": 2})
        self.assertEqual(first, second)
        self.assertEqual(first, '{"a":1,"b":2}')


if __name__ == "__main__":
    unittest.main()
