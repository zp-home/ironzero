# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the Java bridge wire format.

The bridge is the one component that cannot be exercised by pytest (it needs the
Rust War game engine). These tests therefore assert the *source-level* contract
that caused real, observed bugs, so a regression fails fast in CI instead of at
runtime against the real game:

* receipts must not emit fields absent from ``protocol.ReceiptMessage``
  (``accepted_count`` / ``rejected_count`` previously broke every official
  client with ``extra_forbidden``);
* entity payloads must not emit ``team_id`` values below 0, because
  ``protocol.Entity.team_id`` is ``Field(ge=0)``. Real bridge output sent
  ``-1`` for unowned map features (trees), which made *every* observation fail
  protocol validation;
* unowned map features must be excluded from the enemy list;
* build orders must be pre-validated, otherwise the bridge reports a false
  "accepted" for an order the engine drops one frame later.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = REPO_ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"


class BridgeWireContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.src = BRIDGE.read_text(encoding="utf-8")
        # Strip comments so commented-out code does not affect the assertions.
        cls.code = re.sub(r"//[^\n]*", "", cls.src)
        cls.code = re.sub(r"/\*.*?\*/", "", cls.code, flags=re.S)

    def test_bridge_source_exists(self) -> None:
        self.assertTrue(BRIDGE.is_file(), f"missing bridge source: {BRIDGE}")

    def test_receipts_do_not_emit_non_protocol_fields(self) -> None:
        """ReceiptMessage is extra='forbid'; count fields are not declared."""

        for forbidden in ("accepted_count", "rejected_count"):
            self.assertNotIn(
                forbidden,
                self.code,
                f"{forbidden!r} is not part of protocol.ReceiptMessage "
                f"(extra='forbid') and breaks official client parsing",
            )

    def test_entity_team_id_is_never_negative(self) -> None:
        """protocol.Entity.team_id is Field(ge=0); -1 fails validation."""

        self.assertNotIn(
            "teamId : -1",
            self.code.replace("  ", " "),
            "entityJson must not fall back to a negative team id",
        )
        # The neutral sentinel must exist and be non-negative.
        match = re.search(r"NEUTRAL_TEAM_ID\s*=\s*(\d+)", self.code)
        self.assertIsNotNone(match, "NEUTRAL_TEAM_ID sentinel is missing")
        self.assertGreaterEqual(int(match.group(1)), 0)

    def test_unowned_map_features_are_not_enemies(self) -> None:
        """Trees/rocks (team id < 0) must be skipped, not counted as enemies."""

        # The entity loop must guard on a negative team id before adding.
        self.assertRegex(
            self.code,
            r"unit\.team\s*==\s*null\s*\|\|\s*unit\.team\.teamId\s*<\s*0",
            "unowned map features must be filtered before the own/enemy split",
        )

    def test_build_orders_are_pre_validated(self) -> None:
        """A build order must be checked before it is dispatched.

        Otherwise the bridge emits accepted=true and the engine drops the order
        on the next frame ("can not queue build"), i.e. a false-positive receipt.
        """

        self.assertIn(
            "isValidNewWaypoint",
            self.code,
            "buildWithBuilders must validate the waypoint before dispatching",
        )
        self.assertIn(
            "canQueueAction",
            self.code,
            "buildWithBuilders must check the build menu before dispatching",
        )
        self.assertIn(
            "isLocked",
            self.code,
            "buildWithBuilders must reject tech-locked unit types",
        )

    def test_protocol_structure_names_are_translated(self) -> None:
        """Protocol snake_case names must map onto engine camelCase constants."""

        self.assertIn(
            "resolveUnitTypeName",
            self.code,
            "snake_case structure names need translation to UnitType constants",
        )
        # The engine enum constant is camelCase; the protocol uses snake_case.
        self.assertNotIn(
            'default", "land_factory"',
            self.code,
            "land_factory is not a UnitType constant; use landFactory or translate",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
