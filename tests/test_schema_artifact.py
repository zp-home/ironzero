# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "ironzero" / "1" / "messages.schema.json"


class SchemaArtifactTests(unittest.TestCase):
    def test_schema_is_valid_json_and_versioned(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["x-schema-version"], "1.0.0")
        self.assertEqual(schema["x-license"], "Apache-2.0")
        self.assertEqual(schema["discriminator"]["propertyName"], "type")

    def test_schema_contains_all_message_types(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        defs = schema["$defs"]
        for name in [
            "HelloMessage",
            "ResetRequest",
            "StepRequest",
            "ObservationMessage",
            "ReceiptMessage",
            "DoneMessage",
            "ErrorMessage",
            "HeartbeatMessage",
            "ShutdownMessage",
        ]:
            self.assertIn(name, defs)

    def test_schema_is_deterministic(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertGreater(len(canonical), 1000)


if __name__ == "__main__":
    unittest.main()
