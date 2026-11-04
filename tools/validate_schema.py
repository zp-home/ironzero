"""Validate representative messages against the checked-in JSON Schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "rwrl" / "1" / "messages.schema.json"

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:
    raise SystemExit("Install the dev dependency: python -m pip install 'jsonschema>=4.23,<5'") from exc


SAMPLES = [
    {
        "protocol": "rwrl/1",
        "type": "hello",
        "message_id": "m-1",
        "adapter_version": "0.1.0",
        "game_version": "locked-version",
        "capabilities": ["ndjson"],
    },
    {
        "protocol": "rwrl/1",
        "type": "reset",
        "message_id": "m-2",
        "episode_id": "e-1",
        "seed": 42,
        "scenario": "micro_8v8",
        "map_id": "fixed_micro_01",
    },
    {
        "protocol": "rwrl/1",
        "type": "step",
        "message_id": "m-3",
        "episode_id": "e-1",
        "tick": 12,
        "decision_id": "d-1",
        "actions": [
            {"kind": "move_group", "unit_group": 0, "x": 1.0, "y": 2.0}
        ],
    },
    {
        "protocol": "rwrl/1",
        "type": "obs",
        "message_id": "m-4",
        "episode_id": "e-1",
        "tick": 12,
        "decision_id": "d-1",
        "game_time": 0.4,
        "controlled_team": 0,
        "visibility_revision": 1,
        "observation_schema_version": "1.0.0",
        "global_state": {"funds": 1000},
        "masks": {
            "action_mask": [True],
            "unit_group_mask": [True],
            "target_mask": [False],
            "position_mask": [[True]],
        },
    },
    {
        "protocol": "rwrl/1",
        "type": "receipt",
        "message_id": "m-5",
        "episode_id": "e-1",
        "tick": 12,
        "decision_id": "d-1",
        "accepted": True,
        "executed_at_tick": 13,
    },
    {
        "protocol": "rwrl/1",
        "type": "done",
        "message_id": "m-6",
        "episode_id": "e-1",
        "tick": 100,
        "result": "victory",
        "terminated": True,
    },
]


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = []
    for index, sample in enumerate(SAMPLES):
        for error in validator.iter_errors(sample):
            errors.append((index, list(error.path), error.message))
    if errors:
        for index, path, message in errors:
            print(f"sample[{index}] path={path}: {message}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validated {len(SAMPLES)} sample messages")


if __name__ == "__main__":
    main()
