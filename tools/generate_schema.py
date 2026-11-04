# SPDX-License-Identifier: Apache-2.0
"""Generate the checked-in RWRL v1 JSON Schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwrl.protocol import message_schema  # noqa: E402


def main() -> None:
    output = ROOT / "schemas" / "rwrl" / "1" / "messages.schema.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(message_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
