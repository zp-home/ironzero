# SPDX-License-Identifier: Apache-2.0
"""Locate which engine classes reference the AI's aggression gates.

The AI's "should I attack yet" decision is spread across a few engine methods.
Knowing which class reads them is the precondition for changing anything: a knob
that nothing reads is a no-op, and this project already paid for that mistake once
(``aiDifficulty`` without ``aiDifficultyLocked`` did nothing at all).

Run: python scripts/find_ai_gates.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

JAR = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15\lib\game-lib-v3-1.15.jar")

TARGETS = [
    b"getAIDifficultyArmyValue",
    b"isTurtling",
    b"aggressiveMode",
    b"desiredUnitCount",
    b"hasDesiredUnitCount",
    b"attackDelayTimer",
    b"stagingStartDelayTimer",
    b"attackGroupCount",
    b"getArmyValue",
    b"AIUnitGroup",
]


def main() -> int:
    if not JAR.is_file():
        print(f"missing jar: {JAR}")
        return 1
    with zipfile.ZipFile(JAR) as archive:
        for name in archive.namelist():
            if not name.endswith(".class"):
                continue
            data = archive.read(name)
            hits = [target.decode() for target in TARGETS if target in data]
            if hits:
                print(f"{name:70s} {', '.join(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
