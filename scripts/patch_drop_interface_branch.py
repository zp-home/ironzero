"""Drop the FactoryInterface branch: that interface has no executeSpecialAction.

``FactoryInterface`` declares only queue introspection (isQueueEmpty,
getTotalQueuedCount, getQueuedItems, ...); the production entry point
``executeSpecialAction`` lives on ``Factory``. The compiled check would not have
caught it as a *runtime* problem (the cast is legal and the call would throw
NoSuchMethodError only when reached), so it is removed here rather than left to
fail on a modded factory.

The custom-unit case is still covered: ``CustomUnit`` implements FactoryInterface
but not Factory, and for those the command fallback below applies.

Usage::

    python scripts/patch_drop_interface_branch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

OLD = '''            if (factory instanceof com.corrodinggames.rts.game.units.buildings.FactoryInterface) {
                ((com.corrodinggames.rts.game.units.buildings.FactoryInterface) factory)
                        .executeSpecialAction(action, false);
                return true;
            }
'''

NEW = '''            // ``FactoryInterface`` has no executeSpecialAction -- it declares only
            // queue introspection -- so a custom unit that implements the interface
            // without extending Factory falls through to the command path below.
'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if OLD not in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"branch occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)
    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("removed invalid FactoryInterface branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
