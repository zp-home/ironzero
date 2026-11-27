"""Trace whether produceUnit reaches Factory.executeSpecialAction or the fallback.

``probe_ai_queue_vs_ours`` proved the difference is real and not a reporting quirk:

    OURS    : peak mobile 0, every queue item at progress -1.0 (3347 samples)
    AI PILOT: peak mobile 13, current item advances 0.017 -> ... -> 0.989

``produceUnit`` was changed to call ``Factory.executeSpecialAction`` -> the engine's
real production path, with a CommandControllerCommand fallback for custom units.
But the army is still empty, so we have to know which branch actually ran. If the
factory is a modded unit implementing FactoryInterface without extending Factory,
the instanceof test fails and the old (broken) fallback is still in use.

This adds a one-line trace to both branches, gated by the existing ``--debug-build``
flag so normal training stays quiet.

Usage::

    python scripts/patch_produce_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

OLD = '''        try {
            if (factory instanceof com.corrodinggames.rts.game.units.buildings.Factory) {
                ((com.corrodinggames.rts.game.units.buildings.Factory) factory)
                        .executeSpecialAction(action, false);
                return true;
            }
            // ``FactoryInterface`` has no executeSpecialAction -- it declares only
            // queue introspection -- so a custom unit that implements the interface
            // without extending Factory falls through to the command path below.
        } catch (RuntimeException failure) {
            // Fall through to the command path rather than reporting a phantom
            // rejection: the queue may still accept the order on older builds.
        }'''

NEW = '''        try {
            if (factory instanceof com.corrodinggames.rts.game.units.buildings.Factory) {
                ((com.corrodinggames.rts.game.units.buildings.Factory) factory)
                        .executeSpecialAction(action, false);
                debugBuild(factory.id, "produce:executeSpecialAction", unitTypeName);
                return true;
            }
            // ``FactoryInterface`` has no executeSpecialAction -- it declares only
            // queue introspection -- so a custom unit that implements the interface
            // without extending Factory falls through to the command path below.
            debugBuild(factory.id,
                    "produce:NOT_FACTORY (" + factory.getClass().getName() + ")", unitTypeName);
        } catch (RuntimeException failure) {
            debugBuild(factory.id,
                    "produce:executeSpecialAction threw " + failure.getClass().getName(),
                    unitTypeName);
            // Fall through to the command path rather than reporting a phantom
            // rejection: the queue may still accept the order on older builds.
        }'''

OLD2 = '''        command.setActionId(action.getActionId());
        return true;
    }

    /**
     * Own factory for {@code factoryId};'''

NEW2 = '''        command.setActionId(action.getActionId());
        debugBuild(factory.id, "produce:COMMAND_FALLBACK actionId=" + action.getActionId(),
                unitTypeName);
        return true;
    }

    /**
     * Own factory for {@code factoryId};'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "produce:executeSpecialAction" in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"factory branch occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)
    assert src.count(OLD2) == 1, f"fallback occurrences: {src.count(OLD2)}"
    src = src.replace(OLD2, NEW2)

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched produce tracing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
