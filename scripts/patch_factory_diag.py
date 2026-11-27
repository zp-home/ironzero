"""Add a one-line factory diagnostic under --debug-build.

Our curriculum and the built-in AI both queue ``c_tank`` at a landFactory, but only
the AI's queue advances. The order path has been ruled out: both use
``executeSpecialAction`` -> ``queueOrCancel``, both pay the price, both produce a
queue item. What has not been measured is the *factory's own state* at that moment.

``Factory.update`` starts work on a queue item only when
``isBuiltAndNotTransported()`` is true. The observation's ``can_produce`` only asks
``isOperational()`` plus a progress check, so a factory can advertise a menu while
still being unable to run its build queue.

This adds ``[ironzero] factory <id> built=<b> operational=<o> progress=<p>
queue=<n>`` lines, gated by the existing ``--debug-build`` flag.

Usage::

    python scripts/patch_factory_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

ANCHOR = '''            out.append(",\\"queued\\":[");
            java.util.List<com.corrodinggames.rts.game.units.buildings.BuildQueueItem> queueItems =
                    queuedItems(unit);'''

NEW = '''            out.append(",\\"queued\\":[");
            java.util.List<com.corrodinggames.rts.game.units.buildings.BuildQueueItem> queueItems =
                    queuedItems(unit);
            if (debugBuild) {
                boolean builtNotTransported = false;
                boolean operational = false;
                float buildProg = -1.0f;
                try {
                    builtNotTransported = unit.isBuiltAndNotTransported();
                    operational = isOperational(unit);
                    buildProg = buildProgress(unit);
                } catch (RuntimeException ignored) {
                    // Older engine: leave the flags at their false defaults.
                }
                System.out.println("[ironzero] factory diag id=" + unit.id
                        + " builtNotTransported=" + builtNotTransported
                        + " operational=" + operational
                        + " buildProgress=" + buildProg
                        + " queue=" + queueItems.size());
            }'''

PROBE_FLAG = '''        win_mode="objective",
        frame_ms=1,'''

PROBE_FLAG_NEW = '''        win_mode="objective",
        debug_build=True,
        frame_ms=1,'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "factory diag id=" in src:
        print("already patched")
        return 0

    assert src.count(ANCHOR) == 1, f"anchor occurrences: {src.count(ANCHOR)}"
    src = src.replace(ANCHOR, NEW)
    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched factory diagnostic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
