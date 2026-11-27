"""Expose the engine's built-not-transported flag on factories in the observation.

The decisive comparison (scripts/run_factory_diag.py) showed our queue wedged on
factory 27 with ``builtNotTransported=false, buildProgress=0.0`` while the AI's
working queue advanced. Both modes had the same factory state -- so the difference
is *which factory they queue at*:

  * ours  : every order went to factory 27 (advertising a menu, but not built)
  * AI    : queued at whichever factory had actually finished construction

``Factory.update`` returns immediately when ``isBuiltAndNotTransported()`` is false,
so a queue item at an unfinished factory never advances. The Python side could not
tell: ``can_produce`` was non-empty and ``is_operational`` was true for an
under-construction factory, so it looked usable.

This adds ``is_built`` to each factory entry, straight from the engine's own
``isBuiltAndNotTransported()``, so the curriculum can select only factories that
can actually run a build queue.

Usage::

    python scripts/patch_factory_built_flag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

OLD = '''            java.util.List<String> names = factoryCanProduceNow(unit)
                    ? producibleNamesFor(unit)
                    : java.util.Collections.<String>emptyList();'''

NEW = '''            // Whether this factory can actually WORK on a queued item, not merely
            // whether it advertises a build menu. ``Factory.update`` returns
            // immediately while ``isBuiltAndNotTransported()`` is false, so a queue
            // item placed here would sit at progress -1.0 forever. Measured on Lake
            // at difficulty 1: every curriculum order went to factory 27 with
            // ``builtNotTransported=false, buildProgress=0.0`` while the AI queued
            // at whichever factory had finished, producing 18 units vs our zero.
            boolean builtNotTransported = false;
            try {
                builtNotTransported = unit.isBuiltAndNotTransported();
            } catch (RuntimeException ignored) {
                // Older engine: report false so callers never queue at an unknown
                // factory. A conservative false cannot create a wedged queue.
            }
            out.append(",\\"is_built\\":").append(builtNotTransported);
            java.util.List<String> names = factoryCanProduceNow(unit)
                    ? producibleNamesFor(unit)
                    : java.util.Collections.<String>emptyList();'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if '"is_built"' in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"anchor occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)
    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched factory is_built flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
