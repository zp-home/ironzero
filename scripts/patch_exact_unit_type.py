"""produceUnit must resolve the unit type against the factory's OWN menu, exactly.

Measured with scripts/probe_produce_acceptance.py on Lake at difficulty 1:

    ordered : {"kind": "produce_unit", "factory_entity_id": 27, "unit_type": "tank"}
    receipt : ok            (produceUnit returned true)
    queue   : [["c_tank", -1.0]]        <- a different type, never started

The observation's menu for factory 27 lists ``tank``; the queue item is ``c_tank``.
``produceUnit`` resolves the requested name with ``resolveUnitType`` ->
``UnitType.findUnitType(name)``, which is the engine's *lookup by string* and can
return a different type than the one the menu advertised (``c_tank`` is a separate
registry entry from ``tank``). The factory then had an action for that other type,
``produceUnit`` reported success, and the queued item never advanced past
``progress = -1.0`` -- "not started" -- for the rest of the episode. Because
``production_queue_depth`` counts it, the stage read ``queue_full`` on 771 of 900
decisions and the army stayed at zero while three land factories stood idle.

The fix is to make the two agree by construction: pick the ``UnitType`` whose
internal name **exactly** matches an entry of this factory's own producible menu
(the same list ``factoriesJson`` reports, produced by ``producibleNamesFor``), and
only fall back to the fuzzy registry lookup when no exact match exists (custom or
aliased units, where the fuzzy path is the only one available).

Usage::

    python scripts/patch_exact_unit_type.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"

OLD = '''    private boolean produceUnit(long factoryId, String unitTypeName) {
        UnitTypeInterface unitType = resolveUnitType(unitTypeName);
        if (unitType == null) {
            return false;
        }
        OrderableUnit factory = findOwnFactory(factoryId);
        if (factory == null) {
            return false;
        }'''

NEW = '''    private boolean produceUnit(long factoryId, String unitTypeName) {
        OrderableUnit factory = findOwnFactory(factoryId);
        if (factory == null) {
            return false;
        }
        // Resolve the requested type against THIS factory's own menu, by exact name.
        //
        // ``UnitType.findUnitType(name)`` is the engine's string lookup and is not
        // guaranteed to return the type the menu advertised: measured on Lake at
        // difficulty 1, ordering ``tank`` at a factory whose menu lists ``tank``
        // queued ``c_tank`` instead (a separate registry entry), the receipt said
        // "ok", and the queue item sat at progress -1.0 for the whole episode. A
        // queue that never drains holds every ``production_queue_depth`` slot, so the
        // stage read ``queue_full`` on 771 of 900 decisions with three factories
        // idle and the army at zero.
        //
        // Matching the menu first makes the dispatched type and the advertised type
        // the same object, which is what "the policy asked for a unit the factory
        // offers" is supposed to mean. The fuzzy lookup stays as the fallback for
        // custom or aliased units, which never appear in ``UnitType.values()``.
        UnitTypeInterface unitType = exactProducibleType(factory, unitTypeName);
        if (unitType == null) {
            unitType = resolveUnitType(unitTypeName);
        }
        if (unitType == null) {
            return false;
        }'''

HELPER_ANCHOR = '''    private boolean produceUnit(long factoryId, String unitTypeName) {'''

HELPER = '''    /**
     * The {@link UnitType} in this factory's own producible menu whose internal name
     * equals {@code name}, or {@code null} when the menu has no such entry.
     *
     * <p>Deliberately an exact match: the whole point is to hand
     * {@link #produceUnit} the same type the observation advertised, so a receipt of
     * "accepted" cannot describe a unit the policy did not ask for. See the caller
     * for the measurement that motivated this.
     */
    private UnitTypeInterface exactProducibleType(OrderableUnit factory, String name) {
        if (name == null || name.isEmpty()) {
            return null;
        }
        UnitType[] types;
        try {
            types = UnitType.values();
        } catch (RuntimeException ignored) {
            return null;
        }
        for (UnitType candidate : types) {
            if (candidate == null) {
                continue;
            }
            String internal;
            try {
                internal = candidate.getInternalName();
            } catch (RuntimeException ignored) {
                continue;
            }
            if (internal == null || !internal.equals(name)) {
                continue;
            }
            try {
                if (factory.findActionForUnitType(candidate) != null) {
                    return candidate;
                }
            } catch (RuntimeException ignored) {
                // Unqueryable: keep looking rather than aborting the whole menu.
            }
        }
        return null;
    }

    private boolean produceUnit(long factoryId, String unitTypeName) {'''


def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "exactProducibleType" in src:
        print("already patched")
        return 0

    assert src.count(OLD) == 1, f"produceUnit head occurrences: {src.count(OLD)}"
    src = src.replace(OLD, NEW)

    assert src.count(HELPER_ANCHOR) == 1, f"anchor occurrences: {src.count(HELPER_ANCHOR)}"
    src = src.replace(HELPER_ANCHOR, HELPER)

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched produceUnit to resolve types against the factory menu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
