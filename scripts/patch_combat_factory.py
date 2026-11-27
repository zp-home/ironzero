"""Patch the combat-factory predicate and the production target selection.

Two coupled defects, both measured at the end of a full episode on Lake at
difficulty 1 (scripts/probe_endgame_buildings.py):

    own_buildings : commandCenter 1, extractorT1 2, landFactory 3, c_turret_t1 2
    own_mobile    : scout 6          <- three land factories, zero tanks
    enemy_mobile  : 17 units incl. tanks, artillery, gunships
    closest_approach_px : 3.0        <- they do fight, and lose

1. ``combat_factories`` accepted the command centre.
   The predicate was "any menu entry that is not ``builder``", and the command
   centre's menu is ``[builder, scout]``, so it qualified. The command centre is
   not a combat factory: it exists to make builders, and its scout is the weakest
   unit on the map.

2. Production picked the first option belonging to *any* combat factory.
   ``produce_options`` orders entries by factory, so the command centre's
   ``(23, scout)`` came first and every production order was "make a scout". The
   three land factories were never asked to produce anything, which is why the army
   was six scouts.

Together they cap the army at the weakest unit in the game while the enemy fields
tanks and aircraft -- an unwinnable configuration no amount of PPO can fix.

The fix: exclude the command centre from ``combat_factories`` and prefer the
*strongest* affordable combat unit from a real factory rather than the first entry.

Usage::

    python scripts/patch_combat_factory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "src" / "ironzero" / "rwpp" / "env.py"

OLD_COMBAT = '''        combat_factories = [
            entry
            for entry in (state.get("factories") or [])
            if isinstance(entry, dict)
            and any(
                str(name).lower() != "builder"
                for name in (entry.get("can_produce") or [])
            )
        ]'''

NEW_COMBAT = '''        combat_factories = _combat_factories(state)'''

OLD_TARGET = '''        options = produce_options(state)
        combat_ids = {
            int(entry.get("entity_id", -1))
            for entry in combat_factories
            if entry.get("entity_id") is not None
        }
        target_index = None
        for index, (factory_id, _unit) in enumerate(options):
            if factory_id in combat_ids:
                target_index = index
                break
        if target_index is None:'''

NEW_TARGET = '''        options = produce_options(state)
        combat_ids = {
            int(entry.get("entity_id", -1))
            for entry in combat_factories
            if entry.get("entity_id") is not None
        }
        # Prefer the *strongest* combat unit a real factory can make, not the first
        # entry in the menu.
        #
        # ``produce_options`` walks factories in payload order and, inside each, the
        # menu order, so "first entry" is really "whatever the command centre lists
        # first" -- measured as ``(23, scout)`` while three land factories sat idle.
        # Ranking by the unit's combat value is what turns three factories into
        # tanks instead of six scouts.
        target_index = None
        best_rank = None
        for index, (factory_id, unit_name) in enumerate(options):
            if factory_id not in combat_ids:
                continue
            rank = _unit_combat_rank(str(unit_name))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                target_index = index
        if target_index is None:'''

HELPERS = '''
#: Rough combat value per unit type, used to prefer a real fighting unit over the
#: cheapest one. Higher is better. Unknown types rank 0.
#:
#: The numbers are only an ordering, not a balance claim: what matters is that
#: ``scout`` ranks below every actual combat unit, because a menu-driven
#: "first entry" policy was measured to build nothing but scouts while three land
#: factories stood idle (``.runtime`` endgame probe on Lake at difficulty 1).
_UNIT_COMBAT_RANK: dict[str, int] = {
    "scout": 1,
    "hoverTank": 3,
    "tank": 4,
    "c_tank": 5,
    "artillery": 5,
    "c_artillery": 6,
    "lightGunship": 4,
    "c_helicopter": 5,
    "gunship": 5,
    "bomber": 6,
    "c_bomber": 6,
}


def _unit_combat_rank(name: str) -> int:
    """Combat value of a producible unit name; 0 when unknown."""
    return _UNIT_COMBAT_RANK.get(name.split(".")[-1], 0)


def _combat_factories(state: dict) -> list[dict]:
    """Factories that can produce a *real* combat unit.

    A factory qualifies only when its menu contains something better than a scout,
    and the command centre never qualifies. That building's purpose is to make
    builders; its ``scout`` entry is the weakest unit on the map, so counting it as
    a combat factory both mislabels the base and -- because
    ``produce_options`` lists it first -- captures every production order.

    Measured consequence of the old predicate on Lake at difficulty 1: three
    ``landFactory`` buildings stood idle for a whole episode while the army was six
    ``scout`` against an enemy holding tanks, artillery and gunships.
    """
    out: list[dict] = []
    for entry in state.get("factories") or []:
        if not isinstance(entry, dict):
            continue
        type_name = str(entry.get("unit_type") or "").lower()
        if "commandcenter" in type_name or "command_centre" in type_name:
            continue
        menu = [str(name) for name in (entry.get("can_produce") or [])]
        if any(_unit_combat_rank(name) > _unit_combat_rank("scout") for name in menu):
            out.append(entry)
    return out

'''

def main() -> int:
    src = PATH.read_text(encoding="utf-8")

    if "_combat_factories(state: dict)" in src:
        print("already patched")
        return 0

    assert src.count(OLD_COMBAT) >= 1, f"combat block occurrences: {src.count(OLD_COMBAT)}"
    src = src.replace(OLD_COMBAT, NEW_COMBAT)

    assert src.count(OLD_TARGET) == 1, f"target block occurrences: {src.count(OLD_TARGET)}"
    src = src.replace(OLD_TARGET, NEW_TARGET)

    # Insert the helpers just before the first top-level class definition.
    anchor = "\nclass "
    index = src.find(anchor)
    assert index > 0, "no class definition found"
    src = src[:index] + "\n" + HELPERS + src[index:]

    PATH.write_text(src, encoding="utf-8", newline="\n")
    print("patched combat-factory predicate + production ranking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
