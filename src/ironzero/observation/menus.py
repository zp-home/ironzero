# SPDX-License-Identifier: Apache-2.0
"""What the engine says we may build and produce, as ordered menus.

These are pure reads of one bridge payload. They live in ``observation`` (not in
``rwpp``) because *both* layers need them and only one of them may own the
definition:

* :mod:`ironzero.observation.masks` answers "may the policy select this kind /
  slot at all" from the same list;
* :mod:`ironzero.rwpp.decoding` answers "which structure / unit does target slot
  *i* mean" from the same list.

If the two derived their own menus they would disagree, and the disagreement is
exactly the class of bug that made the real game unplayable: the mask opened a
kind the decode then translated into a different order, so the policy paid
``invalid_penalty`` for obeying its own mask.

Why the *engine* is the source of truth
---------------------------------------
A build menu is not a property of the game, it is a property of the
(builder type x map x tech) triple, and it changes during an episode. Measured
on ``[p2]Lake (2p)``: the map's ``builder`` reports ``can_build = ["landFactory"]``
yet the same builder *accepts* ``structure_type="turret"`` -- because the menu
the engine dispatches through is ``getSpecialAction`` over the whole
``UnitType`` registry, not ``canQueueAction``. Hard-coding either the reported
list or a static "these are the Rusted Warfare buildings" list is therefore
wrong; the bridge reports the dispatchable set (see
``IronZeroBridge.buildableNamesFor``) and these helpers only order it.

Stable ordering is load-bearing
-------------------------------
The target head indexes these lists, so the *i*-th entry must mean the same thing
on every frame of an episode and across episodes of the same map. The bridge
already emits both menus in a deterministic order (engine registry order, with
canonical economy names appended and a per-type cache), and the helpers below
only de-duplicate while preserving first-seen order. They never sort: a sort
would be stable across frames but would renumber the slots the moment a new
building type completed, which is worse.
"""

from __future__ import annotations

from typing import Any

def menu_index(menu: list[str], wanted: str) -> int:
    """Index of ``wanted`` in ``menu``, or 0 when it is not offered.

    Returning 0 rather than -1 is deliberate: the target head has no "invalid"
    encoding for these kinds, and ``-1 % len(menu)`` would silently select the
    *last* entry. Falling back to 0 keeps the behaviour identical to the old
    hard-coded default (the first menu entry), so a curriculum that asks for a
    structure the map cannot offer degrades to exactly the pre-existing order
    instead of to an arbitrary one.

    Matching is case-insensitive on the exact name first, then on a
    prefix/substring so ``"turret"`` finds ``"turretT1"`` and ``"land_factory"``
    finds ``"landFactory"`` -- the protocol and the engine disagree about both
    case and the ``T1`` suffix, and the bridge resolves snake_case itself.
    """

    if not menu:
        return 0
    target = str(wanted or "").strip().lower()
    for index, name in enumerate(menu):
        if name.lower() == target:
            return index
    # ``land_factory`` -> ``landfactory``, which is a substring of ``landfactory``.
    collapsed = target.replace("_", "")
    for index, name in enumerate(menu):
        if collapsed == name.lower().replace("_", ""):
            return index
    for index, name in enumerate(menu):
        if target in name.lower() or name.lower() in target:
            return index
    return 0


def find_structure(menu: list[str], *candidates: str) -> str | None:
    """First offered name matching the earliest candidate, else ``None``.

    Used by the curriculum to answer "give me a factory if one exists, otherwise
    an air factory, otherwise nothing" without hard-coding a single name that a
    particular map may not offer.
    """

    for candidate in candidates:
        index = menu_index(menu, candidate)
        if menu and (menu[index].lower() == candidate.lower() or candidate.lower() in menu[index].lower()):
            return menu[index]
    return None


def is_defence_type(name: str) -> bool:
    """Whether this structure name denotes a static defensive building that SHOOTS.

    Name-based because the *menu* carries names, not entities: a structure the
    policy has not built yet has no ``is_defence`` flag to read.

    ``laserDefence`` is deliberately **excluded**, and that exclusion is the whole
    subtlety of this function. Its name promises a defence, and ``is_defence`` is
    true for it, but the engine reports ``can_attack: false`` with ``is_laser: true``
    (measured, ``.runtime/q2_laser_defence.json``): it is a projectile **interceptor**
    that destroys incoming missiles and cannot shoot at anything. Building one where a
    gun turret was intended therefore costs 500 credits and produces no firepower.

    Measured consequence of getting this wrong: the curriculum picked the *first*
    defensible name in the engine's menu (``laserDefence`` sorts before ``turret``),
    spent 50 orders per episode on it, and the episode ended with no gun turret --
    which looks exactly like "the agent cannot build defences" from the outside
    (``.runtime/defence_direct.json``).

    ``repairbay`` is excluded for a different reason: it repairs instead of shooting.
    """

    lowered = str(name or "").lower()
    if "repair" in lowered or "laser" in lowered:
        return False
    return any(
        marker in lowered
        for marker in ("turret", "antiair", "aa_", "_aa")
    )


__all__ = [
    "POOL_ONLY_MARKERS",
    "build_menu",
    "find_structure",
    "is_defence_type",
    "is_pool_only_type",
    "menu_index",
    "produce_menu",
    "produce_options",
    "upgradable_buildings",
]


#: Substrings that identify a structure the engine only accepts on a resource
#: pool.
#:
#: ``Unit.placeOnlyOnResPool()`` is a *type-level* property with no per-instance
#: answer (the instance method is hard-coded ``false`` and no stock subclass
#: overrides it -- the truth is ``UnitType$1.placeOnlyOnResPool()``), and the
#: bridge does not report it per entity. An extractor is the only stock type in
#: this category, and the protocol already has a dedicated kind for it
#: (``build_extractor``, which resolves a real pool coordinate).
#:
#: So ``build_structure`` must never offer these: it carries an arbitrary
#: ``x``/``y`` from the position head, and dispatching an extractor there makes
#: the engine queue a placement that cannot legally finish. Measured in
#: ``.runtime/buildnames.json``: ``structure_type="extractor"`` at (1170, 2050)
#: -- a spot with no pool -- came back ``accepted: true``. A receipt saying
#: "accepted" for a structure that can never be built is worse than a rejection,
#: because it pays ``bonus_build``-shaped exploration value for nothing.
POOL_ONLY_MARKERS: tuple[str, ...] = ("extractor",)


def is_pool_only_type(name: str) -> bool:
    """Whether this unit type must be placed on a resource pool."""

    lowered = str(name or "").strip().lower()
    return any(marker in lowered for marker in POOL_ONLY_MARKERS)


def _names_from(entries: Any, key: str, *, skip: tuple[str, ...] = ()) -> list[str]:
    """First-seen-order union of ``entries[][key]``, minus ``skip`` names."""

    if not isinstance(entries, list):
        return []
    skipped = {name.lower() for name in skip}
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        values = entry.get(key)
        if not isinstance(values, list):
            continue
        for raw in values:
            name = str(raw or "").strip()
            if not name:
                continue
            if name.lower() in skipped:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def build_menu(state: dict[str, Any]) -> list[str]:
    """Structure types the policy may order through ``build_structure``.

    Empty when the payload reports no builder capabilities at all (an older
    bridge, the offline mock engine, or a hand-written test fixture). Callers
    must treat empty as "no information" and fall back to their historical
    behaviour rather than as "nothing can be built" -- masking every structure
    kind because the bridge stayed silent would silently delete the whole
    economy on those environments.
    """

    return [
        name
        for name in _names_from(state.get("builders"), "can_build")
        if not is_pool_only_type(name)
    ]


def produce_menu(state: dict[str, Any]) -> list[str]:
    """Unit types the policy may order through ``produce_unit``.

    ``builder`` is excluded, and that exclusion is the whole point: every factory
    in the game can make builders, they are the cheapest menu entry, and they are
    not combat units. Leaving them in made the decoded order deterministically
    pick them and the episode end with 12 builders and no army (documented in
    :mod:`ironzero.observation.masks`).

    Delivered in the engine's own order, so "index 0" is a *stable* choice and
    not an accident of dictionary iteration.
    """

    return _names_from(state.get("factories"), "can_produce", skip=("builder",))


def produce_options(state: dict[str, Any]) -> list[tuple[int, str]]:
    """``[(factory_entity_id, unit_type), ...]`` the policy may order.

    The **cross product** of "which factory" and "which unit", in a stable order,
    because a single product is not enough to express the decision the game
    actually offers. Measured on ``[p2]Lake (2p)``: the map's ``landFactory``
    offers exactly two combat types, so a decode that always takes the first one
    can only ever field one of them -- which is precisely the reported "only
    builds one kind of tank".

    ``produce_unit``'s target head indexes this list, so:
      * one factory with three units gives three distinguishable choices;
      * two factories with two units each gives four, and *which factory* is part
        of the decision (a forward factory reinforces faster).

    Factories whose menu is empty or unreported are skipped: an unreported menu
    is handled by :func:`produce_menu`'s "no information" rule, not here, and a
    reported-but-empty menu is the engine saying "queue nothing here right now".
    """

    factories = state.get("factories")
    if not isinstance(factories, list):
        return []
    out: list[tuple[int, str]] = []
    for entry in factories:
        if not isinstance(entry, dict):
            continue
        values = entry.get("can_produce")
        if not isinstance(values, list):
            continue
        try:
            factory_id = int(entry.get("entity_id", -1))
        except (TypeError, ValueError):
            factory_id = -1
        for raw in values:
            name = str(raw or "").strip()
            if not name or name.lower() == "builder":
                continue
            out.append((factory_id, name))
    return out


def upgradable_buildings(state: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """``(buildings_the_engine_says_can_upgrade_now, payload_reported_the_field)``.

    The second element is what separates "the engine was asked and said no" from
    "this bridge cannot answer". They look identical in the list but must not be
    treated identically: a reported-and-empty list means the kind is genuinely
    unavailable right now (so masking it is honest), while an unreported field
    means an older bridge where the historical fallback must stay in place.
    """

    own = state.get("own_entities")
    if not isinstance(own, list):
        return [], False
    reported = any(isinstance(unit, dict) and "can_upgrade" in unit for unit in own)
    upgradable = [
        unit
        for unit in own
        if isinstance(unit, dict) and unit.get("can_upgrade")
    ]
    return upgradable, reported
