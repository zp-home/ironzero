# SPDX-License-Identifier: Apache-2.0
"""Bridge observation -> encoder state, and policy action -> ``ironzero/1`` steps.

Pure conversion helpers: no socket, no process, no reward bookkeeping. Keeping
them side-effect free is what lets the tests exercise the protocol contract
without a game runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..micro_config import MicroDuelConfig
from ..mock_engine import ACTION_KINDS


def observation_to_state(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one bridge ``obs`` message into the encoder's raw state dict.

    The real bridge only reports positions/hp for alive units; group and order
    metadata are optional and default to group 0 / idle in the encoder.

    ``resource_slots`` must be carried through: both the encoder (how many
    extractors are placed) and :func:`decode_action` (which pool a
    ``build_extractor`` order targets) read it. Dropping it here made every
    extractor order resolve to slot 0 and told the policy that no extractor was
    ever built, so builders were free to wander off to the enemy base instead of
    expanding the economy.
    """

    enemies = [dict(entity) for entity in message.get("visible_enemy_entities", [])]
    info = message.get("info")
    slots = message.get("resource_slots")
    state = {
        "tick": int(message.get("tick", 0)),
        "game_time": float(message.get("game_time", 0.0)),
        "controlled_team": int(message.get("controlled_team", 0)),
        "global_state": dict(message.get("global_state") or {}),
        "own_entities": [dict(entity) for entity in message.get("own_entities", [])],
        "visible_enemy_entities": enemies,
        "target_ids": [int(entity.get("entity_id", -1)) for entity in enemies],
        # Resource pools: ``slot_id``/``x``/``y``/``occupied``/``extractor_level``.
        # ``build_extractor`` resolves ``slot_id`` against this list, and the
        # encoder counts occupied slots from it. Absent (older bridge) -> empty.
        "resource_slots": [dict(slot) for slot in slots] if isinstance(slots, list) else [],
        # ``info.builders`` is the engine's own answer to "which economy unit
        # types can our builders queue". The action mask reads it so a kind is
        # only selectable when the game would actually accept it; without this
        # the policy keeps picking extractor orders that the map cannot honour.
        "builders": list((info or {}).get("builders") or []) if isinstance(info, dict) else [],
        # ``info.factories`` is the engine's own answer to "which unit types can
        # our factories produce". ``produce_unit`` is a *factory* action, so the
        # mask gates it on this rather than on the builder's build menu (they are
        # different capabilities: on a stock map the builder's menu is just
        # ``['landFactory']``, which contains no ``tank``). Without it the kind
        # was permanently masked and the agent could never field a combat unit.
        "factories": list((info or {}).get("factories") or []) if isinstance(info, dict) else [],
        # Behaviour-cloning labels: the expert's current order per own unit
        # (``--ai-pilot --record-ai-orders``). Kept verbatim -- this is the target
        # the supervised phase trains against, and ``None`` when the bridge is not
        # in teacher mode so normal runs carry no extra payload.
        "info": {
            "ai_orders": list((info or {}).get("ai_orders") or []),
            "ai_pilot": bool((info or {}).get("ai_pilot", False)),
            # Resolved teacher strength (-2..3), None for a human-driven team. Kept
            # so a BC dataset can record which difficulty produced the labels.
            "ai_difficulty": (info or {}).get("ai_difficulty"),
            # Per-team resolved difficulty + income multiplier. This is the value
            # that actually drives the AI's resource cheat, so it is recorded rather
            # than inferred from funds.
            "ai_teams": list((info or {}).get("ai_teams") or []),
        } if isinstance(info, dict) else {},
        "receipts": [],
        "result": message.get("result"),
        "terminated": bool(message.get("terminated", False)),
        "truncated": bool(message.get("truncated", False)),
    }
    return state


def _builder_group(state: dict[str, Any]) -> int:
    """The ``unit_group`` index a build order must use to reach a builder.

    Build orders are addressed to a *unit group*, and ``IronZeroBridge`` selects
    group members with ``unit.id % GROUP_COUNT == unit_group`` -- so a builder
    belongs to exactly one numbered group for the whole episode. The policy
    meanwhile picks ``group`` freely (it is a learned head), and nothing ties its
    choice to where the builder actually is.

    Measured on ``[p2]Lake (2p)`` (``.runtime/econ_gate.json``): the map-placed
    builder is unit 24 (group 0) and the command center is unit 23 (group 3).
    A ``build_extractor`` addressed to group 3 is rejected ``BUILD_UNAVAILABLE``
    while the *same* order to group 0 or to the broadcast slot (4) is accepted.
    The action mask opens every numbered group that owns any unit, so group 3 was
    selectable solely because the command center lives there -- an address that
    can never build. In the 200-update Lake run this produced
    ``BUILD_UNAVAILABLE=1620``, the single largest rejection bucket.

    Returning the broadcast slot (``cfg.groups``) is the robust choice: the
    bridge treats ``unit_group == GROUP_COUNT`` as "every living friendly unit",
    which includes the builder wherever it sits and stays correct when the map
    places *several* builders or when a factory produces more of them. Group
    membership of the builder is not part of what the policy is being asked to
    learn here -- only *whether* and *what* to build -- so this does not remove a
    real decision from the action space.
    """

    from ..micro_config import DEFAULT_CONFIG

    return int(DEFAULT_CONFIG.groups)


def decode_action(
    action: np.ndarray,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode one policy action into ``ironzero/1`` step actions.

    The bridge understands the same group head as the mock engine: 0..G-1 order
    the units whose id maps to that group, and the last slot (G) broadcasts to
    every living friendly unit.
    """

    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    expected_heads = len(config.action_nvec)
    if vector.shape[0] != expected_heads:
        raise ValueError(
            f"action must have {expected_heads} heads, got {vector.shape[0]}"
        )
    return _decode_heads(vector, state, config)


def _decode_heads(
    vector: np.ndarray,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode exactly one 5-head slot (``kind, group, x, y, target``).

    Split out of :func:`decode_action` so the two-slot decoder
    (:func:`decode_action_multi`) can reuse the *identical* per-kind translation
    instead of maintaining a second copy of it. It deliberately does **not**
    validate the vector length against ``config.action_nvec``: it is called with
    one slot of a possibly doubled head layout.
    """

    kind = ACTION_KINDS[int(vector[0]) % len(ACTION_KINDS)]
    group = int(vector[1])
    gx = int(np.clip(vector[2], 0, config.position_grid - 1))
    gy = int(np.clip(vector[3], 0, config.position_grid - 1))
    target_index = int(vector[4])
    # Enemies keep the bridge's stable stream order: the encoder writes them in
    # that order, so target slot i names the i-th streamed enemy and the mapping
    # does not shuffle between frames.
    enemies = state.get("visible_enemy_entities", [])
    own = state.get("own_entities", [])

    if kind in {"move_group", "attack_move"}:
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": kind, "unit_group": group, "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "build_extractor":
        # ``slot_id`` indexes the resource slots reported by the bridge. Only
        # slots without an extractor are candidates: re-ordering an existing pool
        # is a no-op the engine accepts, which would let the policy "succeed"
        # forever without ever growing the economy. Among the free slots the
        # bridge already sorts nearest-to-own-base first, so the counter keeps
        # that preference. When the bridge reports no slots at all we still emit
        # the action and let the Java side resolve slot 0 against its own map
        # resource list.
        slots = state.get("resource_slots") or []
        free = [
            index
            for index, slot in enumerate(slots)
            if not (isinstance(slot, dict) and slot.get("occupied"))
        ]
        if free:
            # The target head is shared with the enemy head, so a slot index is
            # taken modulo the number of free pools rather than trusted raw.
            candidates = free
        elif slots:
            # Every pool is taken: fall back to all of them so the order is still
            # well-formed (the engine rejects it as already occupied).
            candidates = list(range(len(slots)))
        else:
            candidates = [0]
        return [{"kind": "build_extractor", "unit_group": _builder_group(state),
                 "slot_id": int(candidates[int(target_index) % len(candidates)])}]
    if kind == "build_structure":
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": "build_structure", "unit_group": _builder_group(state),
                 "structure_type": "land_factory",
                 "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "produce_unit":
        # Production is a *factory* action: pick the factory, then ask the game
        # which unit types it can actually queue here. ``"tank"`` used to be
        # hard-coded, which silently asked every factory for a tank -- wrong on
        # any map whose factory cannot make one (a Lake start's commandCenter can
        # only make a scout), and the order then either got rejected or produced
        # something the policy never chose. Falls back to ``"tank"`` only when the
        # bridge reports no factory capabilities at all.
        factories = state.get("factories") or []
        reported = [
            entry
            for entry in factories
            if isinstance(entry, dict) and isinstance(entry.get("can_produce"), list)
        ]
        if reported:
            # Prefer a factory that offers a *combat* unit. The list is in the
            # engine's unit order, which puts the command center first, and a
            # command center only makes builders -- so taking ``target_index``
            # raw would spend the whole economy on builders even with a real
            # factory standing.
            combat_factories = [
                entry
                for entry in reported
                if any(
                    str(name).lower() != "builder"
                    for name in (entry.get("can_produce") or [])
                )
            ]
            pool = combat_factories or reported
            factory = pool[int(target_index) % len(pool)]
            producible = [str(name) for name in factory.get("can_produce") or []]
            if not producible:
                return _hold_action(state, group)
            # The group head is not a unit-type head, so one deterministic pick
            # is made per factory. Prefer a *combat* unit: a factory's menu is
            # reported in engine registry order and can put ``builder`` (or
            # ``hoverTank``) first, and always picking the first entry would
            # spend the whole economy on builders. Falling back to the first
            # entry keeps every menu usable, including one that only offers
            # builders (a Lake command center).
            combat = [name for name in producible if name.lower() != "builder"]
            unit_type = combat[0] if combat else producible[0]
            return [
                {
                    "kind": "produce_unit",
                    "unit_group": group,
                    "factory_entity_id": int(factory.get("entity_id", -1)),
                    "unit_type": unit_type,
                }
            ]
        factories = [u for u in own if u.get("is_factory") or u.get("is_factory_building")]
        if not factories:
            return _hold_action(state, group)
        factory = factories[int(target_index) % len(factories)]
        return [{"kind": "produce_unit", "unit_group": group,
                 "factory_entity_id": int(factory.get("entity_id", -1)),
                 "unit_type": "tank"}]
    if kind == "upgrade_building":
        # Select the target by what the bridge says the engine would accept, not
        # by a hard-coded type id. The old test was
        # ``entity_type_id == EXTRACTOR_TYPE_ID (10)``, but the bridge used to
        # report every building as ``entity_type_id = 2``, so this branch never
        # found a building and degraded every upgrade order into ``hold`` -- the
        # extractor tier upgrade, the core of the Rusted Warfare economy, simply
        # did not exist on the real game.
        #
        # ``can_upgrade`` is the engine's own answer ("this building has a next
        # tier available right now", from ``getBuildActionId``). Bridges that do
        # not report it fall back to any own building, which at worst yields a
        # rejected order instead of a silently dropped one.
        upgradable = [u for u in own if u.get("can_upgrade")]
        if not upgradable:
            upgradable = [u for u in own if u.get("is_building")]
        if not upgradable:
            return _hold_action(state, group)
        building = upgradable[int(target_index) % len(upgradable)]
        return [{"kind": "upgrade_building", "unit_group": group,
                 "building_entity_id": int(building.get("entity_id", -1))}]
    if kind == "attack_target":
        if 0 <= target_index < len(enemies):
            return [
                {
                    "kind": kind,
                    "unit_group": group,
                    "target_entity_id": int(enemies[target_index].get("entity_id", -1)),
                }
            ]
        return _hold_action(state, group)
    if kind == "noop":
        # Issue nothing at all. Construction and production need hundreds of
        # uninterrupted ticks, and every order cancels the in-flight one, so the
        # policy must be able to skip a decision without stopping its builders.
        return []
    if kind in {"stop", "hold"}:
        return _hold_action(state, group)
    return [{"kind": kind, "unit_group": group}]


def _hold_action(state: dict[str, Any], group: int = 4) -> list[dict[str, Any]]:
    """Stop/hold order; the bridge keeps each unit at its current position."""

    return [{"kind": "hold", "unit_group": int(group)}]


# ----------------------------------------------------------------------
# Two-slot ("economic + tactical") decoding
# ----------------------------------------------------------------------
# Why this exists
# ---------------
# One decision used to produce exactly one order, chosen from a single flat kind
# head. The builder is a *dual-purpose* unit -- an economy unit (it queues
# extractors / factories) and a mobile unit (it belongs to a ``unit.id % 4``
# group like everything else). A single order therefore cannot say "keep
# building AND do not walk anywhere": picking ``build_*`` implicitly stops it
# from moving, while ``hold``/``stop`` cancels the in-flight construction,
# because *every* real order cancels the previous one (that is why ``noop``
# exists in ``ACTION_KINDS``).
#
# Measured on ``[p2]Lake (2p)``: the map-placed builder is unit 24 (group 0) and
# the command center is unit 23 (group 3), so while the builder was the only
# mover, the tactical options and the economic options competed for the same
# single slot. The policy had to alternate -- build, cancel, hold, build -- and
# a structure needs ~140 uninterrupted ticks, so nothing finished; the observed
# symptom was "the builder finishes the extractor, then charges the enemy base
# and dies".
#
# The fix is a *second slot in the same decision*: an economy order and a
# tactical order are decoded independently, and an empty tactical slot emits
# nothing at all. Two orders per decision is exactly what the bridge already
# supports (``applyStep`` iterates the whole ``actions`` array, dispatches once
# and emits one receipt per entry -- verified, no Java change needed).

#: Kinds the **economy** slot may select, in an order that is a **contract**: the
#: policy's economy head indexes this tuple, so appending is safe but reordering
#: or removing silently re-maps every trained weight.
#:
#: The trailing ``noop`` is the "no economic order this decision" entry (see
#: :data:`ECONOMY_EMPTY_INDEX`). It is not decoration: an agent that has just
#: queued an extractor must be able to spend the next ~140 decisions *not
#: re-issuing the order*. Re-issuing is not harmless -- it re-queues, and the
#: measured "accepted = true every single decision" pattern in
#: ``reward_shaping.compute_step_shaping`` is exactly what that looks like.
ECONOMY_ACTION_KINDS: tuple[str, ...] = (
    "build_extractor",
    "build_structure",
    "produce_unit",
    "upgrade_building",
    "noop",
)
#: Index of the economy slot's "do nothing" entry.
ECONOMY_EMPTY_INDEX = len(ECONOMY_ACTION_KINDS) - 1

#: Kinds the **tactical** slot may select, ending with its own "no tactical
#: order" entry. This empty entry is the *load-bearing* part of the whole design:
#: it emits no order at all, so the builder's in-flight construction is not
#: cancelled while the policy is thinking about the army.
TACTICAL_ACTION_KINDS: tuple[str, ...] = (
    "move_group",
    "attack_move",
    "attack_target",
    "hold",
    "stop",
    "noop",
)
#: Index of the tactical slot's "no tactical order" entry.
TACTICAL_EMPTY_INDEX = len(TACTICAL_ACTION_KINDS) - 1

#: Number of action heads per slot (``kind, group, x, y, target``).
SLOT_HEADS = 5


def multi_action_nvec(config: MicroDuelConfig) -> tuple[int, ...]:
    """Per-head cardinalities of the two-slot action layout.

    Layout (10 heads, economy first)::

        economy:  [len(ECONOMY_ACTION_KINDS), groups+1, P, P, target_slots]
        tactical: [len(TACTICAL_ACTION_KINDS), groups+1, P, P, target_slots]

    The group / position / target widths are the **same numbers** the
    single-slot ``action_nvec`` uses, so "target slot 3" and "group 4" mean the
    same thing in both layouts and a mask or a checkpoint explaining one layout
    still explains the other. Only the two ``kind`` heads differ, because each
    slot is restricted to its own vocabulary.

    This is a function rather than a ``MicroDuelConfig.action_nvec`` variant on
    purpose: ``action_nvec`` is read by PPO, the env spaces, the mask builder and
    the checkpoint loader, all of which assume one slot. Flipping its meaning
    behind a config flag would change what an existing checkpoint means without
    touching any of those call sites. The Lead merges the flag (see the report's
    "micro_config merge checklist"); until then this stays an explicit call.

    Example, default micro-duel config::

        >>> multi_action_nvec(DEFAULT_CONFIG)
        (5, 5, 8, 8, 9, 6, 5, 8, 8, 9)
    """

    group = int(config.groups) + 1
    grid = int(config.position_grid)
    target = int(config.target_slots)
    economy = (len(ECONOMY_ACTION_KINDS), group, grid, grid, target)
    tactical = (len(TACTICAL_ACTION_KINDS), group, grid, grid, target)
    return economy + tactical


def decode_action_multi(
    action: np.ndarray,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode one two-slot decision into **at most two** ``ironzero/1`` orders.

    The first ``SLOT_HEADS`` entries are the economy slot, the last
    ``SLOT_HEADS`` are the tactical slot. Each slot is decoded by the *same*
    per-kind translation :func:`decode_action` uses (``_decode_heads``), so the
    two paths cannot drift.

    The contract that makes this useful:

    * an economy slot outside :data:`ECONOMY_ACTION_KINDS` decodes as **nothing**
      (not a fallback ``hold`` -- a fallback would cancel the very construction
      the slot exists to protect);
    * a tactical slot whose kind is one of the empty entries decodes as
      **nothing**;
    * therefore ``decode_action_multi`` returns ``[]``, ``[econ]``, ``[tac]`` or
      ``[econ, tac]`` and never a synthesised order the policy did not choose.

    Empty economy + empty tactical is the historical ``noop``: the bridge
    receives an empty ``actions`` array, dispatches no command and reports one
    acceptance (see ``applyStep``), so construction keeps running.

    Order matters: the economy order is emitted first. The bridge dispatches
    every queued command once, at the end of ``applyStep``, and the two slots
    address different units -- the economy slot broadcasts to whichever builder
    can queue the build, the tactical slot addresses one numbered group. When
    they *do* overlap (a tactical order aimed at the builder's own group) the
    later command wins the unit, which is the same "last order wins" rule a
    human player sees. Emitting the economy order first makes the policy's
    explicit tactical choice the one that sticks, and leaves the
    no-tactical-order case exactly as it was.
    """

    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    expected = 2 * SLOT_HEADS
    if vector.shape[0] != expected:
        raise ValueError(
            f"multi-action vector must have {expected} heads "
            f"(2 slots x {SLOT_HEADS}), got {vector.shape[0]}"
        )

    economy_vector = vector[:SLOT_HEADS]
    tactical_vector = vector[SLOT_HEADS:]

    orders: list[dict[str, Any]] = []

    economy_kind = _slot_kind(
        economy_vector[0], ECONOMY_ACTION_KINDS, ECONOMY_EMPTY_INDEX
    )
    if economy_kind is not None:
        orders.extend(_decode_slot(economy_vector, economy_kind, state, config))

    tactical_kind = _slot_kind(
        tactical_vector[0], TACTICAL_ACTION_KINDS, TACTICAL_EMPTY_INDEX
    )
    if tactical_kind is not None:
        orders.extend(_decode_slot(tactical_vector, tactical_kind, state, config))

    return orders


def _slot_kind(index: int, vocabulary: tuple[str, ...], empty_index: int) -> str | None:
    """Map a slot's kind index onto a kind name, or ``None`` for "do nothing".

    ``empty_index`` is the caller's explicit "this slot issues nothing" entry
    (``ECONOMY_EMPTY_INDEX`` / ``TACTICAL_EMPTY_INDEX``), passed in rather than
    guessed so a future vocabulary edit cannot silently turn a real order into a
    no-op or vice versa.

    The index is folded with ``%``: the policy heads are ``Categorical`` over
    exactly ``len(vocabulary)`` logits so a sampled index is always in range, but
    hand-built test arrays and older checkpoints can carry anything.
    """

    position = int(index) % len(vocabulary)
    if position == empty_index:
        return None
    return vocabulary[position]


def _decode_slot(
    vector: np.ndarray,
    kind: str,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode one slot with ``kind`` already resolved, reusing ``_decode_heads``.

    ``_decode_heads`` reads the kind from vector[0], so the slot's own kind name
    is substituted back into the head vector. That keeps a single translation
    table for *both* decoders: a new build/produce rule added to ``_decode_heads``
    automatically applies to the two-slot layout, and the two cannot disagree
    about e.g. which resource pool a ``build_extractor`` names.
    """

    head = np.array(vector, dtype=np.int64).reshape(-1).copy()
    head[0] = ACTION_KINDS.index(kind)
    return _decode_heads(head, state, config)


def merge_done_message(state: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    """Fold a ``done`` message into the previous state (terminal-observation path).

    A ``done`` message ends the episode. Some bridge builds attach the final
    entities/tick to it, others only the result: merge whatever is present so the
    returned observation/masks describe the real terminal state instead of the
    previous decision's state.
    """

    merged = dict(state)
    if message.get("tick") is not None:
        try:
            merged["tick"] = int(message["tick"])
        except (TypeError, ValueError):
            pass
    if message.get("game_time") is not None:
        merged["game_time"] = float(message["game_time"])
    if message.get("result") is not None:
        merged["result"] = message["result"]
    if message.get("terminated") is not None:
        merged["terminated"] = bool(message["terminated"])
    if message.get("truncated") is not None:
        merged["truncated"] = bool(message["truncated"])
    for key in ("own_entities", "visible_enemy_entities"):
        if isinstance(message.get(key), list):
            merged[key] = [dict(entity) for entity in message[key]]
    # Resource pools belong to the terminal state too; the encoder counts them.
    if isinstance(message.get("resource_slots"), list):
        merged["resource_slots"] = [dict(slot) for slot in message["resource_slots"]]
    if isinstance(message.get("own_entities"), list) or isinstance(
        message.get("visible_enemy_entities"), list
    ):
        merged["target_ids"] = [
            int(entity.get("entity_id", -1))
            for entity in merged.get("visible_enemy_entities", [])
        ]
    return merged


__all__ = [
    "ECONOMY_ACTION_KINDS",
    "ECONOMY_EMPTY_INDEX",
    "SLOT_HEADS",
    "TACTICAL_ACTION_KINDS",
    "TACTICAL_EMPTY_INDEX",
    "_hold_action",
    "decode_action",
    "decode_action_multi",
    "merge_done_message",
    "multi_action_nvec",
    "observation_to_state",
]
