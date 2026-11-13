# SPDX-License-Identifier: Apache-2.0
"""Two-slot ("economy + tactical") action decoding must not interrupt building.

The bug this guards
-------------------
A decision used to produce exactly **one** order. The builder is a dual-purpose
unit -- it queues extractors/factories *and* it belongs to a ``unit.id % 4``
group like any other mobile unit -- so a single order could never express "keep
building and do not walk anywhere":

* ``build_*`` implicitly stops the builder from doing anything else;
* ``hold`` / ``stop`` **cancels** the in-flight construction, because every real
  order cancels the previous one;
* so the policy had to alternate build -> cancel -> hold -> build, and a
  structure needs ~140 uninterrupted ticks. Measured symptom on
  ``[p2]Lake (2p)``: the builder finished the extractor, then charged the enemy
  base and died.

:func:`ironzero.rwpp.decoding.decode_action_multi` splits one decision into an
economy slot and a tactical slot. The invariant that fixes the bug is *negative*:
an empty tactical slot must emit **no order at all**, so the construction in
flight is never cancelled. That is the assertion this file spends most of its
length on.

Backward compatibility is the other half of the contract: ``multi_action`` is off
by default, so ``action_nvec`` must stay bit-identical to the historical
``(10, 5, 8, 8, 9)`` and :func:`decode_action` must keep its exact behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ironzero.mock_engine import ACTION_KINDS
from ironzero.rwpp.decoding import (
    ECONOMY_ACTION_KINDS,
    ECONOMY_EMPTY_INDEX,
    SLOT_HEADS,
    TACTICAL_ACTION_KINDS,
    TACTICAL_EMPTY_INDEX,
    decode_action,
    decode_action_multi,
    multi_action_nvec,
)

ECON = {kind: index for index, kind in enumerate(ECONOMY_ACTION_KINDS)}
TAC = {kind: index for index, kind in enumerate(TACTICAL_ACTION_KINDS)}
KIND = {kind: index for index, kind in enumerate(ACTION_KINDS)}

#: Orders that cancel an in-flight construction -- i.e. everything that is not
#: "issue nothing". Any of these appearing in a "tactical slot is empty" decode
#: is the original bug coming back.
BUILD_INTERRUPTING = frozenset(
    {"move_group", "attack_move", "attack_target", "hold", "stop"}
)


def _lake_state(**overrides) -> dict:
    """A ``[p2]Lake (2p)`` start, trimmed to what the decoder reads.

    Real numbers, taken from ``.runtime/econ_gate.json``: the command center is
    unit 23 (group 3) and the map-placed builder is unit 24 (group 0). The
    builder is deliberately *not* in the command center's group -- that is what
    made "address the builder" and "address the army" collide in one slot.
    """

    state = {
        "own_entities": [
            {
                "entity_id": 23, "entity_type_id": 11, "unit_type": "commandCenter",
                "hp": 4000.0, "max_hp": 4000.0, "x": 400.0, "y": 2200.0,
                "is_building": True, "is_factory": True, "group_id": 3,
            },
            {
                "entity_id": 24, "entity_type_id": 2, "unit_type": "builder",
                "hp": 170.0, "max_hp": 170.0, "x": 430.0, "y": 2180.0,
                "is_builder": True, "group_id": 0,
            },
        ],
        "visible_enemy_entities": [
            {"entity_id": 91, "hp": 210.0, "max_hp": 210.0, "x": 2200.0, "y": 400.0},
        ],
        "resource_slots": [
            {"slot_id": 0, "x": 600.0, "y": 2100.0, "occupied": False},
            {"slot_id": 1, "x": 700.0, "y": 2000.0, "occupied": False},
        ],
        "global_state": {"funds": 4000.0},
        "builders": [
            {
                "entity_id": 24, "unit_type": "builder",
                "can_build": ["landFactory", "extractor"],
            }
        ],
        "factories": [
            {"entity_id": 23, "unit_type": "commandCenter", "can_produce": ["builder"]}
        ],
    }
    state.update(overrides)
    return state


def _slot(kind_index: int, group: int = 4, gx: int = 3, gy: int = 3,
          target: int = 0) -> list[int]:
    return [kind_index, group, gx, gy, target]


def _multi(economy: list[int], tactical: list[int]) -> np.ndarray:
    return np.array(economy + tactical, dtype=np.int64)


# ----------------------------------------------------------------------
# 1. Backward compatibility: default layout is byte-identical
# ----------------------------------------------------------------------
def test_default_config_keeps_the_historical_action_nvec() -> None:
    """``multi_action`` defaults to off, so the head layout cannot have moved.

    This is the hard requirement of the task: every existing test, script and
    checkpoint assumes ``(10, 5, 8, 8, 9)``. ``10`` is ``len(ACTION_KINDS)``,
    ``5`` is ``groups + 1``, ``8`` is ``position_grid`` and ``9`` is
    ``target_slots``.
    """

    assert DEFAULT_CONFIG.action_nvec == (10, 5, 8, 8, 9)
    assert DEFAULT_CONFIG.action_nvec == (
        len(ACTION_KINDS),
        DEFAULT_CONFIG.groups + 1,
        DEFAULT_CONFIG.position_grid,
        DEFAULT_CONFIG.position_grid,
        DEFAULT_CONFIG.target_slots,
    )


def test_there_is_no_multi_action_flag_on_the_config_yet() -> None:
    """The flag is the Lead's merge (see the report's micro_config checklist).

    Asserting its *absence today* keeps this task honest: the two-slot layout is
    opt-in by calling ``decode_action_multi`` explicitly, not by a config field
    that would silently change ``action_nvec`` for every parallel engineer. When
    the Lead merges the flag this test is the one to replace with a
    ``multi_action=True`` variant -- its whole purpose is to make that change
    deliberate rather than incidental.
    """

    assert not hasattr(DEFAULT_CONFIG, "multi_action")


def test_single_slot_decode_action_is_unchanged() -> None:
    """``decode_action`` keeps its signature and its output for every kind."""

    state = _lake_state()
    expected = {
        "move_group": 1,
        "attack_move": 1,
        "attack_target": 1,
        "hold": 1,
        "stop": 1,
        "noop": 0,
        "build_extractor": 1,
        "build_structure": 1,
        "produce_unit": 1,
        "upgrade_building": 1,
    }
    for kind, count in expected.items():
        action = np.array([KIND[kind], 4, 3, 3, 0], dtype=np.int64)
        orders = decode_action(action, state, DEFAULT_CONFIG)
        assert len(orders) == count, f"{kind} -> {orders}"
        # Never more than one order: that is the limitation being lifted, and
        # the old entry point must keep it so old callers stay bit-identical.
        assert len(orders) <= 1


# ----------------------------------------------------------------------
# 2. New layout shape
# ----------------------------------------------------------------------
def test_multi_layout_doubles_the_heads_and_keeps_the_shared_widths() -> None:
    """Two slots of 5 heads, sharing group/position/target widths."""

    nvec = multi_action_nvec(DEFAULT_CONFIG)
    assert len(nvec) == 2 * SLOT_HEADS == 10
    economy, tactical = nvec[:SLOT_HEADS], nvec[SLOT_HEADS:]
    assert economy == (len(ECONOMY_ACTION_KINDS), 5, 8, 8, 9)
    assert tactical == (len(TACTICAL_ACTION_KINDS), 5, 8, 8, 9)
    # The three non-kind widths must equal the single-slot ones exactly, or the
    # same mask could not describe both layouts.
    assert nvec[1:5] == DEFAULT_CONFIG.action_nvec[1:]
    assert nvec[6:10] == DEFAULT_CONFIG.action_nvec[1:]


def test_multi_layout_tracks_a_bigger_map_contract() -> None:
    """A 25v25 map widens target slots in both slots, not just the first."""

    big = MicroDuelConfig(max_entities=25, target_slots=26)
    nvec = multi_action_nvec(big)
    assert nvec[4] == 26 and nvec[9] == 26
    assert nvec[1] == nvec[6] == big.groups + 1


def test_slot_vocabularies_are_restricted_to_their_own_semantics() -> None:
    """The economy slot cannot move units; the tactical slot cannot build.

    That separation is what turns "keep building" into an *expressible* choice
    instead of a trade-off: neither head can spend its slot on the other's job.
    """

    assert set(ECONOMY_ACTION_KINDS) == {
        "build_extractor", "build_structure", "produce_unit",
        "upgrade_building", "noop",
    }
    assert set(TACTICAL_ACTION_KINDS) == {
        "move_group", "attack_move", "attack_target", "hold", "stop", "noop",
    }
    # Every vocabulary entry must be a kind the bridge actually dispatches.
    assert set(ECONOMY_ACTION_KINDS) <= set(ACTION_KINDS)
    assert set(TACTICAL_ACTION_KINDS) <= set(ACTION_KINDS)


# ----------------------------------------------------------------------
# 3. The point of the whole exercise: two orders in one decision
# ----------------------------------------------------------------------
def test_build_and_tactical_move_are_emitted_together() -> None:
    """One decision, two independent orders: build + move the army."""

    state = _lake_state()
    orders = decode_action_multi(
        _multi(_slot(ECON["build_structure"]), _slot(TAC["move_group"], group=1)),
        state,
        DEFAULT_CONFIG,
    )
    assert len(orders) == 2
    assert [order["kind"] for order in orders] == ["build_structure", "move_group"]
    # The build order names a real spot and reaches a builder via the broadcast
    # group, exactly as the single-slot decoder does.
    build, move = orders
    assert build["unit_group"] == DEFAULT_CONFIG.groups
    assert build["structure_type"] == "land_factory"
    assert move["unit_group"] == 1
    # The two orders must address different concerns; a shared group is allowed
    # (a player may order the same units twice) but the *kinds* must differ.
    assert build["kind"] != move["kind"]


def test_extractor_slot_plus_attack_move() -> None:
    """The economy slot reads the target head as a pool index, as before."""

    state = _lake_state()
    orders = decode_action_multi(
        _multi(
            _slot(ECON["build_extractor"], target=1),
            _slot(TAC["attack_move"], group=2, gx=6, gy=7),
        ),
        state,
        DEFAULT_CONFIG,
    )
    kinds = [order["kind"] for order in orders]
    assert kinds == ["build_extractor", "attack_move"]
    # Slot 1 is the second *free* pool; the decoder keeps its "never re-order an
    # existing extractor" rule in the two-slot path too.
    assert orders[0]["slot_id"] == 1
    assert orders[0]["unit_group"] == DEFAULT_CONFIG.groups


def test_economy_only_decision_is_allowed() -> None:
    """Economy order with an empty tactical slot: exactly one order."""

    state = _lake_state()
    orders = decode_action_multi(
        _multi(
            _slot(ECON["produce_unit"]),
            _slot(TACTICAL_EMPTY_INDEX, group=0),
        ),
        state,
        DEFAULT_CONFIG,
    )
    assert [order["kind"] for order in orders] == ["produce_unit"]


def test_tactical_only_decision_is_allowed() -> None:
    """Empty economy slot plus a real tactical order: exactly one order."""

    state = _lake_state()
    orders = decode_action_multi(
        _multi(_slot(ECONOMY_EMPTY_INDEX), _slot(TAC["attack_target"], target=0)),
        state,
        DEFAULT_CONFIG,
    )
    assert [order["kind"] for order in orders] == ["attack_target"]
    assert orders[0]["target_entity_id"] == 91


def test_both_slots_empty_issues_nothing_at_all() -> None:
    """No economy order and no tactical order -> the bridge gets ``[]``.

    This is the historical ``noop``: ``applyStep`` dispatches no command and
    answers with one accepted receipt, so construction in flight keeps running.
    """

    state = _lake_state()
    assert decode_action_multi(
        _multi(_slot(ECONOMY_EMPTY_INDEX), _slot(TACTICAL_EMPTY_INDEX)),
        state,
        DEFAULT_CONFIG,
    ) == []


# ----------------------------------------------------------------------
# 4. The core invariant: an empty tactical slot interrupts nothing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("economy_kind", ECONOMY_ACTION_KINDS)
def test_empty_tactical_slot_never_emits_a_build_interrupting_order(
    economy_kind: str,
) -> None:
    """For *every* economy choice, an empty tactical slot stays silent.

    This is the regression guard for the reported symptom ("the builder charges
    the enemy base instead of finishing the mine"). It is asserted per economy
    kind because the bug would come back through any single branch that
    synthesised a fallback order -- ``decode_action`` falls back to ``hold`` when
    e.g. there is no factory, and a ``hold`` in the tactical slot is precisely
    what cancels the build.
    """

    state = _lake_state()
    orders = decode_action_multi(
        _multi(
            _slot(ECON[economy_kind]),
            _slot(TACTICAL_EMPTY_INDEX, group=0),
        ),
        state,
        DEFAULT_CONFIG,
    )
    interrupting = [
        order for order in orders if order["kind"] in BUILD_INTERRUPTING
    ]
    assert interrupting == [], (
        f"economy={economy_kind} with an empty tactical slot emitted "
        f"{interrupting}, which would cancel the build in flight"
    )
    # And the only order that may survive is the economy one itself.
    assert len(orders) <= 1


def test_empty_tactical_slot_with_no_builder_still_interrupts_nothing() -> None:
    """Even the degenerate state must not synthesise a ``hold``.

    ``decode_action`` turns an unbuildable economy order into ``hold`` (a
    deliberate single-slot behaviour: *something* must be sent or the decision is
    wasted). In the two-slot layout that fallback is dangerous, because it lands
    in a decision that may also carry a live build. A ``hold`` aimed at the
    builder's own group cancels the construction, so an empty tactical slot must
    stay empty even when the economy slot degrades.
    """

    # No builder at all: ``build_structure`` cannot resolve one.
    state = _lake_state(own_entities=[{"entity_id": 23, "is_building": True,
                                       "is_factory": True, "hp": 4000.0}])
    orders = decode_action_multi(
        _multi(_slot(ECON["build_structure"]), _slot(TACTICAL_EMPTY_INDEX, group=0)),
        state,
        DEFAULT_CONFIG,
    )
    assert [order["kind"] for order in orders if order["kind"] in BUILD_INTERRUPTING] == []


def test_an_empty_tactical_slot_is_not_a_noop_order() -> None:
    """The empty slot emits **nothing**; it does not emit ``noop``.

    Both satisfy the bridge, but only "nothing" keeps ``len(orders)`` honest for
    the reward path and the receipt accounting: a spurious ``noop`` would spend a
    receipt slot that ``reward_shaping.invalid`` counts against us if the bridge
    ever changed its mind about accepting it.
    """

    state = _lake_state()
    orders = decode_action_multi(
        _multi(_slot(ECON["build_extractor"]), _slot(TACTICAL_EMPTY_INDEX)),
        state,
        DEFAULT_CONFIG,
    )
    assert [order["kind"] for order in orders] == ["build_extractor"]


# ----------------------------------------------------------------------
# 5. Decoder plumbing: reuse, validation, determinism
# ----------------------------------------------------------------------
def test_multi_decode_matches_single_decode_per_slot() -> None:
    """Each slot decodes exactly like ``decode_action`` would on its own.

    Guarantees the two paths cannot drift: a fix to the build/produce/upgrade
    rules lands in both because ``decode_action_multi`` calls the same
    translation (``_decode_heads``) rather than a copy.
    """

    state = _lake_state()
    single_slot = [KIND[kind] for kind in ACTION_KINDS]
    for kind_index in single_slot:
        kind = ACTION_KINDS[kind_index]
        single_action = np.array([kind_index, 4, 3, 3, 0], dtype=np.int64)
        expected = decode_action(single_action, state, DEFAULT_CONFIG)

        if kind in ECONOMY_ACTION_KINDS:
            orders = decode_action_multi(
                _multi(_slot(ECON[kind]), _slot(TACTICAL_EMPTY_INDEX)),
                state, DEFAULT_CONFIG,
            )
        elif kind in TACTICAL_ACTION_KINDS:
            orders = decode_action_multi(
                _multi(_slot(ECONOMY_EMPTY_INDEX), _slot(TAC[kind])),
                state, DEFAULT_CONFIG,
            )
        else:  # pragma: no cover - every kind is in one of the two vocabularies
            continue
        assert orders == expected, f"{kind}: {orders} != {expected}"


def test_multi_decode_rejects_a_single_slot_vector() -> None:
    """A 5-head vector is a caller bug, not something to guess at."""

    with pytest.raises(ValueError, match="10 heads"):
        decode_action_multi(
            np.array([0, 4, 3, 3, 0], dtype=np.int64), _lake_state(), DEFAULT_CONFIG
        )


def test_multi_decode_accepts_a_2d_batch_of_one() -> None:
    """``decode_action`` reshapes its input; the two-slot decoder must too."""

    state = _lake_state()
    flat = _multi(_slot(ECON["build_extractor"]), _slot(TAC["hold"]))
    assert decode_action_multi(flat.reshape(1, -1), state, DEFAULT_CONFIG) == \
        decode_action_multi(flat, state, DEFAULT_CONFIG)


def test_out_of_range_kind_indices_fold_instead_of_crashing() -> None:
    """Hand-built arrays and older checkpoints can index past the vocabulary."""

    state = _lake_state()
    # ``len(ECONOMY_ACTION_KINDS) + 1`` folds back onto index 1 (build_structure),
    # and the tactical head folds onto ``move_group``.
    orders = decode_action_multi(
        _multi(
            _slot(len(ECONOMY_ACTION_KINDS) + 1),
            _slot(len(TACTICAL_ACTION_KINDS) + 0),
        ),
        state,
        DEFAULT_CONFIG,
    )
    assert [order["kind"] for order in orders] == ["build_structure", "move_group"]


def test_decoding_is_side_effect_free_and_repeatable() -> None:
    """Same input, same output; the state dict is never mutated."""

    state = _lake_state()
    snapshot = {key: list(value) if isinstance(value, list) else value
                for key, value in state.items()}
    action = _multi(_slot(ECON["build_extractor"], target=1),
                    _slot(TAC["attack_move"], group=2))
    first = decode_action_multi(action, state, DEFAULT_CONFIG)
    second = decode_action_multi(action, state, DEFAULT_CONFIG)
    assert first == second
    assert state["own_entities"] == snapshot["own_entities"]
    assert state["resource_slots"] == snapshot["resource_slots"]
