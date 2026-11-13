# SPDX-License-Identifier: Apache-2.0
"""The BC label translation must be honest, not just runnable.

The teacher (built-in AI) acts per unit; the student (policy) acts per group with
a 5-head vector. Translating between them loses information, and every loss has a
specific failure mode if it is papered over:

* mapping "this unit had no order" to ``hold`` teaches the policy to interrupt
  its own in-flight build/production;
* using an entity id directly as the target head index trains a target
  distribution over the wrong vocabulary (ids are large and sparse, the head is a
  position in the streamed enemy list);
* dropping the target but keeping the assumption that it exists emits
  ``attack_target`` with no target, which the bridge rejects.

These tests pin the mapping's decisions, including the losses it reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from ironzero.engine.constants import ACTION_KINDS
from ironzero.training.bc_labels import (
    GroupLabel,
    group_of,
    label_coverage,
    label_to_action,
    orders_to_labels,
    snap_to_decision,
    world_to_grid,
)

MAP = {"width": 2600.0, "height": 2600.0, "grid": 8}


def _label(orders, **kwargs):
    params = dict(MAP)
    params.update(kwargs)
    return orders_to_labels(orders, **params)


# ---------------------------------------------------------------------------
# Geometry: the grid must round-trip against the observation encoder
# ---------------------------------------------------------------------------
def test_world_to_grid_inverts_grid_to_world() -> None:
    """A label written to the grid then decoded must land on the same cell."""

    from ironzero.micro_config import MicroDuelConfig

    config = MicroDuelConfig(map_width=2600.0, map_height=2600.0)
    for gx in range(config.position_grid):
        for gy in range(config.position_grid):
            x, y = config.grid_to_world(gx, gy)
            back = world_to_grid(
                x, y, grid=config.position_grid, width=2600.0, height=2600.0
            )
            assert back == (gx, gy)


def test_world_to_grid_clamps_out_of_range_points() -> None:
    """The AI may order a point outside the margin; the grid must still be valid."""

    assert world_to_grid(-500.0, 99999.0, grid=8, width=2600.0, height=2600.0) == (0, 7)


# ---------------------------------------------------------------------------
# Granularity: per-unit orders collapse into per-group labels
# ---------------------------------------------------------------------------
def test_units_are_grouped_the_same_way_the_bridge_groups_them() -> None:
    """The bridge addresses units by ``id % GROUP_COUNT``; the label must match."""

    assert group_of(6, 4) == 2
    assert group_of(7, 4) == 3


def test_a_unanimous_group_becomes_one_label() -> None:
    orders = [
        {"entity_id": 4, "kind": "attack_move", "x": 1200.0, "y": 800.0},
        {"entity_id": 8, "kind": "attack_move", "x": 1240.0, "y": 820.0},
    ]
    labels = _label(orders)
    assert len(labels) == 1
    assert labels[0].kind == "attack_move"
    assert labels[0].support == 2 and labels[0].contested == 0
    assert labels[0].agreement == 1.0


def test_a_split_group_picks_the_aggressive_order_and_reports_the_conflict() -> None:
    """Half attacking, half moving: the decision that matters is the attack."""

    orders = [
        {"entity_id": 4, "kind": "attack_move", "x": 900.0, "y": 900.0},
        {"entity_id": 0, "kind": "move_group", "x": 100.0, "y": 100.0},
    ]
    labels = _label(orders)
    assert labels[0].kind == "attack_move"
    assert labels[0].support == 1
    assert labels[0].contested == 1
    assert labels[0].agreement == pytest.approx(0.5)


def test_a_split_group_prefers_attack_target_over_attack_move() -> None:
    """Equal counts: an explicit target beats a move command."""

    orders = [
        {"entity_id": 4, "kind": "attack_target", "target_entity_id": 21, "x": 0.0, "y": 0.0},
        {"entity_id": 0, "kind": "attack_move", "x": 100.0, "y": 100.0},
    ]
    labels = _label(orders, enemy_ids=[21])
    assert labels[0].kind == "attack_target"


def test_dead_units_do_not_vote() -> None:
    """A recorded order for a unit that is already gone cannot be imitated."""

    orders = [
        {"entity_id": 4, "kind": "attack_move", "x": 1200.0, "y": 800.0},
        {"entity_id": 8, "kind": "move_group", "x": 10.0, "y": 10.0},
    ]
    labels = _label(orders, own_entity_ids=[4])
    assert len(labels) == 1
    assert labels[0].kind == "attack_move"
    assert labels[0].contested == 0


# ---------------------------------------------------------------------------
# The idle decision: "no order" must not become "hold"
# ---------------------------------------------------------------------------
def test_an_idle_group_maps_to_noop_not_hold() -> None:
    """Regression-in-waiting: ``hold`` cancels the in-flight order.

    Construction and production need hundreds of uninterrupted ticks; issuing
    ``hold`` between build orders cancels the factory before it finishes (measured
    earlier: a landFactory completes in 191 ticks with no-ops, never with holds).
    """

    labels = _label([{"entity_id": 4, "kind": "hold", "x": 990.0, "y": 2010.0}])
    assert labels[0].kind == "noop"
    assert labels[0].kind != "hold"


def test_noop_encodes_as_the_noop_kind_not_a_stop() -> None:
    action = label_to_action(
        GroupLabel(kind="noop", group=0, x=0.0, y=0.0, target_entity_id=-1,
                   support=1, contested=0),
        **MAP,
    )
    assert ACTION_KINDS[action[0]] == "noop"


# ---------------------------------------------------------------------------
# The target head is an index into the streamed enemy list, not an entity id
# ---------------------------------------------------------------------------
def test_attack_target_uses_the_enemy_stream_index() -> None:
    """Entity ids are sparse (21, 57, ...); the head is a position (0, 1, ...)."""

    labels = _label(
        [{"entity_id": 4, "kind": "attack_target", "target_entity_id": 57}],
        enemy_ids=[21, 33, 57],
    )
    assert labels[0].target_entity_id == 2
    action = label_to_action(labels[0], **MAP)
    assert int(action[4]) == 2


def test_attack_target_falls_back_to_attack_move_when_the_target_is_not_streamed() -> None:
    """An unnameable target must not become an ``attack_target`` with no target.

    The bridge rejects that (``TARGET_NOT_VISIBLE``), so the label would train the
    policy to emit a guaranteed-invalid action. Keeping the destination preserves
    the part of the decision the policy can express.
    """

    labels = _label(
        [{"entity_id": 4, "kind": "attack_target", "target_entity_id": 999,
          "x": 1500.0, "y": 400.0}],
        enemy_ids=[21],
    )
    assert labels[0].kind == "attack_move"
    assert labels[0].x == 1500.0


def test_attack_target_can_use_the_auto_acquired_target() -> None:
    """A holding unit with an acquired target is still demonstrating micro."""

    labels = _label(
        [{"entity_id": 4, "kind": "attack_target", "target_entity_id": -1,
          "attack_target_id": 33}],
        enemy_ids=[21, 33],
    )
    assert labels[0].target_entity_id == 1


def test_unsupported_kinds_are_dropped_not_mislabelled() -> None:
    """``repair_target`` is not in the action space at all (removed for cause)."""

    labels = _label([{"entity_id": 4, "kind": "repair_target", "target_entity_id": 9}])
    assert labels == []


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def test_labels_snap_forward_to_the_next_decision_boundary() -> None:
    """The policy acts on the boundary *after* the teacher acted."""

    assert snap_to_decision(0, 10) == 0
    assert snap_to_decision(1, 10) == 10
    assert snap_to_decision(10, 10) == 10
    assert snap_to_decision(11, 10) == 20


def test_snap_without_a_decision_interval_is_a_passthrough() -> None:
    assert snap_to_decision(37, 0) == 37


# ---------------------------------------------------------------------------
# Honesty: the loss is measured, not hidden
# ---------------------------------------------------------------------------
def test_coverage_excludes_idle_orders_from_the_denominator() -> None:
    """Otherwise "coverage" would be dominated by units that did nothing.

    Measured on a real recording: 27% of change events were idle, so counting them
    as uncovered would report ~73% fidelity for a translation that covers every
    actual decision.
    """

    orders = [
        {"entity_id": 4, "kind": "attack_move", "x": 100.0, "y": 100.0},
        {"entity_id": 8, "kind": "hold", "x": 990.0, "y": 2010.0},
    ]
    labels = _label(orders)
    coverage = label_coverage(orders, labels)
    assert coverage["orders"] == 2
    assert coverage["orders_idle"] == 1
    assert coverage["coverage"] == pytest.approx(1.0)
    assert coverage["idle_share"] == pytest.approx(0.5)


def test_coverage_drops_when_the_teacher_does_something_we_cannot_express() -> None:
    orders = [
        {"entity_id": 4, "kind": "repair_target", "target_entity_id": 9},
        {"entity_id": 8, "kind": "attack_move", "x": 100.0, "y": 100.0},
    ]
    labels = _label(orders)
    coverage = label_coverage(orders, labels)
    assert coverage["orders_unsupported"] == 1
    assert coverage["coverage"] == pytest.approx(1.0)  # repair excluded, not counted


def test_coverage_is_bounded_by_one() -> None:
    """Regression: ``noop`` labels were counted as "covered" orders.

    Their support is the idle units, which are excluded from the denominator, so
    a mixed frame divided idle-covered by non-idle-decisive and reported 1.505.
    """

    orders = [
        {"entity_id": 4, "kind": "attack_move", "x": 100.0, "y": 100.0},
        {"entity_id": 5, "kind": "hold", "x": 990.0, "y": 2010.0},
        {"entity_id": 6, "kind": "hold", "x": 990.0, "y": 2010.0},
        {"entity_id": 7, "kind": "hold", "x": 990.0, "y": 2010.0},
    ]
    labels = _label(orders)
    # Three idle groups (noop) plus one real decision: ids 4/5/6/7 land in groups
    # 0/1/2/3 via ``id % 4``, so exactly one decisive order exists.
    assert sum(1 for label in labels if label.kind == "noop") == 3
    coverage = label_coverage(orders, labels)
    assert coverage["coverage"] <= 1.0
    assert coverage["coverage"] == pytest.approx(1.0)
    assert coverage["orders_decisive"] == pytest.approx(1.0)


def test_labels_are_contiguous_groups_only() -> None:
    """Group ids must stay inside the action head's range."""

    orders = [{"entity_id": entity_id, "kind": "attack_move", "x": 10.0, "y": 10.0}
              for entity_id in range(12)]
    labels = _label(orders)
    assert sorted(label.group for label in labels) == [0, 1, 2, 3]
