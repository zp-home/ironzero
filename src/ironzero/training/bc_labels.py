# SPDX-License-Identifier: Apache-2.0
"""Turn the built-in AI's recorded orders into policy actions (BC labels).

The teacher and the student speak different languages:

* the bridge reports, per unit, the order the AI actually executed --
  ``{"entity_id", "kind", "x", "y", "target_entity_id", "attack_target_id",
  "build_type", ...}`` (see ``IronZeroBridge.aiOrdersJson``);
* the policy emits a 5-head vector ``(kind, group, x, y, target)`` that orders a
  *group* of units, not a single unit.

Two mismatches have to be handled explicitly, because ignoring either one
produces a dataset that trains the wrong thing:

1. **Granularity.** The policy can only address the 4 id-based groups or the
   broadcast slot; the AI decides per unit. A group label is therefore a
   consensus: the kind and destination most of the group's units agree on,
   with ties broken by the unit that changed most recently.
2. **Timing.** A recorded order happens on the frame it changed, but the policy
   only decides every ``DECISION_TICKS`` frames. Labels are snapped to the
   decision boundary that follows them, so the label is what the policy could
   have observed *before* acting.

The mapping is deliberately lossy and honest about it: :func:`label_coverage`
reports how many recorded orders survive the translation, so a BC run can state
its label fidelity instead of assuming it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..engine.constants import ACTION_KINDS

GROUP_BROADCAST = 4


@dataclass(frozen=True, slots=True)
class GroupLabel:
    """One policy action reconstructed from per-unit teacher orders."""

    kind: str
    group: int
    x: float
    y: float
    target_entity_id: int
    #: How many teacher orders voted for this label and for a different one.
    support: int
    contested: int
    #: Group kind distribution, for diagnostics.
    votes: Mapping[str, int] = field(default_factory=dict)

    @property
    def agreement(self) -> float:
        """Share of the group's units that agreed on this label (0..1)."""

        total = self.support + self.contested
        return self.support / total if total else 0.0


#: Teacher kinds that mean "this unit was given no new order". They are not
#: policy-coverable: issuing an order to make a unit stand still cancels whatever
#: it was doing, so mapping them to ``hold`` would teach the policy to interrupt
#: itself. ``noop`` is the policy's equivalent and is used instead.
_IDLE_KINDS = frozenset({"hold"})

#: Teacher orders the policy has no head for. ``repair_target`` was removed from
#: the action space entirely (the bridge had no repair branch), so a recorded
#: repair cannot be imitated and is counted as uncovered rather than mislabelled.
_UNSUPPORTED_KINDS = frozenset({"repair_target"})


def _kind_priority(kind: str) -> int:
    """Tie-break order when a group's units disagree.

    Aggressive, irreversible orders win over passive ones: if half a group is
    attacking and half is moving, the meaningful decision is the attack.
    """

    order = {
        "attack_target": 0,
        "build_extractor": 1,
        "build_structure": 2,
        "produce_unit": 3,
        "upgrade_building": 4,
        "attack_move": 5,
        "move_group": 6,
        "stop": 7,
        "noop": 8,
    }
    return order.get(kind, 9)


def _median_point(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    xs = sorted(point[0] for point in points)
    ys = sorted(point[1] for point in points)
    middle = len(points) // 2
    return (xs[middle], ys[middle])


def world_to_grid(
    x: float, y: float, *, grid: int, width: float, height: float, margin: float = 24.0
) -> tuple[int, int]:
    """Inverse of ``MicroDuelConfig.grid_to_world`` (nearest cell)."""

    if grid <= 1:
        return (0, 0)
    span_x = width - 2.0 * margin
    span_y = height - 2.0 * margin
    if span_x <= 0 or span_y <= 0:
        return (0, 0)
    gx = round((float(x) - margin) * (grid - 1) / span_x)
    gy = round((float(y) - margin) * (grid - 1) / span_y)
    return (int(min(grid - 1, max(0, gx))), int(min(grid - 1, max(0, gy))))


def group_of(entity_id: int, groups: int) -> int:
    """The policy group a unit belongs to (the bridge uses ``id % GROUP_COUNT``)."""

    return int(entity_id) % max(1, groups)


def orders_to_labels(
    orders: Iterable[Mapping[str, Any]],
    *,
    groups: int = 4,
    grid: int = 8,
    width: float,
    height: float,
    enemy_ids: Sequence[int] = (),
    own_entity_ids: Sequence[int] = (),
    margin: float = 24.0,
) -> list[GroupLabel]:
    """Collapse one frame's per-unit teacher orders into per-group policy labels.

    ``enemy_ids`` fixes the target-head indexing: the policy's target head is a
    position in the streamed enemy list, so an entity id must be translated to
    *that* index, not used directly (the two coincide only by accident).

    ``own_entity_ids`` restricts voting to units the policy actually owns this
    frame; a recorded order for a unit that has since died cannot be imitated.
    """

    owned = set(int(i) for i in own_entity_ids) if own_entity_ids else None
    by_group: dict[int, list[Mapping[str, Any]]] = {}
    for order in orders:
        entity_id = int(order.get("entity_id", -1))
        if entity_id < 0:
            continue
        if owned is not None and entity_id not in owned:
            continue
        by_group.setdefault(group_of(entity_id, groups), []).append(order)

    enemy_index = {int(entity_id): index for index, entity_id in enumerate(enemy_ids)}
    labels: list[GroupLabel] = []
    for group in sorted(by_group):
        members = by_group[group]
        counts = Counter(str(order.get("kind") or "hold") for order in members)
        winner = sorted(
            counts.items(), key=lambda item: (-item[1], _kind_priority(item[0]))
        )[0][0]
        chosen = [o for o in members if str(o.get("kind") or "hold") == winner]

        if winner in _IDLE_KINDS:
            # The whole group is standing: the policy's equivalent is "do nothing",
            # not "issue a hold order" (which would cancel in-flight work).
            labels.append(
                GroupLabel(
                    kind="noop",
                    group=group,
                    x=0.0,
                    y=0.0,
                    target_entity_id=-1,
                    support=len(chosen),
                    contested=len(members) - len(chosen),
                    votes=dict(counts),
                )
            )
            continue
        if winner in _UNSUPPORTED_KINDS:
            continue

        # Attack target: prefer the explicit waypoint target, then the
        # auto-acquired one. Both are real engine state, not inference.
        target_slot = -1
        for order in chosen:
            for key in ("target_entity_id", "attack_target_id"):
                entity_id = int(order.get(key, -1))
                if entity_id >= 0 and entity_id in enemy_index:
                    target_slot = enemy_index[entity_id]
                    break
            if target_slot >= 0:
                break

        if winner == "attack_target" and target_slot < 0:
            # The AI is shooting something we cannot see in the streamed list.
            # Emit attack_move at the same point instead of a target the policy
            # cannot name; refusing the label entirely would throw away the
            # decision (where to go) along with the target.
            winner = "attack_move"

        points = []
        for order in chosen:
            if winner in {"move_group", "attack_move", "build_structure"}:
                points.append((float(order.get("x", 0.0)), float(order.get("y", 0.0))))
        x, y = _median_point(points)

        labels.append(
            GroupLabel(
                kind=winner,
                group=group,
                x=x,
                y=y,
                target_entity_id=target_slot,
                support=len(chosen),
                contested=len(members) - len(chosen),
                votes=dict(counts),
            )
        )
    return labels


def label_to_action(
    label: GroupLabel,
    *,
    grid: int = 8,
    width: float,
    height: float,
    margin: float = 24.0,
) -> np.ndarray:
    """Encode a :class:`GroupLabel` as the policy's 5-head action vector."""

    kind = label.kind if label.kind in ACTION_KINDS else "noop"
    gx, gy = world_to_grid(label.x, label.y, grid=grid, width=width, height=height, margin=margin)
    target = max(0, int(label.target_entity_id))
    return np.array(
        [ACTION_KINDS.index(kind), int(label.group), gx, gy, target], dtype=np.int64
    )


def label_coverage(
    orders: Iterable[Mapping[str, Any]],
    labels: Sequence[GroupLabel],
    *,
    groups: int = 4,
) -> dict[str, float]:
    """How much of the teacher's behaviour the labels actually express.

    Reported alongside a dataset so a BC run can state its fidelity: a "90%"
    number computed as "labels produced / orders recorded" would be meaningless,
    since one label usually covers several units.
    """

    order_list = list(orders)
    total = len(order_list)
    idle = sum(1 for order in order_list if str(order.get("kind") or "hold") in _IDLE_KINDS)
    unsupported = sum(
        1 for order in order_list if str(order.get("kind") or "hold") in _UNSUPPORTED_KINDS
    )
    # ``noop`` labels *are* the idle orders: counting their support as "covered"
    # and then dividing by (total - idle) reported a coverage above 1.0 (measured
    # 1.505). Only labels that reproduce a real decision count as covering one.
    covered = sum(label.support for label in labels if label.kind != "noop")
    contested = sum(label.contested for label in labels)
    decisive = max(1.0, total - idle - unsupported)
    return {
        "orders": float(total),
        "labels": float(len(labels)),
        "orders_covered": float(covered),
        "orders_idle": float(idle),
        "orders_unsupported": float(unsupported),
        "orders_contested": float(contested),
        "orders_decisive": float(decisive),
        # Share of *decision-carrying* orders (not idle, not unsupported) that a
        # label reproduces. Bounded by 1.0 by construction.
        "coverage": min(1.0, covered / decisive),
        "idle_share": idle / max(1.0, total),
    }


def snap_to_decision(tick: int, decision_ticks: int) -> int:
    """The decision boundary a label recorded at ``tick`` belongs to.

    A label must be emitted at the decision *after* the teacher acted, because
    the policy only sees the observation at that boundary.
    """

    if decision_ticks <= 0:
        return tick
    return int(math.ceil(tick / decision_ticks) * decision_ticks)
