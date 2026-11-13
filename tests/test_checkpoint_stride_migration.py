# SPDX-License-Identifier: Apache-2.0
"""Stride-change checkpoint migration must keep every weight on its own feature.

Raising ``MicroDuelConfig.type_features`` 7 -> 9 (the ``can_attack`` /
``attack_range`` columns) changed the *per-entity stride*, not just a block size.
``_column_source`` used to give up on that case and return ``None``, which made
``_migrate_state_dict`` fall back to "keep the leading columns" for
``trunk.0.weight``.

That fallback is only correct when the new vector *starts* with the old one. It
does not here: both type blocks grow, so the block start of ``type_enemy`` moves
left relative to the old vector and a prefix copy walks the old *enemy* identity
columns into the new *own* identity block. Measured on the 25v25 layout: 50 of
225 new own-type columns came from the old enemy block (12800 weights in a
256-wide trunk). The policy loads without a single warning and behaves like
noise -- exactly the failure ``_migrate_state_dict`` exists to prevent.

This module pins the slot-by-slot behaviour that replaced it.
"""

from __future__ import annotations

import torch

from ironzero.training.ppo import _column_source, _layout_blocks, _migrate_state_dict

TRUNK_ROWS = 256


def _layout(type_features: int, *, max_entities: int, map_channels: int) -> dict:
    """One observation layout, matching ``encoding.encode_observation``'s order."""

    return {
        "global_dim": 15,
        "entity_features": 10,
        "type_features": type_features,
        "max_entities": max_entities,
        "map_channels": map_channels,
        "map_grid": 8,
        "target_slots": max_entities + 1,
    }


def _blocks(layout: dict) -> dict[str, tuple[int, int]]:
    """Block offsets straight from the production helper (never re-derived here).

    Re-computing these by hand is exactly how this test first went wrong: the
    ``global`` block is part of the offset walk, so ``type_own`` starts *after* it.
    Using ``_layout_blocks`` keeps the test honest about the real layout.
    """

    return {name: (start, length) for name, start, length in _layout_blocks(layout)}


def _total(layout: dict) -> int:
    return sum(length for _start, length in _blocks(layout).values())


def _migrated(old_layout: dict, new_layout: dict, saved: torch.Tensor) -> torch.Tensor:
    target = {"trunk.0.weight": torch.zeros(TRUNK_ROWS, _total(new_layout))}
    out = _migrate_state_dict(
        target, {"trunk.0.weight": saved}, old_layout, new_layout
    )
    return out["trunk.0.weight"]


def test_stride_change_no_longer_returns_none() -> None:
    """7 -> 9 must produce an explicit mapping, not the prefix-copy fallback."""

    old = _layout(7, max_entities=25, map_channels=4)
    new = _layout(9, max_entities=25, map_channels=4)
    mapping = _column_source(old, new)
    assert mapping is not None, (
        "a type_features stride change must be re-mapped slot-by-slot; returning "
        "None silently corrupts every column after the type blocks"
    )
    assert len(mapping) == _total(new)


def test_old_enemy_identity_never_lands_in_the_new_own_block() -> None:
    """The concrete corruption: old enemy type columns leaking into new own type.

    Marked with distinct values so the two blocks can be told apart after the
    migration. This assertion fails on the old code for the 25v25 layout.
    """

    old = _layout(7, max_entities=25, map_channels=4)
    new = _layout(9, max_entities=25, map_channels=4)
    old_blocks = _blocks(old)
    new_blocks = _blocks(new)

    saved = torch.zeros(TRUNK_ROWS, sum(l for _s, l in old_blocks.values()))
    saved[:, old_blocks["type_own"][0]:old_blocks["type_own"][0]+old_blocks["type_own"][1]] = 0.25
    saved[:, old_blocks["type_enemy"][0]:] = 0.5

    migrated = _migrated(old, new, saved)
    own = migrated[:, new_blocks["type_own"][0]:new_blocks["type_enemy"][0]]
    enemy = migrated[:, new_blocks["type_enemy"][0]:]

    assert not bool((own == 0.5).any()), (
        "old enemy identity columns leaked into the new own identity block; every "
        "weight there now reads a feature it was never trained on"
    )
    assert bool((own == 0.25).any()), "old own identity columns were lost"
    assert bool((enemy == 0.5).any()), "old enemy identity columns were lost"


def test_appended_features_start_at_zero_and_earlier_ones_keep_their_weight() -> None:
    """Slot-by-slot: old feature f keeps its column, new features start at 0."""

    old = _layout(7, max_entities=8, map_channels=3)
    new = _layout(9, max_entities=8, map_channels=3)
    old_blocks = _blocks(old)
    new_blocks = _blocks(new)

    saved = torch.zeros(TRUNK_ROWS, sum(l for _s, l in old_blocks.values()))
    for column in range(saved.shape[1]):
        saved[:, column] = float(column % 97) + 1.0  # every column distinguishable

    migrated = _migrated(old, new, saved)

    start_old = old_blocks["type_own"][0]
    start_new = new_blocks["type_own"][0]
    max_entities = new["max_entities"]
    for record in range(max_entities):
        for feature in range(7):
            assert torch.equal(
                migrated[:, start_new + record * 9 + feature],
                saved[:, start_old + record * 7 + feature],
            ), f"record {record} feature {feature} did not keep its weight"
        for feature in (7, 8):
            assert torch.equal(
                migrated[:, start_new + record * 9 + feature],
                torch.zeros(TRUNK_ROWS),
            ), f"newly appended feature {feature} must start at zero"


def test_equal_stride_still_uses_the_cheap_path() -> None:
    """No stride change -> ``None`` so the caller keeps the plain prefix copy."""

    same = _layout(9, max_entities=25, map_channels=4)
    assert _column_source(same, dict(same)) is None


def test_block_appearing_from_nothing_gets_no_source() -> None:
    """0 -> 9 type features is a fresh block: it must be zeroed, not mis-filled."""

    old = _layout(0, max_entities=8, map_channels=3)
    new = _layout(9, max_entities=8, map_channels=3)
    mapping = _column_source(old, new)
    assert mapping is not None
    new_blocks = _blocks(new)
    type_columns = mapping[new_blocks["type_own"][0]:]
    assert type_columns and all(source is None for source in type_columns)
