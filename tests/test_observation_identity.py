# SPDX-License-Identifier: Apache-2.0
"""The observation must be able to tell *what an entity is*, not only where it is.

Regression guard for a structural blindness: the ten ``entity_features`` encode
position, hp fraction, order and distance only, so two completely different
battlefield states produced a bit-identical observation. Measured with
``runs/_scratch/probe_obs_type_blind.py`` before the fix::

    commandCenter(4000HP) + builder(170HP)   vs   tank(4000HP) + scout(170HP)
    max per-element difference = 0.00000000, differing dimensions = 0 / 367

The policy could therefore not see that it owned a factory, that a builder was
still alive, that a structure was still under construction, or that the enemy
"unit" was a turret. "Build a factory, then produce tanks" was unlearnable for
lack of information, not for lack of capacity.

Two invariants are asserted here:

1. the identity block distinguishes those states;
2. it is *appended* after the control grid, so every pre-existing column keeps
   its offset and an older checkpoint keeps its exact meaning.
"""

from __future__ import annotations

import numpy as np

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.observation.encoding import encode_observation

CFG = DEFAULT_CONFIG

# Real bridge payload shapes (see ``IronZeroBridge.entityJson``), not just a
# ``unit_type`` name: the encoder reads the booleans/ids the bridge sends.
COMMAND_CENTER = {
    "entity_id": 1, "entity_type_id": 1007741, "x": 1000.0, "y": 2000.0,
    "hp": 4000.0, "max_hp": 4000.0, "is_building": True, "is_factory": True,
    "is_builder": False, "is_air_unit": False, "build_progress": 1.0,
}
BUILDER = {
    "entity_id": 2, "entity_type_id": 2, "x": 1010.0, "y": 2000.0,
    "hp": 170.0, "max_hp": 170.0, "is_building": False, "is_factory": False,
    "is_builder": True, "is_air_unit": False, "build_progress": 1.0,
}
TANK = {
    "entity_id": 1, "entity_type_id": 1, "x": 1000.0, "y": 2000.0,
    "hp": 4000.0, "max_hp": 4000.0, "is_building": False, "is_factory": False,
    "is_builder": False, "is_air_unit": False, "build_progress": 1.0,
}
SCOUT = {
    "entity_id": 2, "entity_type_id": 1001234, "x": 1010.0, "y": 2000.0,
    "hp": 170.0, "max_hp": 170.0, "is_building": False, "is_factory": False,
    "is_builder": False, "is_air_unit": False, "build_progress": 1.0,
}


def _state(own: list[dict], enemies: list[dict] | None = None) -> dict:
    return {
        "tick": 0,
        "own_entities": own,
        "visible_enemy_entities": enemies or [],
        "global_state": {"funds": 500.0, "income": 0.0},
        "resource_slots": [],
    }


def _geo(vector: np.ndarray, block: int = 0) -> np.ndarray:
    start = CFG.global_dim + block * CFG.max_entities * CFG.entity_features
    end = start + CFG.max_entities * CFG.entity_features
    return vector[start:end].reshape(CFG.max_entities, CFG.entity_features)


def _identity(vector: np.ndarray, block: int = 0) -> np.ndarray:
    start = CFG.type_block_offset + block * CFG.max_entities * CFG.type_features
    end = start + CFG.max_entities * CFG.type_features
    return vector[start:end].reshape(CFG.max_entities, CFG.type_features)


def test_economy_and_army_states_are_no_longer_identical() -> None:
    """The exact pair that used to encode identically."""

    eco = encode_observation(_state([COMMAND_CENTER, BUILDER]), CFG)
    army = encode_observation(_state([TANK, SCOUT]), CFG)
    # The geometric block is deliberately unchanged: same hp fractions, same
    # positions, same orders -> same numbers.
    np.testing.assert_array_equal(_geo(eco), _geo(army))
    # The identity block is what tells them apart.
    assert not np.array_equal(_identity(eco), _identity(army))
    assert np.abs(eco - army).max() > 0.5


def test_identity_block_reports_building_factory_and_builder() -> None:
    eco = encode_observation(_state([COMMAND_CENTER, BUILDER]), CFG)
    identity = _identity(eco)
    occupancy, building, factory, builder, air, progress, _hashed = range(7)
    # Slot 0 = the command center (buildings are prioritised into the first slots).
    assert identity[0][occupancy] == 1.0
    assert identity[0][building] == 1.0
    assert identity[0][factory] == 1.0
    assert identity[0][builder] == 0.0
    # Slot 1 = the builder.
    assert identity[1][builder] == 1.0
    assert identity[1][building] == 0.0
    assert identity[1][factory] == 0.0
    # Padding slots read as empty, not as "a building at 0,0".
    assert identity[2:].sum() == 0.0
    assert all(0.0 <= value <= 1.0 for value in identity.reshape(-1))


def test_identity_block_separates_two_different_vehicles() -> None:
    """A tank and a scout at the same hp fraction must not look identical."""

    tank_state = _state([dict(TANK, hp=210.0, max_hp=210.0)])
    scout_state = _state([dict(SCOUT, hp=350.0, max_hp=350.0)])
    tank = encode_observation(tank_state, CFG)
    scout = encode_observation(scout_state, CFG)
    assert not np.array_equal(_identity(tank), _identity(scout))


def test_build_progress_is_observable() -> None:
    """A half-built factory must differ from a finished one."""

    finished = encode_observation(_state([COMMAND_CENTER]), CFG)
    building = encode_observation(
        _state([dict(COMMAND_CENTER, build_progress=0.35)]), CFG
    )
    progress_index = 5
    assert _identity(finished)[0][progress_index] == 1.0
    assert abs(_identity(building)[0][progress_index] - 0.35) < 1e-6
    assert not np.array_equal(_identity(finished), _identity(building))


def test_enemy_identity_is_observable_too() -> None:
    """The enemy block gets the same treatment: a turret is not a tank."""

    turret = dict(SCOUT, entity_id=9, entity_type_id=12, is_building=True)
    as_tank = encode_observation(_state([BUILDER], [dict(TANK, entity_id=9)]), CFG)
    as_turret = encode_observation(_state([BUILDER], [turret]), CFG)
    assert not np.array_equal(_identity(as_tank, block=1), _identity(as_turret, block=1))
    assert _identity(as_turret, block=1)[0][1] == 1.0  # is_building
    assert _identity(as_tank, block=1)[0][1] == 0.0


def test_identity_block_is_appended_after_the_grid() -> None:
    """Checkpoint migration relies on this offset being the *old* total size."""

    legacy_dim = (
        CFG.global_dim
        + 2 * CFG.max_entities * CFG.entity_features
        + CFG.map_channels * CFG.map_grid * CFG.map_grid
    )
    assert CFG.type_block_offset == legacy_dim
    assert CFG.flat_obs_dim == legacy_dim + 2 * CFG.max_entities * CFG.type_features


def test_missing_identity_fields_default_to_harmless_values() -> None:
    """A bridge that omits the new flags must not produce a broken observation."""

    bare = {"entity_id": 3, "x": 10.0, "y": 10.0, "hp": 100.0, "max_hp": 100.0}
    vector = encode_observation(_state([bare]), CFG)
    assert vector.shape == (CFG.flat_obs_dim,)
    assert np.isfinite(vector).all()
    identity = _identity(vector)
    assert identity[0][0] == 1.0        # occupancy
    assert identity[0][1] == 0.0        # not a building
    assert identity[0][5] == 1.0        # unknown progress reads as finished
