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


# ----------------------------------------------------------------------
# Columns 7/8: can this entity attack, and how far does it reach?
#
# The identity block could already say *what* an entity was, but not whether
# walking into range of it is fatal: the policy had to induce "the thing with
# hash 0.236 kills me" from reward alone, and a builder was observed walking
# straight into the enemy base to die.
# ----------------------------------------------------------------------

#: Column indices inside one identity slot (``MicroDuelConfig.type_features`` = 9).
CAN_ATTACK = 7
ATTACK_RANGE = 8

# Real bridge payloads: ``can_attack``/``attack_range`` come from
# ``IronZeroBridge.entityJson`` (engine ``Unit.canAttack()`` and
# ``OrderableUnit.getMaxAttackRange()``).
ATTACKING_TANK = dict(TANK, can_attack=True, attack_range=130.0)
PEACEFUL_BUILDER = dict(BUILDER, can_attack=False, attack_range=0.0)
# Measured on a real [p2]Lake (2p) game (``scripts/rwpp_obs_probe.py``) -- a
# command center in stock 1.15 *does* shoot, so it is NOT a non-combatant. Its
# range is 280 px, not the tank's 130 px.
ARMED_COMMAND_CENTER = dict(COMMAND_CENTER, can_attack=True, attack_range=280.0)


def test_can_attack_column_separates_builder_from_tank() -> None:
    """The core regression: a builder must not look like something that fights."""

    vector = encode_observation(_state([ATTACKING_TANK, PEACEFUL_BUILDER]), CFG)
    identity = _identity(vector)
    # ``prioritise_own`` puts the builder (rank 2) before the combat unit (rank 3),
    # so slot 0 is the builder and slot 1 is the tank.
    builder_row, tank_row = identity[0], identity[1]
    assert builder_row[3] == 1.0, "slot 0 should be the builder"
    assert tank_row[3] == 0.0, "slot 1 should be the tank"

    assert builder_row[CAN_ATTACK] == 0.0
    assert tank_row[CAN_ATTACK] == 1.0
    # An attack range of 0 for the builder, 130 px -> 0.13 for the tank.
    assert builder_row[ATTACK_RANGE] == 0.0
    assert abs(float(tank_row[ATTACK_RANGE]) - 0.13) < 1e-6


def test_command_center_reports_its_real_weapon() -> None:
    """Real-game values: the command center shoots (280 px), the builder cannot.

    Measured on ``[p2]Lake (2p)``::

        commandCenter  can_attack=true   attack_range=280.0
        builder        can_attack=false  attack_range=0.0

    A command center therefore must not be lumped in with a builder as
    "harmless": the policy has to be able to tell that standing next to the
    enemy base is dangerous even for a unit that cannot shoot back.
    """

    vector = encode_observation(
        _state([ARMED_COMMAND_CENTER, PEACEFUL_BUILDER]), CFG
    )
    identity = _identity(vector)
    # Buildings are prioritised into slot 0, the builder takes slot 1.
    center_row, builder_row = identity[0], identity[1]
    assert center_row[1] == 1.0                   # is_building
    assert builder_row[3] == 1.0                  # is_builder

    assert center_row[CAN_ATTACK] == 1.0          # armed
    assert center_row[ATTACK_RANGE] > 0.0         # 280 px -> 0.28
    assert abs(float(center_row[ATTACK_RANGE]) - 0.28) < 1e-6
    assert builder_row[CAN_ATTACK] == 0.0         # harmless
    assert builder_row[ATTACK_RANGE] == 0.0
    # The two are distinguishable on the attack columns *and* on the type columns.
    assert not np.array_equal(center_row, builder_row)


def test_new_attack_columns_stay_in_the_unit_interval() -> None:
    """The block's [0, 1] contract survives absurd and malformed engine values."""

    cases = [
        dict(ATTACKING_TANK, attack_range=130.0),
        dict(ATTACKING_TANK, attack_range=0.0),
        dict(ATTACKING_TANK, attack_range=5000.0),      # clamped
        dict(ATTACKING_TANK, attack_range=float("nan")),  # -> 0, not NaN
        dict(ATTACKING_TANK, attack_range=float("inf")),  # -> 0
        dict(ATTACKING_TANK, attack_range=-50.0),       # -> 0
        dict(ATTACKING_TANK, attack_range="?"),         # unparseable -> 0
        dict(ATTACKING_TANK, attack_range=None),        # missing-ish -> 0
        dict(PEACEFUL_BUILDER, can_attack="yes"),       # truthy -> 1.0
    ]
    for unit in cases:
        identity = _identity(encode_observation(_state([unit]), CFG))
        assert np.isfinite(identity).all(), f"non-finite feature for {unit}"
        assert (identity >= 0.0).all(), f"negative feature for {unit}"
        assert (identity <= 1.0).all(), f"feature > 1 for {unit}"
    # A hostile range is clamped, not propagated.
    huge = _identity(encode_observation(_state([dict(ATTACKING_TANK, attack_range=5000.0)]), CFG))
    assert huge[0][ATTACK_RANGE] == 1.0


def test_attack_columns_are_appended_without_touching_the_first_seven() -> None:
    """Appending must not shift a column: an old checkpoint keeps its meaning."""

    # The exact same entity, once without and once with the two new fields. The
    # bridge change is purely additive, so the leading columns must be identical.
    before = dict(TANK)
    after = dict(before, can_attack=True, attack_range=130.0)
    vec_before = encode_observation(_state([before]), CFG)
    vec_after = encode_observation(_state([after]), CFG)

    row_before = _identity(vec_before)[0]
    row_after = _identity(vec_after)[0]
    # Columns 0..6 -- occupancy, is_building, is_factory, is_builder, is_air_unit,
    # build_progress, type hash -- are bit-identical.
    np.testing.assert_array_equal(row_before[:7], row_after[:7])
    # Only the two appended columns moved.
    assert row_before[CAN_ATTACK] == 0.0
    assert row_after[CAN_ATTACK] == 1.0
    assert row_before[ATTACK_RANGE] == 0.0
    assert row_after[ATTACK_RANGE] > 0.0

    # The overall vector length is unchanged by the payload, and every earlier
    # block (global / geometric / grid) keeps its exact offset and values.
    assert vec_before.shape == vec_after.shape == (CFG.flat_obs_dim,)
    geo_end = CFG.type_block_offset
    np.testing.assert_array_equal(vec_before[:geo_end], vec_after[:geo_end])


def test_enemy_attack_capability_is_observable() -> None:
    """An enemy turret must look dangerous in the *enemy* identity block too."""

    turret = dict(
        SCOUT, entity_id=9, entity_type_id=12, is_building=True,
        can_attack=True, attack_range=290.0,
    )
    as_tank = encode_observation(
        _state([PEACEFUL_BUILDER], [dict(ATTACKING_TANK, entity_id=9)]), CFG
    )
    as_turret = encode_observation(_state([PEACEFUL_BUILDER], [turret]), CFG)
    assert _identity(as_tank, block=1)[0][CAN_ATTACK] == 1.0
    assert _identity(as_turret, block=1)[0][CAN_ATTACK] == 1.0
    # Same capability, different reach: 130 px vs 290 px.
    assert abs(float(_identity(as_tank, block=1)[0][ATTACK_RANGE]) - 0.13) < 1e-6
    assert abs(float(_identity(as_turret, block=1)[0][ATTACK_RANGE]) - 0.29) < 1e-6


def test_type_features_grew_to_nine_with_the_attack_columns_last() -> None:
    """The layout constant and the feature vector must agree, at the tail."""

    assert CFG.type_features == 9
    from ironzero.observation.features import _entity_type_features

    feats = _entity_type_features(dict(ATTACKING_TANK))
    assert feats.shape == (CFG.type_features,)
    assert feats[CAN_ATTACK] == 1.0
    assert feats[ATTACK_RANGE] > 0.0
