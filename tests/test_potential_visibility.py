# SPDX-License-Identifier: Apache-2.0
"""``potential`` must be able to see an enemy that out-produces us.

This locks a defect that made the dominant reward term a **constant zero** for a
whole episode on every economy map, which nothing else in the suite caught
because the term was *plausible*: same formula, same clamps, no crash.

Measured on Lake (40-update run, 16 full episodes)::

    reward_components: potential = +0.000000
    eval: own_alive=1, enemy_alive=20

The cause was the ``[0, 1]`` clamp in ``combat_value_fractions``. Both sides
start at 4170 HP (a 4000 HP command center plus a 170 HP builder), and the AI
reinforces well past that. Once total combat value exceeds the *opening* total it
saturated at 1.0, so ``potential = own - enemy = 1.0 - 1.0 = 0.0`` forever --
even while the enemy went from 2 units to 20 and we lost 7. The policy could not
observe being out-produced, which is exactly what it had to learn to avoid.
"""

from __future__ import annotations

import pytest

from ironzero.rwpp.reward import (
    combat_value_fractions,
    combat_value_potential,
    hp_fractions,
)

# Lake's real opening: a 4000 HP command center plus a 170 HP builder per side.
OPENING_REF = 4170.0


def _side(extra_tanks: int) -> list[dict]:
    units = [
        {"hp": 4000.0, "max_hp": 4000.0, "is_building": True},
        {"hp": 170.0, "max_hp": 170.0},
    ]
    units += [{"hp": 210.0, "max_hp": 210.0} for _ in range(extra_tanks)]
    return units


def _state(own_extra: int, enemy_extra: int) -> dict:
    return {
        "own_entities": _side(own_extra),
        "visible_enemy_entities": _side(enemy_extra),
    }


def test_matched_opening_armies_are_neutral() -> None:
    state = _state(0, 0)
    assert combat_value_potential(state, OPENING_REF, OPENING_REF) == pytest.approx(0.0)


@pytest.mark.parametrize("tanks", [1, 2, 4, 8, 16, 24])
def test_enemy_reinforcement_is_visible_as_a_negative_potential(tanks: int) -> None:
    """Every extra enemy unit must lower the potential. This used to be 0.0."""

    state = _state(0, tanks)
    potential = combat_value_potential(state, OPENING_REF, OPENING_REF)
    assert potential < 0.0, f"{tanks} extra enemy units still read as neutral"
    assert potential == pytest.approx(-0.0504 * tanks, abs=0.01)


def test_our_own_reinforcement_is_visible_as_a_positive_potential() -> None:
    state = _state(4, 0)
    potential = combat_value_potential(state, OPENING_REF, OPENING_REF)
    assert potential > 0.0
    assert potential == pytest.approx(0.2014, abs=0.01)


def test_potential_grows_monotonically_with_the_enemy_army() -> None:
    values = [
        combat_value_potential(_state(0, n), OPENING_REF, OPENING_REF)
        for n in range(0, 25)
    ]
    assert values[0] == pytest.approx(0.0)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


def test_potential_is_antisymmetric_between_the_two_sides() -> None:
    """Swapping the armies must flip the sign exactly."""

    forward = combat_value_potential(_state(3, 0), OPENING_REF, OPENING_REF)
    backward = combat_value_potential(_state(0, 3), OPENING_REF, OPENING_REF)
    assert forward == pytest.approx(-backward)


def test_fractions_are_no_longer_saturated_at_one() -> None:
    own, enemy = combat_value_fractions(_state(0, 16), OPENING_REF, OPENING_REF)
    assert own == pytest.approx(1.0)
    assert enemy > 1.0, "an enemy above the opening total must exceed 1.0"


def test_a_wiped_army_reads_as_a_large_negative_potential() -> None:
    """Losing everything must not be flattened to -1.0 by a clamp."""

    state = {
        "own_entities": [{"hp": 170.0, "max_hp": 170.0}],
        "visible_enemy_entities": _side(20),
    }
    potential = combat_value_potential(state, OPENING_REF, OPENING_REF)
    assert potential < -1.5


def test_hp_fractions_stay_clamped() -> None:
    """``hp_fractions`` feeds the timeout adjudication; its contract is unchanged.

    It must keep returning ``[0, 1]`` -- ``_decide_by_army`` compares those two
    values against each other, and the truncation tests depend on the clamp.
    """

    own, enemy = hp_fractions(_state(0, 16), OPENING_REF, OPENING_REF)
    assert 0.0 <= own <= 1.0
    assert 0.0 <= enemy <= 1.0
