# SPDX-License-Identifier: Apache-2.0
"""``engagement`` / ``idle`` must only look at units that can actually fight.

The whole-army centroid diluted both combat terms with non-combatants, and the
builder is the worst case because it spends the game travelling alone. Measured
on Lake (2600x2600 px, diagonal ~3677 px), our opening army is a stationary
command center plus one builder, so the builder's 1786 px -> 100 px approach
drags the whole-army centroid by 1686/2 = 843 px and pays

    ``0.3 * 843 / 3677 = +0.0688``

of ``engagement`` for a unit with no weapon. With no attack-capable unit on our
side the honest ``engagement`` is ``0.0``, so all of that was noise -- and it
aimed the only dense combat gradient at "walk the builder at the enemy".

``idle`` had the same defect plus a magnitude problem: it charges
``idle_penalty = 0.02`` on *every* decision, and an episode is ~266 decisions,
so a policy holding ground in range could be billed ``-5.32`` -- 5x
``win_reward`` and 26x ``timeout_penalty``. Keyed off the whole-army centroid, a
builder that reached the enemy and correctly stopped was billed the whole time.

This is **not** a claim that fixing these terms stops builder suicide. The whole
round trip is worth ``0.3 * (1786 - 100) / 3677 = +0.1376`` while dying costs
``loss_penalty = 0.25``, so a one-way run nets ``-0.11`` and never paid. The
wandering is upstream: a uniform-random policy picks group and kind
independently, the builder owns a group of its own, and every move order cancels
an in-flight build. The observation gaining ``can_attack`` and the action space
gaining "keep building + hold" are what address that; this module only stops the
combat signal from being polluted.
"""

from __future__ import annotations

import pytest

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.reward_shaping import (
    combat_army_distance_norm,
    combat_centroid,
    compute_step_shaping,
    targetable_units,
    unit_can_attack,
)

# Lake's real geometry, so the arithmetic in the docstring is reproducible here.
MAP_W, MAP_H = 2600.0, 2600.0
DIAGONAL = (MAP_W ** 2 + MAP_H ** 2) ** 0.5  # ~3677 px
BASE_X, ENEMY_BASE_X = 1786.0, 100.0


def _micro():
    """A Lake-sized micro config, so distances match the measurements above."""

    from dataclasses import replace

    return replace(RwppGameConfig().micro_config(), map_width=MAP_W, map_height=MAP_H)


def _builder(x: float, *, y: float = 1300.0, with_field: bool = True, **extra) -> dict:
    unit = {
        "entity_id": 24,
        "x": x,
        "y": y,
        "hp": 170.0,
        "max_hp": 170.0,
        "is_building": False,
        "is_builder": True,
    }
    if with_field:
        unit["can_attack"] = False
    unit.update(extra)
    return unit


def _tank(x: float, *, y: float = 1300.0, entity_id: int = 100, with_field: bool = True, **extra) -> dict:
    unit = {
        "entity_id": entity_id,
        "x": x,
        "y": y,
        "hp": 210.0,
        "max_hp": 210.0,
        "is_building": False,
        "is_builder": False,
    }
    if with_field:
        unit["can_attack"] = True
    unit.update(extra)
    return unit


def _command_center(x: float = BASE_X, *, y: float = 1300.0, with_field: bool = True) -> dict:
    unit = {
        "entity_id": 23,
        "x": x,
        "y": y,
        "hp": 4000.0,
        "max_hp": 4000.0,
        "is_building": True,
        "is_factory": False,
    }
    if with_field:
        unit["can_attack"] = False
    return unit


def _enemy_tank(x: float = ENEMY_BASE_X, *, y: float = 1300.0, with_field: bool = True) -> dict:
    return _tank(x, y=y, entity_id=200, with_field=with_field)


def _shaping(
    own: list[dict],
    enemies: list[dict],
    *,
    actions: list[dict] | None = None,
    combat: bool = True,
    previous_own: list[dict] | None = None,
    previous_enemies: list[dict] | None = None,
):
    """Run one decision, in combat-centroid mode unless ``combat=False``.

    ``previous_own`` / ``previous_enemies`` describe the *previous* decision's
    positions (defaults to the same as this one, i.e. "nothing moved").
    """

    micro = _micro()
    game = RwppGameConfig()
    state = {
        "tick": 30,
        "own_entities": own,
        "visible_enemy_entities": enemies,
        "global_state": {"income": 0.0},
    }
    prev_state = {
        "own_entities": previous_own if previous_own is not None else own,
        "visible_enemy_entities": (
            previous_enemies if previous_enemies is not None else enemies
        ),
    }
    kwargs: dict = {}
    if combat:
        kwargs["combat_distance"] = combat_army_distance_norm(state, DIAGONAL)
        kwargs["previous_combat_distance"] = combat_army_distance_norm(prev_state, DIAGONAL)
    return compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=actions if actions is not None else [],
        receipts=[],
        message={"global_state": {"income": 0.0}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        # Deliberately the *whole-army* distance, so a passing engagement value
        # proves the combat path ignored these two arguments.
        distance=0.5,
        previous_distance=0.5,
        own_alive=len(own),
        enemy_alive=len(enemies),
        previous_own_alive=len(own),
        previous_enemy_alive=len(enemies),
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
        previous_own_buildings=sum(1 for u in own if u.get("is_building")),
        previous_own_combat=sum(
            1 for u in own if not u.get("is_building") and not u.get("is_builder")
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Which units count as combatants
# ---------------------------------------------------------------------------
def test_explicit_can_attack_field_is_authoritative() -> None:
    assert unit_can_attack(_tank(0.0)) is True
    assert unit_can_attack(_builder(0.0)) is False
    # A building is not assumed to shoot, even with no explicit field.
    assert unit_can_attack(_command_center(with_field=False)) is False
    # An explicit ``false`` must be honoured even on something that *looks*
    # combat-capable by class, and ``true`` on something that does not.
    assert unit_can_attack(_tank(0.0, can_attack=False)) is False
    assert unit_can_attack(_builder(0.0, can_attack=True)) is True


def test_missing_can_attack_falls_back_to_the_combat_heuristic() -> None:
    """Old bridge / offline mock: no field, so infer from the entity flags.

    ``can_attack`` is added by a parallel change to ``entityJson()``; until it
    lands (and on the mock engine) the reward must keep working. The conservative
    reading is "not a building and not a builder" -- exactly like
    ``bonus_produce`` counts combat units.
    """

    assert unit_can_attack(_tank(0.0, with_field=False)) is True
    assert unit_can_attack(_builder(0.0, with_field=False)) is False
    assert unit_can_attack(_command_center(with_field=False)) is False
    # Missing field *and* missing flags: not a builder, not a building -> can fight.
    assert unit_can_attack({"x": 0.0, "y": 0.0}) is True


def test_targetable_units_filters_mixed_armies() -> None:
    units = [
        _command_center(),
        _builder(0.0),
        _tank(10.0, entity_id=101),
        _tank(20.0, entity_id=102),
    ]
    assert [u["entity_id"] for u in targetable_units(units)] == [101, 102]


def test_combat_centroid_is_none_without_a_combatant() -> None:
    """``None``, not the whole-army centroid and not ``(0, 0)``."""

    assert combat_centroid([_command_center(), _builder(0.0)]) is None
    assert combat_centroid([]) is None
    # With a combatant present, only the combatants are averaged.
    assert combat_centroid([_command_center(), _builder(0.0), _tank(100.0, entity_id=101)]) == (
        100.0,
        1300.0,
    )


def test_combat_distance_is_none_when_a_side_has_no_combatant() -> None:
    only_builder = {"own_entities": [_command_center(), _builder(0.0)],
                    "visible_enemy_entities": [_enemy_tank()]}
    assert combat_army_distance_norm(only_builder, DIAGONAL) is None

    no_enemy_combatant = {"own_entities": [_tank(0.0)],
                          "visible_enemy_entities": [{"x": 5.0, "y": 5.0, "is_building": True}]}
    assert combat_army_distance_norm(no_enemy_combatant, DIAGONAL) is None


def test_combat_distance_normalises_like_the_army_distance() -> None:
    """Same scale and clamp as ``army_distance_norm``, so they telescope alike."""

    state = {
        "own_entities": [_tank(0.0, y=0.0)],
        "visible_enemy_entities": [_enemy_tank(300.0, y=400.0)],
    }
    # 300/400 triangle -> 500 px over a 3677 px diagonal.
    assert combat_army_distance_norm(state, DIAGONAL) == pytest.approx(500.0 / DIAGONAL)
    # Far apart: clamped to 1.0 exactly like the whole-army helper.
    far = {
        "own_entities": [_tank(0.0, y=0.0)],
        "visible_enemy_entities": [_enemy_tank(9000.0, y=9000.0)],
    }
    assert combat_army_distance_norm(far, DIAGONAL) == 1.0


# ---------------------------------------------------------------------------
# 2. The core assertion: a builder alone moves engagement by exactly nothing
# ---------------------------------------------------------------------------
def test_only_a_builder_moving_toward_the_enemy_pays_no_engagement() -> None:
    """THE core assertion of this change.

    Same state on both decisions except the builder walks 1686 px closer. In the
    old whole-army arithmetic that was ``+0.0688``; it must now be exactly zero,
    because we have no unit that can shoot.
    """

    before = [_command_center(), _builder(BASE_X)]
    after = [_command_center(), _builder(ENEMY_BASE_X)]

    shaping = _shaping(after, [_enemy_tank()], previous_own=before)
    assert shaping.components["engagement"] == 0.0

    # Sanity: the builder really did move, and the *old* path would have paid.
    assert combat_army_distance_norm(
        {"own_entities": before, "visible_enemy_entities": [_enemy_tank()]}, DIAGONAL
    ) is None
    old_shaping = _shaping(after, [_enemy_tank()], previous_own=before, combat=False)
    # Legacy path keeps its historical meaning (distance is the caller's 0.5).
    assert old_shaping.components["engagement"] == pytest.approx(0.3 * (0.5 - 0.5))


def test_builder_motion_cannot_move_a_combat_centroid() -> None:
    """A tank is present, so engagement is real -- and the builder cannot sway it.

    Both scenarios move the builder 1686 px closer; the tank is stationary in
    both, so the combat-centroid gap is unchanged and engagement is exactly 0.0.
    """

    enemies = [_enemy_tank()]
    tank_x = 1200.0
    scenario_a = _shaping(
        [_command_center(), _tank(tank_x), _builder(ENEMY_BASE_X)],
        enemies,
        previous_own=[_command_center(), _tank(tank_x), _builder(BASE_X)],
    )
    scenario_b = _shaping(
        [_command_center(), _tank(tank_x), _builder(BASE_X)],
        enemies,
        previous_own=[_command_center(), _tank(tank_x), _builder(ENEMY_BASE_X)],
    )
    assert scenario_a.components["engagement"] == 0.0
    assert scenario_b.components["engagement"] == 0.0


# ---------------------------------------------------------------------------
# 3. Engagement still works when something can actually fight
# ---------------------------------------------------------------------------
def test_a_tank_closing_on_the_enemy_pays_positive_engagement() -> None:
    tank_before, tank_after = 1600.0, 1200.0
    shaping = _shaping(
        [_tank(tank_after)],
        [_enemy_tank()],
        previous_own=[_tank(tank_before)],
    )
    expected = 0.3 * (tank_before - tank_after) / DIAGONAL
    assert shaping.components["engagement"] > 0.0
    assert shaping.components["engagement"] == pytest.approx(expected)


def test_a_tank_retreating_pays_negative_engagement() -> None:
    shaping = _shaping(
        [_tank(1600.0)],
        [_enemy_tank()],
        previous_own=[_tank(1200.0)],
    )
    assert shaping.components["engagement"] < 0.0


def test_engagement_telescopes_to_the_total_approach() -> None:
    """Summing per-step engagement equals the endpoint difference.

    This is the property that a mixed-centroid implementation would destroy, so
    it is asserted across a multi-step walk rather than one step.
    """

    diag = DIAGONAL
    path = [1600.0, 1400.0, 1200.0, 1000.0]
    total = 0.0
    for previous_x, current_x in zip(path, path[1:]):
        total += _shaping(
            [_tank(current_x)], [_enemy_tank()], previous_own=[_tank(previous_x)]
        ).components["engagement"]
    expected = 0.3 * (path[0] - path[-1]) / diag
    assert total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Idle: only meaningful when we actually have something in the fight
# ---------------------------------------------------------------------------
def _hold() -> list[dict]:
    return [{"kind": "hold", "unit_group": 0}]


def _stop() -> list[dict]:
    return [{"kind": "stop", "unit_group": 0}]


def test_builder_alone_near_the_enemy_holding_is_not_penalised() -> None:
    """Correct survival behaviour must not be billed.

    The builder is parked right next to the enemy base, with the command center
    far away, and we issue ``hold``. There is no combatant on our side, so
    ``idle`` is exactly ``0.0``.
    """

    own = [_command_center(BASE_X), _builder(ENEMY_BASE_X)]
    shaping = _shaping(own, [_enemy_tank()], actions=_hold())
    assert shaping.components["idle"] == 0.0


def test_builder_alone_near_the_enemy_stopping_is_not_penalised() -> None:
    own = [_command_center(BASE_X), _builder(ENEMY_BASE_X)]
    shaping = _shaping(own, [_enemy_tank()], actions=_stop())
    assert shaping.components["idle"] == 0.0


def test_a_tank_in_engagement_range_holding_is_penalised() -> None:
    """A combatant inside the band that holds still is eating damage for free."""

    micro = _micro()
    engaged_radius = micro.attack_range * 3.0  # 390 px
    # Comfortably inside the band, measured combat-centroid to combat-centroid.
    own = [_tank(ENEMY_BASE_X + engaged_radius * 0.5)]
    shaping = _shaping(own, [_enemy_tank()], actions=_hold())
    assert shaping.components["idle"] < 0.0
    assert shaping.components["idle"] == pytest.approx(-RwppGameConfig().idle_penalty)


def test_a_tank_outside_the_band_holding_is_not_penalised() -> None:
    micro = _micro()
    engaged_radius = micro.attack_range * 3.0
    own = [_tank(ENEMY_BASE_X + engaged_radius * 5.0)]
    shaping = _shaping(own, [_enemy_tank()], actions=_hold())
    assert shaping.components["idle"] == 0.0


def test_a_builder_nearby_cannot_drag_a_distant_tank_into_the_idle_band() -> None:
    """The whole-army centroid is exactly what made this fire incorrectly.

    Our tank sits far outside the band; only the builder is next to the enemy.
    The whole-army centroid lands inside the band, so the legacy path penalises
    holding -- the combat centroid does not.
    """

    micro = _micro()
    engaged_radius = micro.attack_range * 3.0
    # The tank is outside the band, but only ~1.8x out: averaging it with a
    # builder that is standing *on* the enemy base pulls the whole-army centroid
    # back inside the band. That is precisely the false positive.
    tank_gap = engaged_radius * 1.8
    far_tank = _tank(ENEMY_BASE_X + tank_gap)
    own = [far_tank, _builder(ENEMY_BASE_X)]

    army_gap_px = (tank_gap + 0.0) / 2  # whole-army centroid vs the enemy tank
    assert army_gap_px <= engaged_radius
    assert tank_gap > engaged_radius  # the tank itself is genuinely outside

    combat_shaping = _shaping(own, [_enemy_tank()], actions=_hold())
    assert combat_shaping.components["idle"] == 0.0

    # Legacy path still does what it always did (2 units -> centroid in band).
    legacy = compute_step_shaping(
        micro=micro,
        game=RwppGameConfig(),
        state={"tick": 30, "own_entities": own,
               "visible_enemy_entities": [_enemy_tank()],
               "global_state": {"income": 0.0}},
        actions=_hold(),
        receipts=[],
        message={"global_state": {"income": 0.0}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=army_gap_px / DIAGONAL,
        previous_distance=army_gap_px / DIAGONAL,
        own_alive=len(own),
        enemy_alive=1,
        previous_own_alive=len(own),
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
        previous_own_buildings=0,
        previous_own_combat=1,
    )
    assert legacy.components["idle"] < 0.0


def test_idle_still_requires_a_stop_or_hold_order() -> None:
    """A tank in the band that *moves* must not be charged ``idle``."""

    micro = _micro()
    own = [_tank(ENEMY_BASE_X + micro.attack_range * 1.5)]
    shaping = _shaping(own, [_enemy_tank()], actions=[{"kind": "move_group"}])
    assert shaping.components["idle"] == 0.0


# ---------------------------------------------------------------------------
# 5. Backward compatibility and the both-or-neither contract
# ---------------------------------------------------------------------------
def test_legacy_callers_keep_the_whole_army_behaviour() -> None:
    """No combat distances supplied -> engagement/idle exactly as before.

    Every pre-existing caller (older tests, offline harnesses) must keep working
    rather than silently reading ``engagement == 0``.
    """

    shaping = compute_step_shaping(
        micro=_micro(),
        game=RwppGameConfig(),
        state={"tick": 30, "own_entities": [_builder(0.0)],
               "visible_enemy_entities": [_enemy_tank()],
               "global_state": {"income": 0.0}},
        actions=[],
        receipts=[],
        message={"global_state": {"income": 0.0}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.4,
        previous_distance=0.5,
        own_alive=1,
        enemy_alive=1,
        previous_own_alive=1,
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
    )
    assert shaping.components["engagement"] == pytest.approx(0.3 * (0.5 - 0.4))


def test_supplying_only_one_combat_distance_is_rejected() -> None:
    """Both endpoints or neither: a mixed pair cannot telescope."""

    base: dict = dict(
        micro=_micro(),
        game=RwppGameConfig(),
        state={"tick": 30, "own_entities": [_tank(0.0)],
               "visible_enemy_entities": [_enemy_tank()],
               "global_state": {"income": 0.0}},
        actions=[],
        receipts=[],
        message={"global_state": {"income": 0.0}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=1,
        enemy_alive=1,
        previous_own_alive=1,
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
    )
    for kwargs in (
        {"combat_distance": 0.4},
        {"previous_combat_distance": 0.4},
    ):
        with pytest.raises(ValueError, match="must be supplied together"):
            compute_step_shaping(**base, **kwargs)


# ---------------------------------------------------------------------------
# 6. Invariants that must survive the change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("combat", [True, False])
def test_component_key_order_is_unchanged(combat: bool) -> None:
    keys = list(
        _shaping([_tank(1200.0)], [_enemy_tank()], combat=combat).components
    )
    assert keys == [
        "potential",
        "engagement",
        "strategy",
        "kills",
        "losses",
        "income",
        "bonus_build",
        "invalid",
        "idle",
        "terminal",
        "timeout",
        "bonus_produce",
        "building_lost",
    ]


def test_reward_is_still_the_plain_component_sum() -> None:
    shaping = _shaping([_tank(1200.0), _builder(100.0)], [_enemy_tank()])
    assert shaping.reward == pytest.approx(sum(shaping.components.values()))


# ---------------------------------------------------------------------------
# 7. Real-env wiring: the caller must feed both endpoints from one definition
# ---------------------------------------------------------------------------
def test_env_tracks_a_separate_combat_distance_without_touching_last_distance() -> None:
    """``last_distance`` keeps its whole-army meaning; a sibling field is added."""

    import inspect

    from ironzero.rwpp.env import RwppDuelEnv

    step_source = inspect.getsource(RwppDuelEnv.step)
    reset_source = inspect.getsource(RwppDuelEnv.reset)

    # The historical whole-army distance must still be computed and stored.
    assert "army_distance_norm(" in step_source
    assert "self.last_distance = distance" in step_source
    assert "self.last_distance = army_distance_norm(" in reset_source
    # The combat-centroid distance is tracked alongside it, and fed to the reward.
    assert "combat_army_distance_norm(" in step_source
    assert "self.last_combat_distance = combat_distance" in step_source
    assert "previous_combat_distance" in step_source
    assert "combat_distance=" in step_source
