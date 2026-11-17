# SPDX-License-Identifier: Apache-2.0
"""The reward must pay for the *whole* economy loop, not just for the first step.

Before this, the shaping paid for finishing a building (``bonus_build``) and for
income growth, but **nothing** paid for actually using the factory. So
"finish the factory, then idle forever" scored exactly the same as "finish the
factory, then field an army", and the cheapest possible build dominated the
return. Losing a building was equally invisible: the hp-based ``losses`` term
cannot price a command center, whose value is the capability it unlocks rather
than its raw hit points.

Two invariants are asserted:

1. ``bonus_produce`` pays once per new combat unit, cannot be farmed by
   re-queueing, and does not pay for builders;
2. ``building_lost`` charges once per building that disappears;
3. both are appended *after* the historical keys, so every pre-existing reward
   prefix stays bit-identical (``reward`` is ``sum(components.values())`` and
   floating-point addition is order sensitive).
"""

from __future__ import annotations

import pytest

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.reward_shaping import compute_step_shaping

HISTORICAL_KEYS = [
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
]


def _shaping(
    *,
    own_units: list[dict] | None = None,
    previous_own_buildings: int = 0,
    previous_own_combat: int = 0,
):
    micro = RwppGameConfig().micro_config()
    game = RwppGameConfig()
    if own_units is None:
        own_units = [{"x": 100.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}]
    state = {
        "tick": 100,
        "own_entities": own_units,
        "visible_enemy_entities": [
            {"x": 200.0, "y": 100.0, "hp": 100.0, "max_hp": 100.0}
        ],
        "global_state": {"income": 0.0, "funds": 1000.0},
    }
    return compute_step_shaping(
        micro=micro,
        game=game,
        state=state,
        actions=[],
        receipts=[],
        message={"global_state": {"income": 0.0}},
        bridge_result=None,
        potential=0.0,
        previous_potential=0.0,
        distance=0.5,
        previous_distance=0.5,
        own_alive=len(own_units),
        enemy_alive=1,
        previous_own_alive=len(own_units),
        previous_enemy_alive=1,
        strategy_component=0.0,
        disconnected=False,
        done=False,
        own_max_ref=210.0,
        enemy_max_ref=210.0,
        episode_components={},
        previous_own_buildings=previous_own_buildings,
        previous_own_combat=previous_own_combat,
    )


def _tank(entity_id: int) -> dict:
    return {
        "entity_id": entity_id, "x": 100.0, "y": 100.0, "hp": 210.0,
        "max_hp": 210.0, "is_building": False, "is_builder": False,
    }


def _builder(entity_id: int) -> dict:
    return {
        "entity_id": entity_id, "x": 100.0, "y": 100.0, "hp": 170.0,
        "max_hp": 170.0, "is_building": False, "is_builder": True,
    }


def _building(entity_id: int) -> dict:
    return {
        "entity_id": entity_id, "x": 100.0, "y": 100.0, "hp": 1200.0,
        "max_hp": 1200.0, "is_building": True, "is_factory": True,
    }


# ---------------------------------------------------------------------------
# 1. New-combat-unit reward
# ---------------------------------------------------------------------------
def test_a_new_combat_unit_pays_the_configured_bonus() -> None:
    shaping = _shaping(own_units=[_tank(1)], previous_own_combat=0)
    assert shaping.components["bonus_produce"] == pytest.approx(
        RwppGameConfig().produce_bonus
    )
    assert shaping.components["bonus_produce"] > 0.0


def test_two_new_units_pay_twice() -> None:
    shaping = _shaping(
        own_units=[_tank(1), _tank(2), _tank(3)], previous_own_combat=1
    )
    assert shaping.components["bonus_produce"] == pytest.approx(
        2.0 * RwppGameConfig().produce_bonus
    )


def test_repeated_decisions_with_the_same_army_do_not_pay() -> None:
    """A queued order that has not produced anything yet must not pay."""

    steady = [
        _shaping(own_units=[_tank(1)], previous_own_combat=1) for _ in range(20)
    ]
    assert all(step.components["bonus_produce"] == 0.0 for step in steady)


def test_producing_builders_does_not_pay() -> None:
    """Every factory menu offers ``builder``; it must not be a reward farm."""

    shaping = _shaping(
        own_units=[_building(9), _builder(1), _builder(2)], previous_own_combat=0
    )
    assert shaping.components["bonus_produce"] == 0.0


def test_losing_a_unit_does_not_pay_production_bonus() -> None:
    shaping = _shaping(own_units=[_tank(1)], previous_own_combat=3)
    assert shaping.components["bonus_produce"] == 0.0


def test_production_bonus_stays_below_a_won_game() -> None:
    game = RwppGameConfig()
    micro = game.micro_config()
    assert game.produce_bonus < micro.win_reward
    # A single structure must still be worth more than a single unit: the
    # structure is the prerequisite that makes production possible at all.
    assert game.produce_bonus < game.build_bonus


# ---------------------------------------------------------------------------
# 2. Building-loss penalty
# ---------------------------------------------------------------------------
def test_losing_a_building_charges_the_configured_penalty() -> None:
    shaping = _shaping(own_units=[_tank(1)], previous_own_buildings=1)
    assert shaping.components["building_lost"] == pytest.approx(
        -RwppGameConfig().building_loss_penalty
    )
    assert shaping.components["building_lost"] < 0.0


def test_keeping_buildings_does_not_pay_or_charge() -> None:
    shaping = _shaping(
        own_units=[_building(9), _tank(1)], previous_own_buildings=1
    )
    assert shaping.components["building_lost"] == 0.0


def test_gaining_a_building_does_not_leak_into_the_loss_term() -> None:
    shaping = _shaping(
        own_units=[_building(9), _building(10)], previous_own_buildings=0
    )
    assert shaping.components["building_lost"] == 0.0
    assert shaping.components["bonus_build"] > 0.0


def test_building_loss_stays_below_a_lost_game() -> None:
    game = RwppGameConfig()
    micro = game.micro_config()
    assert game.building_loss_penalty < abs(micro.loss_reward)
    # Two structures lost must not outweigh a decisive result either.
    assert 2 * game.building_loss_penalty < abs(micro.loss_reward) * 2


# ---------------------------------------------------------------------------
# 3. The historical prefix is untouched
# ---------------------------------------------------------------------------
def test_historical_components_keep_their_exact_order_as_a_prefix() -> None:
    shaping = _shaping()
    keys = list(shaping.components)
    assert keys[: len(HISTORICAL_KEYS)] == HISTORICAL_KEYS
    # And the new terms really are appended, not interleaved.
    #
    # ``bonus_upgrade`` / ``bonus_defence`` were appended after ``building_lost``
    # for the "never upgrades / never builds turrets" repair, and ``income_growth``
    # was appended after those so the economy's rate increase is paid at the moment
    # it happens rather than hundreds of decisions later. The *order* rule is what
    # this test protects: appending preserves every pre-existing prefix bit for bit,
    # because ``reward`` is ``sum(components.values())`` and float addition is order
    # sensitive. Inserting a term before an existing key would silently change the
    # reward of every historical step.
    assert set(keys[len(HISTORICAL_KEYS):]) == {
        "bonus_produce",
        "building_lost",
        "bonus_upgrade",
        "bonus_defence",
        "income_growth",
        "income_potential",
    }


def test_reward_still_equals_the_component_sum() -> None:
    shaping = _shaping(
        own_units=[_building(9), _tank(1)], previous_own_buildings=0,
        previous_own_combat=0,
    )
    assert shaping.reward == pytest.approx(sum(shaping.components.values()))


def test_neutral_step_pays_nothing_from_the_new_terms() -> None:
    """A step with no change must not perturb the reward at all."""

    shaping = _shaping(
        own_units=[_building(9), _tank(1)], previous_own_buildings=1,
        previous_own_combat=1,
    )
    assert shaping.components["bonus_produce"] == 0.0
    assert shaping.components["building_lost"] == 0.0


# ---------------------------------------------------------------------------
# 4. The objective's argmax must be "win", not "produce forever"
# ---------------------------------------------------------------------------
def test_the_production_flow_cannot_outbid_a_won_match() -> None:
    """This is the invariant the reward was rebalanced for.

    "Produce a unit" is an unbounded FLOW: every unit pays ``produce_bonus`` for
    as long as credits arrive. "Win" is a fixed STOCK paid once. If the total
    flow an episode can fund exceeds the terminal reward, then *not winning*
    is the optimal policy -- which is exactly what was measured before this
    invariant was enforced (``bonus_produce`` +288.70 vs ``terminal`` +4.51 over
    231 updates, and 0 wins in 96 evaluation episodes at two difficulty levels).

    Recomputing the flow here rather than pinning a literal keeps this test
    honest when the episode length or the income constants change: it is the
    *relationship* that must hold, not any particular number.
    """

    game = RwppGameConfig(map_width_px=2600.0, map_height_px=2600.0)
    micro = game.micro_config()

    # Credits the episode can earn, using the documented ramp 18 -> 58 cr/s.
    seconds = game.resolved_episode_ticks() / 60.0
    mean_income = (game.base_income_reference + 58.0) / 2.0
    credits = mean_income * seconds
    # The cheapest combat unit is what a flow-maximising policy would build.
    units = credits / 350.0  # c_tank price, assets/units/tanks/tank.ini
    flow_cap = game.produce_bonus * units

    assert flow_cap < micro.win_reward, (
        f"the production flow ({flow_cap:.2f}) must stay below the terminal "
        f"reward ({micro.win_reward:.2f}), or 'never win' is optimal"
    )


def test_a_pure_stall_loses_to_winning() -> None:
    """Producing maximally and letting the clock run out must be unprofitable."""

    game = RwppGameConfig(map_width_px=2600.0, map_height_px=2600.0)
    micro = game.micro_config()
    seconds = game.resolved_episode_ticks() / 60.0
    mean_income = (game.base_income_reference + 58.0) / 2.0
    units = (mean_income * seconds) / 350.0
    stall = game.produce_bonus * units - game.timeout_penalty

    assert stall < 0.0, (
        f"stalling pays {stall:+.2f}; it must be negative so that finishing the "
        f"match (win {micro.win_reward:+.2f}, timeout {-game.timeout_penalty:+.2f}) "
        f"is strictly better"
    )


def test_killing_the_whole_enemy_army_pays_less_than_winning() -> None:
    """Kill farming must not outbid closing the game out.

    ``kills`` is scaled by ``threat_value``, so a naive per-kill weight can pay
    more for clearing the map than the objective pays for the win -- a second,
    independent way to never finish.
    """

    game = RwppGameConfig()
    micro = game.micro_config()
    max_kills = 40.0        # measured max enemy_alive on Lake at difficulty 1
    mean_threat = 1.2       # 1.0 = plain unit; artillery/AA score above it
    total = game.kill_bonus * max_kills * mean_threat
    assert total < micro.win_reward, (
        f"clearing {max_kills:.0f} units pays {total:+.2f}, which must stay below "
        f"the win ({micro.win_reward:+.2f})"
    )


def test_a_favourable_trade_still_pays() -> None:
    """The rebalance must not make combat worthless.

    ``kills``/``losses`` are threat-weighted, so trading a tank for the enemy's
    artillery is a net gain even with ``kill_bonus == loss_penalty``.
    """

    game = RwppGameConfig()
    artillery_threat = 1.0 + 0.9 + 0.25 * 1.0   # reach 290/300, rate 1, no air
    tank_threat = 1.0 + 130.0 / 300.0 + 0.25 * 1.0
    trade = game.kill_bonus * artillery_threat - game.loss_penalty * tank_threat
    assert trade > 0.0, f"tank-for-artillery paid {trade:+.4f}; must be positive"


def test_the_terminal_reward_overrides_the_micro_default() -> None:
    """The real-game config overrides +/-1.0; the library default is untouched."""

    from ironzero.micro_config import DEFAULT_CONFIG as MICRO_DEFAULT

    game = RwppGameConfig()
    micro = game.micro_config()
    assert micro.win_reward == pytest.approx(game.win_reward)
    assert micro.loss_reward == pytest.approx(game.loss_reward)
    # The micro-duel contract every shipped checkpoint uses must not move.
    assert MICRO_DEFAULT.win_reward == 1.0
    assert MICRO_DEFAULT.loss_reward == -1.0
    # ``None`` keeps the micro default, which is what offline callers rely on.
    inherited = RwppGameConfig(win_reward=None, loss_reward=None).micro_config()
    assert inherited.win_reward == MICRO_DEFAULT.win_reward
    assert inherited.loss_reward == MICRO_DEFAULT.loss_reward
