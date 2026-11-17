# SPDX-License-Identifier: Apache-2.0
"""The commander's plan actually gates the build stages.

``test_macro_command.py`` pins the *contract* (payload, rules, resolution). This
file pins the **effect**: a directive that names an economy goal order changes
what :class:`RwppDuelEnv` issues, and a directive that omits a goal removes it.

Why this needs its own file
---------------------------
Because "the strategic layer owns the macro game" is exactly the claim that used
to be false while every attribute looked fine. The old code read
``self.game.economy_army_target`` directly, so a directive could be parsed,
validated, logged -- and change nothing. Asserting ``env.macro.army_target == 12``
would pass against that code too; only asserting the *dispatched orders* does not.

The stages are exercised directly (``env._issue_*``) against a hand-built state
instead of through a live game: each stage is a pure function of ``env.state``,
``env.micro`` and now ``env.macro``, so no bridge, JVM or socket is needed. The
one thing that *must* be real is the observable consequence, which is the
returned order list.
"""

from __future__ import annotations

import numpy as np
import pytest

from ironzero.micro_config import MicroDuelConfig
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.env import RwppDuelEnv
from ironzero.rwpp.macro import ResolvedMacro, resolve_macro
from ironzero.strategy import MacroPlan, RuleStrategy, StrategicDirective

MAP_PX = 2600.0


def _micro() -> MicroDuelConfig:
    return MicroDuelConfig(map_width=MAP_PX, map_height=MAP_PX, episode_ticks=30)


def _env(strategy: object | None = None, **overrides) -> RwppDuelEnv:
    """An env with no game behind it, for calling stages directly."""

    game = RwppGameConfig(episode_ticks=30, **overrides)
    return RwppDuelEnv(game, port=59999, micro_config=_micro(), strategy=strategy)


def _combat_unit(entity_id: int, x: float = 300.0, y: float = 500.0) -> dict:
    return {
        "entity_id": entity_id,
        "x": x,
        "y": y,
        "hp": 210.0,
        "max_hp": 210.0,
        "can_attack": True,
    }


def _enemy_base() -> dict:
    return {
        "entity_id": 900,
        "x": MAP_PX - 200.0,
        "y": 500.0,
        "hp": 4000.0,
        "max_hp": 4000.0,
        "is_building": True,
        "can_attack": True,
        "unit_type": "commandCenter",
    }


def _set_state(
    env: RwppDuelEnv,
    *,
    army: int = 10,
    funds: float = 4000.0,
    enemy_base: bool = True,
    own_entities: list[dict] | None = None,
    resource_slots: list[dict] | None = None,
) -> None:
    own = own_entities if own_entities is not None else [
        _combat_unit(100 + index) for index in range(army)
    ]
    enemies = [_enemy_base()] if enemy_base else []
    state = {
        "tick": 500,
        "own_entities": own,
        "visible_enemy_entities": enemies,
        "global_state": {"funds": funds, "income": 26.0},
        "resource_slots": resource_slots if resource_slots is not None else [],
    }
    env.state = state
    env.obs_state = dict(state)


def _apply_plan(env: RwppDuelEnv, plan: MacroPlan) -> None:
    """Install ``plan`` the same way a real directive would."""

    env.directive = StrategicDirective(intent="defend", macro=plan)
    env.macro = resolve_macro(env.game, env.directive)


def _kinds(orders: list[dict]) -> list[str]:
    return [str(order.get("kind")) for order in orders]


# ----------------------------------------------------------------------
# 1. The identity case: no directive == the configured behaviour
# ----------------------------------------------------------------------
def test_no_directive_reproduces_the_configured_defaults() -> None:
    env = _env()
    try:
        assert env.macro.stage_names == env.macro.stage_names  # always defined
        assert env.macro.army_target == env.game.economy_army_target
        assert env.macro.army_cap == env.game.economy_army_cap
        assert env.macro.max_extractors == env.game.economy_max_extractors
        assert env.macro.max_factories == env.game.economy_max_factories
        assert env.macro.max_upgrades == env.game.economy_max_upgrades
        assert env.macro.max_turrets == env.game.economy_max_turrets
        assert env.macro.credit_reserve == env.game.economy_credit_reserve
        assert env.macro.engage_distance == env.game.economy_engage_distance
        assert env.macro.from_commander is False
    finally:
        env.close()


def test_the_configured_attack_gate_is_an_army_target_not_a_second_number() -> None:
    """``attack_at_army`` defaults to ``army_target``: one number to reason about."""

    env = _env()
    try:
        assert env.macro.attack_at_army == env.game.economy_army_target
    finally:
        env.close()


# ----------------------------------------------------------------------
# 2. Cancelling a goal removes it from the dispatch
# ----------------------------------------------------------------------
def test_cancelling_attack_removes_the_march_order() -> None:
    """The single most important effect: "do not march" is now expressible."""

    env = _env()
    try:
        _set_state(env, army=10)
        assert _kinds(env._issue_attack_drive()) == ["attack_move"], (
            "the fixture must produce a real march order under the default plan, "
            "otherwise this test proves nothing about the cancellation"
        )

        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "production"]))
        assert env.macro.allows("attack") is False
        assert env._issue_attack_drive() == []
    finally:
        env.close()


def test_cancelling_expansion_removes_the_extractor_order() -> None:
    env = _env()
    try:
        # A finished factory with a combat menu + a free pool + enough credits:
        # exactly the state the expansion stage is designed to fire in.
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
            {"entity_id": 24, "x": 210.0, "y": 220.0, "hp": 170.0, "max_hp": 170.0,
             "is_builder": True},
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0, "income": 26.0},
            "resource_slots": [{"slot_id": 0, "x": 300.0, "y": 300.0, "occupied": False}],
            "factories": [{"entity_id": 50, "can_produce": ["hoverTank"], "queue_length": 0}],
        }
        env.state = state
        env.obs_state = dict(state)

        assert _kinds(env._issue_late_expansion()) == ["build_extractor"]

        _apply_plan(env, MacroPlan(build_order=["upgrade", "extractor", "factory", "production"]))
        assert env.macro.allows("expansion") is False
        assert env._issue_late_expansion() == []
    finally:
        env.close()


def test_cancelling_upgrade_removes_the_tier_order() -> None:
    env = _env()
    try:
        # A finished factory + one living combat unit + an upgradable extractor and
        # a bank above the upgrade floor: the state the upgrade stage fires in.
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
            {
                "entity_id": 60,
                "x": 300.0,
                "y": 300.0,
                "hp": 700.0,
                "max_hp": 700.0,
                "is_building": True,
                "unit_type": "extractorT1",
                "tech_level": 1,
                "can_upgrade": True,
                "build_progress": 1.0,
                "is_operational": True,
            },
            _combat_unit(101),
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0, "income": 26.0},
            "resource_slots": [],
        }
        env.state = state
        env.obs_state = dict(state)

        assert _kinds(env._issue_upgrades()) == ["upgrade_building"]

        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "production"]))
        assert env.macro.allows("upgrade") is False
        assert env._issue_upgrades() == []
    finally:
        env.close()


def test_cancelling_production_stops_queueing_units() -> None:
    env = _env()
    try:
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0, "income": 26.0},
            "resource_slots": [],
            "factories": [{"entity_id": 50, "can_produce": ["hoverTank"], "queue_length": 0}],
        }
        env.state = state
        env.obs_state = dict(state)

        assert _kinds(env._issue_opening_production()) == ["produce_unit"]

        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "upgrade"]))
        assert env.macro.allows("production") is False
        assert env._issue_opening_production() == []
    finally:
        env.close()


# ----------------------------------------------------------------------
# 3. Budgets change the gates
# ----------------------------------------------------------------------
def test_army_cap_controls_when_production_stops() -> None:
    """``army_cap`` is the gate production actually stops at.

    ``army_target`` only marks the stage "done" for logging; the stage keeps
    replacing losses up to the cap, which is the measured fix for the endgame
    stalemates (the AI out-reinforced a factory that stopped at the target). So
    the plan field that changes behaviour here is ``army_cap`` -- and it must.
    """

    env = _env()
    try:
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
            *[_combat_unit(100 + index) for index in range(6)],
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0, "income": 26.0},
            "resource_slots": [],
            "factories": [{"entity_id": 50, "can_produce": ["hoverTank"], "queue_length": 0}],
        }
        env.state = state
        env.obs_state = dict(state)

        # 6 living units against the default cap of 16: keep producing.
        assert _kinds(env._issue_opening_production()) == ["produce_unit"]

        # A commander that caps the army at 6 stops it immediately.
        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "production"],
                                   army_target=6, army_cap=6))
        assert env.macro.army_cap == 6
        assert env._issue_opening_production() == []
    finally:
        env.close()


def test_credit_reserve_controls_the_production_affordability_gate() -> None:
    env = _env()
    try:
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 1000.0, "income": 26.0},
            "resource_slots": [],
            "factories": [{"entity_id": 50, "can_produce": ["hoverTank"], "queue_length": 0}],
        }
        env.state = state
        env.obs_state = dict(state)

        # 1000 credits clears the default 900 reserve: produce.
        assert _kinds(env._issue_opening_production()) == ["produce_unit"]

        # A commander that hoards 1200 blocks it instead.
        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "production"],
                                   credit_reserve=1200.0))
        assert env._issue_opening_production() == []
    finally:
        env.close()


def test_attack_threshold_changes_when_the_march_starts() -> None:
    env = _env()
    try:
        _set_state(env, army=6)
        # The default gate is 10 units: six is not enough to march.
        assert env._issue_attack_drive() == []

        # "We already outnumber them, go" -- the rule layer's own decision.
        _apply_plan(env, MacroPlan(build_order=["extractor", "factory", "production",
                                                "upgrade", "defence", "expansion", "attack"],
                                   attack_at_army=5, army_cap=16))
        assert _kinds(env._issue_attack_drive()) == ["attack_move"]
    finally:
        env.close()


def _enemy_unit(x: float, y: float = 500.0) -> dict:
    """A visible enemy that is *not* a structure.

    Deliberately not a building: the attack stage ignores ``engage_distance``
    while enemy buildings stand (measured: stopping there left the army parked
    next to an intact base for ~9000 ticks), so the distance gate is only
    observable once nothing is left to destroy.
    """

    return {"entity_id": 800, "x": x, "y": y, "hp": 210.0, "max_hp": 210.0}


def test_engage_distance_changes_when_the_march_stops_being_reissued() -> None:
    env = _env()
    try:
        # Own army at x=300, a lone enemy unit at x=2300: a 2000 px gap.
        own = [_combat_unit(100 + index, x=300.0, y=500.0) for index in range(10)]
        _set_state(env, own_entities=own, enemy_base=False)
        env.state["visible_enemy_entities"] = [_enemy_unit(MAP_PX - 300.0)]

        # Default 700 px: outside it, so the march keeps being re-issued.
        assert _kinds(env._issue_attack_drive()) == ["attack_move"]

        # A commander that calls the army "engaged" from 2200 px out stops the
        # re-issue: the policy owns every movement decision from now on.
        _apply_plan(env, MacroPlan(
            build_order=["extractor", "factory", "production",
                         "upgrade", "defence", "expansion", "attack"],
            engage_distance=2200.0,
        ))
        assert env._issue_attack_drive() == []
    finally:
        env.close()


def test_engage_distance_zero_keeps_pressing_all_the_way_in() -> None:
    """``0`` is a legal, meaningful value: "always treat the army as engaged"."""

    env = _env()
    try:
        # Own army at x=1200, enemy unit at x=1400: a 200 px gap.
        own = [_combat_unit(100 + index, x=1200.0, y=500.0) for index in range(10)]
        _set_state(env, own_entities=own, enemy_base=False)
        env.state["visible_enemy_entities"] = [_enemy_unit(1400.0)]

        # Default 700 px and only 200 px to go: the stage yields to the policy.
        assert env._issue_attack_drive() == []

        _apply_plan(env, MacroPlan(
            build_order=["extractor", "factory", "production",
                         "upgrade", "defence", "expansion", "attack"],
            engage_distance=0.0,
        ))
        # ... unless the commander says "keep the order standing".
        assert _kinds(env._issue_attack_drive()) == ["attack_move"]
    finally:
        env.close()


# ----------------------------------------------------------------------
# 4. The stage *order* is the plan's order
# ----------------------------------------------------------------------
def test_stage_order_follows_the_plan() -> None:
    """``build_order`` is a priority, not a set.

    The order is observable even when both stages fire: upgrade addresses a
    *building* and production addresses a *factory*, so neither blocks the other
    and both land in the same dispatch -- in the order the plan walks. That order
    is load-bearing, because the bridge applies "last order wins" within one
    dispatch.

    (Most build stages cannot be reordered against each other at all, and that is
    correct rather than a limitation: the factory stage refuses until an extractor
    is operational, so "factory before extractor" is not a plan the game can
    honour. The plan reorders *execution*, it cannot invent prerequisites.)
    """

    env = _env()
    try:
        own = [
            {
                "entity_id": 50,
                "x": 200.0,
                "y": 200.0,
                "hp": 4000.0,
                "max_hp": 4000.0,
                "is_building": True,
                "is_factory": True,
                "unit_type": "landFactory",
                "build_progress": 1.0,
                "is_operational": True,
            },
            {
                "entity_id": 60,
                "x": 300.0,
                "y": 300.0,
                "hp": 700.0,
                "max_hp": 700.0,
                "is_building": True,
                "unit_type": "extractorT1",
                "tech_level": 1,
                "can_upgrade": True,
                "build_progress": 1.0,
                "is_operational": True,
            },
            _combat_unit(101),
        ]
        state = {
            "tick": 500,
            "own_entities": own,
            "visible_enemy_entities": [],
            "global_state": {"funds": 4000.0, "income": 26.0},
            "resource_slots": [{"slot_id": 0, "x": 900.0, "y": 900.0, "occupied": True}],
            "factories": [{"entity_id": 50, "can_produce": ["hoverTank"], "queue_length": 0}],
        }
        env.state = state
        env.obs_state = dict(state)

        # Default order: upgrade is first, so it heads the dispatch.
        default_orders = res_kinds(env._curriculum_actions())
        assert default_orders[0] == "upgrade_building", default_orders

        # A commander that wants the army first reorders the plan, and the
        # dispatch order follows -- which is what decides the "last order wins"
        # outcome inside one message.
        #
        # (The two cannot both appear in the second run: the production stage
        # returns a blocking order, so the loop stops there. That is the existing,
        # measured dispatch rule and not something the plan changes.)
        _apply_plan(env, MacroPlan(build_order=["production", "upgrade"]))
        reordered = res_kinds(env._curriculum_actions())
        assert reordered[0] == "produce_unit", reordered
        assert env.macro.stage_names == ("production", "upgrade")
    finally:
        env.close()


def res_kinds(orders: list[dict]) -> list[str]:
    return [str(order.get("kind")) for order in orders]


# ----------------------------------------------------------------------
# 5. The rule layer's plan survives the round trip into the executor
# ----------------------------------------------------------------------
def test_rule_layer_plan_reaches_the_executor() -> None:
    """End to end: a battlefield -> the rule directive -> a real stage decision."""

    env = _env()
    try:
        # A broken army: the rule layer's first rule fires.
        own = [_combat_unit(100 + index) for index in range(3)]
        for unit in own:
            unit["hp"] = 10.0
        _set_state(env, own_entities=own, army=3, funds=4000.0)

        directive = RuleStrategy().propose(env._strategy_summary())
        assert directive.macro.is_empty is False
        assert "attack" not in directive.macro.build_order

        env.directive = directive
        env.macro = resolve_macro(env.game, env.directive)

        assert env.macro.from_commander is True
        assert env.macro.allows("attack") is False
        assert env._issue_attack_drive() == []
    finally:
        env.close()


def test_a_broken_strategy_layer_leaves_the_default_plan_in_place() -> None:
    """A strategist that raises must not break the economy, only the shaping."""

    class Exploding:
        def propose(self, _summary):
            raise RuntimeError("boom")

    env = _env(strategy=Exploding())
    try:
        _set_state(env, army=10)
        env._refresh_strategy()
        assert env.directive is None
        assert env.macro.from_commander is False
        assert env.macro.army_target == env.game.economy_army_target
    finally:
        env.close()


def test_resolved_plan_is_visible_in_the_step_info_payload() -> None:
    """What the commander asked for and what ran must both be observable."""

    from ironzero.rwpp.stepping import strategy_payload

    env = _env()
    try:
        _set_state(env, army=10)
        env._refresh_strategy()  # no strategy -> directive cleared
        assert strategy_payload(env) is None

        env.directive = StrategicDirective(
            intent="push",
            macro=MacroPlan(build_order=["production", "defence"]),
        )
        env.macro = resolve_macro(env.game, env.directive)
        env.macro_state = {"funds": 1234.0, "army": 7}
        payload = strategy_payload(env)
        assert payload is not None
        # The raw request ...
        assert payload["macro"]["build_order"] == ["production", "defence"]
        # ... and the resolved plan actually used, next to it.
        assert payload["resolved_macro"]["stage_names"] == ["production", "defence"]
        assert payload["resolved_macro"]["from_commander"] is True
        # ... and the economy facts the commander decided from.
        assert payload["economy"]["funds"] == 1234.0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
