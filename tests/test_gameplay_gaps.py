# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the four reported gameplay gaps.

The user reported, against a policy trained through the web platform on the
original ``[p2]Lake (2p)`` map:

  1. it never upgrades buildings;
  2. it never builds machine-gun / defence structures;
  3. it only builds one kind of tank, and only one factory;
  4. it never builds a shipyard (or any other structure).

All four turned out to be the **same class of defect**: the decision did not exist
in the action space. ``decode_action`` hard-coded ``structure_type="land_factory"``
for every ``build_structure`` sample and ``unit_type = combat[0]`` for every
``produce_unit`` sample, so "which building" and "which unit" were constants. The
targets were further shut on the mask side, and two curriculum stages that would
have exercised the options never fired (a starved priority slot and an attribute
name typo).

These tests are written against the *observable contract* rather than the
implementation, so they keep holding if the internals move again:
  * the decoder must be able to emit more than one structure type;
  * the decoder must be able to emit more than one producible unit;
  * the mask must open the target slots those menus are indexed by;
  * the two-factory site latch must not reuse the first factory's coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from ironzero.mock_engine import ACTION_KINDS
from ironzero.observation.masks import build_masks
from ironzero.observation.menus import (
    build_menu,
    is_defence_type,
    is_pool_only_type,
    menu_index,
    produce_menu,
    produce_options,
    upgradable_buildings,
)
from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.decoding import decode_action

GROUP = 4


def _micro():
    return RwppGameConfig(
        map_name="[p2]Lake (2p)",
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_width_px=2600.0,
        map_height_px=2600.0,
        units_per_team=8,
        max_entities=8,
        target_slots=9,
    ).micro_config()


def _lake_state() -> dict:
    """A payload shaped like the bridge's, with the *real* Lake menus.

    The menu contents are the measured ones (``.runtime/q1_build_menu.json`` and
    ``.runtime/q3_factory_menu_after.json``): the builder offers seven structure
    types, and a finished ``landFactory`` offers four combat units once the bridge
    reports them.
    """

    return {
        "tick": 900,
        "global_state": {"funds": 4200.0, "income": 30.0},
        "own_entities": [
            {"entity_id": 23, "unit_type": "commandCenter", "x": 990.0, "y": 2010.0,
             "hp": 4000.0, "max_hp": 4000.0, "is_building": True, "is_factory": True,
             "is_factory_building": True, "can_upgrade": False, "tech_level": 1,
             "is_operational": True, "build_progress": 1.0},
            {"entity_id": 24, "unit_type": "builder", "x": 1080.0, "y": 2050.0,
             "hp": 170.0, "max_hp": 170.0, "is_builder": True, "can_upgrade": False,
             "tech_level": 1, "is_operational": True, "build_progress": 1.0},
            {"entity_id": 40, "unit_type": "extractorT1", "x": 1130.0, "y": 2050.0,
             "hp": 800.0, "max_hp": 800.0, "is_building": True, "can_upgrade": True,
             "tech_level": 1, "is_operational": True, "build_progress": 1.0},
            {"entity_id": 55, "unit_type": "landFactory", "x": 1120.0, "y": 2250.0,
             "hp": 1200.0, "max_hp": 1200.0, "is_building": True, "is_factory": True,
             "is_factory_building": True, "can_upgrade": True, "tech_level": 1,
             "is_operational": True, "build_progress": 1.0},
        ],
        "visible_enemy_entities": [
            {"entity_id": 90, "unit_type": "tank", "x": 1230.0, "y": 300.0,
             "hp": 210.0, "max_hp": 210.0, "can_attack": True},
        ],
        "resource_slots": [
            {"slot_id": 0, "x": 1130.0, "y": 2050.0, "occupied": True,
             "extractor_level": 1},
            {"slot_id": 1, "x": 1200.0, "y": 2400.0, "occupied": False,
             "extractor_level": 0},
        ],
        "builders": [
            {"entity_id": 24, "unit_type": "builder",
             "can_build": ["landFactory", "airFactory", "seaFactory",
                           "laserDefence", "repairbay", "turret", "extractor"]},
        ],
        "factories": [
            {"entity_id": 23, "unit_type": "commandCenter",
             "can_produce": ["builder"], "queued": [], "queue_length": 0,
             "queue_progress": 0.0},
            {"entity_id": 55, "unit_type": "landFactory",
             "can_produce": ["builder", "hoverTank", "tank", "artillery", "scout"],
             "queued": [], "queue_length": 0, "queue_progress": 0.0},
        ],
        "receipts": [],
    }


def _decode(kind: str, target: int, state: dict | None = None):
    micro = _micro()
    action = np.array(
        [ACTION_KINDS.index(kind), GROUP, 3, 5, target], dtype=np.int64
    )
    return decode_action(action, state or _lake_state(), micro)


# ---------------------------------------------------------------------------
# 1. Menus: the engine's answer, filtered and stably ordered
# ---------------------------------------------------------------------------
def test_build_menu_excludes_pool_only_types() -> None:
    """``build_structure`` must not offer an extractor.

    The engine accepts ``structure_type="extractor"`` anywhere and then queues a
    placement that can never finish (measured in ``.runtime/buildnames.json``:
    ``accepted: true`` on dry land). Offering it through ``build_structure`` would
    pay exploration value for a structure that cannot exist; ``build_extractor``
    is the kind that resolves a real pool coordinate.
    """

    menu = build_menu(_lake_state())
    assert "extractor" not in menu
    assert is_pool_only_type("extractor")
    assert is_pool_only_type("extractorT2")
    assert not is_pool_only_type("turret")
    # The other six survive, in the engine's own order.
    assert menu == [
        "landFactory", "airFactory", "seaFactory",
        "laserDefence", "repairbay", "turret",
    ]


def test_produce_menu_drops_builder() -> None:
    """Every factory can make builders; they are not a combat choice."""

    menu = produce_menu(_lake_state())
    assert "builder" not in menu
    assert menu == ["hoverTank", "tank", "artillery", "scout"]


def test_produce_options_is_the_factory_times_unit_product() -> None:
    """``(factory_id, unit)`` pairs, so "which factory" stays a decision too."""

    options = produce_options(_lake_state())
    assert (55, "tank") in options
    assert (55, "artillery") in options
    # The command centre contributes nothing: its only entry is a builder.
    assert all(factory == 55 for factory, _ in options)
    assert len(options) == 4


def test_upgradable_buildings_separates_reported_from_silent() -> None:
    """``(list, reported)`` -- the pair the mask and decoder both depend on."""

    upgradable, reported = upgradable_buildings(_lake_state())
    assert reported is True
    assert {u["entity_id"] for u in upgradable} == {40, 55}

    silent = _lake_state()
    for unit in silent["own_entities"]:
        unit.pop("can_upgrade", None)
    upgradable, reported = upgradable_buildings(silent)
    assert reported is False
    assert upgradable == []


def test_menu_index_folds_protocol_and_engine_spellings() -> None:
    """``land_factory`` must find ``landFactory`` and ``turret`` must find ``turretT1``."""

    menu = ["landFactory", "turretT1", "seaFactory"]
    assert menu_index(menu, "land_factory") == 0
    assert menu_index(menu, "landFactory") == 0
    assert menu_index(menu, "turret") == 1
    assert menu_index(menu, "sea_factory") == 2
    # Unknown names degrade to the first entry, which is the old hard-coded
    # behaviour -- never to the last (``-1 % n``) and never to an index error.
    assert menu_index(menu, "nonsense") == 0
    assert menu_index([], "turret") == 0


def test_is_defence_type_recognises_turrets_but_not_repair_bays() -> None:
    assert is_defence_type("turret")
    assert is_defence_type("turretT1")
    # A repair bay heals rather than shoots, so it is not a defensive structure.
    assert not is_defence_type("repairbay")
    assert not is_defence_type("landFactory")
    # ``laserDefence`` is the trap this predicate exists to avoid. Its name says
    # "defence" and the engine reports ``is_defence: true`` for it, but it is a
    # projectile INTERCEPTOR with ``can_attack: false`` -- it cannot shoot at
    # anything, so building it instead of a gun turret costs 500 credits and adds no
    # firepower. Measured in ``.runtime/q2_laser_defence.json``.
    assert not is_defence_type("laserDefence")
    assert not is_defence_type("laser_defence")


# ---------------------------------------------------------------------------
# 2. Gap 1+2: the decoder can express more than one structure
# ---------------------------------------------------------------------------
def test_build_structure_can_name_a_turret() -> None:
    """Gap 2: "never builds machine-gun / defence structures".

    Before the fix every ``build_structure`` decoded to ``land_factory`` regardless
    of the target head, so a turret was not merely unlikely -- it was unexpressible.
    """

    micro = _micro()
    types = []
    for target in range(micro.target_slots):
        orders = _decode("build_structure", target)
        assert len(orders) == 1
        types.append(orders[0]["structure_type"])
    assert "turret" in types, types
    assert "laserDefence" in types, types
    assert len(set(types)) > 1, "the structure type must be a decision, not a constant"


def test_build_structure_can_name_every_offered_structure() -> None:
    """The head must reach the *whole* menu, not just its first entry."""

    micro = _micro()
    types = {
        _decode("build_structure", target)[0]["structure_type"]
        for target in range(micro.target_slots)
    }
    assert set(build_menu(_lake_state())) <= types


def test_build_structure_falls_back_for_payloads_without_a_menu() -> None:
    """An older bridge must keep the historical behaviour, not lose building."""

    state = _lake_state()
    state["builders"] = []
    orders = _decode("build_structure", 0, state)
    assert orders[0]["structure_type"] == "land_factory"


# ---------------------------------------------------------------------------
# 3. Gap 3: the decoder can express more than one unit
# ---------------------------------------------------------------------------
def test_produce_unit_can_name_every_combat_unit() -> None:
    """Gap 3: "only builds one kind of tank".

    The old decoder ended with ``unit_type = combat[0]``: a constant. The target
    head must select across the whole menu instead.
    """

    micro = _micro()
    units = []
    for target in range(micro.target_slots):
        orders = _decode("produce_unit", target)
        assert len(orders) == 1
        assert orders[0]["kind"] == "produce_unit"
        units.append(orders[0]["unit_type"])
    assert set(units) == set(produce_menu(_lake_state())), units
    assert len(set(units)) > 1


def test_produce_unit_never_orders_a_builder() -> None:
    """Builders are the cheapest menu entry and would farm ``bonus_produce``."""

    micro = _micro()
    for target in range(micro.target_slots):
        for order in _decode("produce_unit", target):
            assert order["unit_type"].lower() != "builder"


# ---------------------------------------------------------------------------
# 4. Gap 1: the upgrade target is selectable
# ---------------------------------------------------------------------------
def test_upgrade_building_can_target_each_upgradable_building() -> None:
    """Gap 1: "never upgrades".

    The kind was reachable but the *choice of building* is what the target head
    carries, and it must map onto the same list the mask sizes itself from.
    """

    micro = _micro()
    ids = {_decode("upgrade_building", target)[0]["building_entity_id"]
           for target in range(micro.target_slots)}
    assert ids == {40, 55}, ids


def test_upgrade_building_dispatch_nothing_when_engine_says_no() -> None:
    """A reported-but-empty list must produce no order, not a doomed one.

    Dispatching anyway would pay ``invalid_penalty`` for obeying information the
    bridge already provided.
    """

    state = _lake_state()
    for unit in state["own_entities"]:
        if "can_upgrade" in unit:
            unit["can_upgrade"] = False
    assert _decode("upgrade_building", 0, state) == []


# ---------------------------------------------------------------------------
# 5. The mask opens the slots the menus are indexed by
# ---------------------------------------------------------------------------
def test_target_mask_opens_menu_slots() -> None:
    """A menu the decoder can address is useless if the mask keeps it shut.

    This is the contract that ties the two layers together: every economy-menu
    slot the decoder indexes must be open whenever that menu is non-empty.
    """

    state = _lake_state()
    masks = build_masks(state, _micro())
    widest = max(
        len(produce_options(state)),
        len(upgradable_buildings(state)[0]),
        len(build_menu(state)),
    )
    assert widest > 0
    for slot in range(widest):
        assert masks["target"][slot], f"target slot {slot} must be open"
    # The "no target" fallback stays open so sampling is always valid.
    assert masks["target"][-1]


def test_build_structure_stays_selectable_outside_a_construction() -> None:
    """Gap 2's *mask* half.

    ``curriculum_owns_build_kinds`` used to shut ``build_structure`` for the whole
    episode, which made turrets unexpressible even after the decoder learned to name
    them. It must be open when nothing is under construction, and shut only while
    something is -- because that is the only window where a duplicate cancels work.
    """

    state = _lake_state()
    masks = build_masks(state, _micro())
    assert masks["kind"][ACTION_KINDS.index("build_structure")]

    # Now leave a structure half-built: the duplicate hazard is real again.
    state["own_entities"][3]["is_operational"] = False
    state["own_entities"][3]["build_progress"] = 0.4
    guarded = build_masks(state, _micro())
    assert not guarded["kind"][ACTION_KINDS.index("build_structure")]


def test_build_extractor_stays_owned_by_the_curriculum() -> None:
    """Its duplicate is catastrophic (the extractor never finishes), so it stays shut.

    The tests above prove ``build_structure`` was *narrowed* rather than closed. This
    one protects the asymmetry: ``build_extractor`` keeps the old treatment, because a
    re-issued extractor order pins ``build_progress`` at 0.0 forever while every
    receipt reports success (``.runtime/ab_reissue.py``), and every downstream stage is
    gated on that extractor finishing.
    """

    micro = _micro()
    # ``RwppGameConfig.micro_config`` enables it whenever the opening curriculum is
    # on, which is the default.
    assert micro.curriculum_owns_build_kinds is True
    owned = build_masks(_lake_state(), micro)
    assert not owned["kind"][ACTION_KINDS.index("build_extractor")]
    # It is genuinely owned by the curriculum, not broken: turning the ownership off
    # must make it selectable again.
    from dataclasses import replace

    free = replace(micro, curriculum_owns_build_kinds=False)
    assert build_masks(_lake_state(), free)["kind"][ACTION_KINDS.index("build_extractor")]


# ---------------------------------------------------------------------------
# 6. Gap 3's other half: two factories must not share a site
# ---------------------------------------------------------------------------
def test_second_factory_site_avoids_the_first(monkeypatch) -> None:
    """Gap 3: "only one factory".

    ``_issue_opening_factory`` latches a site once per episode. Two separate
    defects kept the second factory from existing, and both are covered here:

    * the reset wrote ``self._factory_site`` while the latch lives in
      ``self.factory_site``, so the site was never cleared and the second factory
      was ordered onto the first one's coordinates;
    * the site search took "the grid cell nearest the builder", and the builder
      stands at the first factory, so the nearest cell *was* the occupied one.
    """

    from ironzero.rwpp.env import RwppDuelEnv

    env = RwppDuelEnv.__new__(RwppDuelEnv)
    env.game = RwppGameConfig(
        map_name="[p2]Lake (2p)",
        map_relative="maps/skirmish/[p2]Lake (2p).tmx",
        map_width_px=2600.0,
        map_height_px=2600.0,
        units_per_team=8,
        max_entities=8,
        target_slots=9,
        economy_max_factories=2,
    )
    env.micro = env.game.micro_config()
    # Every curriculum stage reads ``self.macro``; a partially-built env must carry
    # the resolved deployment defaults or the stage raises AttributeError.
    from ironzero.rwpp.macro import resolve_macro

    env.macro = resolve_macro(env.game, None)
    env.factory_site = None
    env.factory_curriculum_done = True
    env._factory_progress = None
    env._factory_stalled = 0
    # The order budget this stage keeps per episode. A fresh instance has it at 0;
    # the test builds the env via ``__new__`` so it must be set explicitly.
    env._factory_orders = 0
    env._curriculum_disabled = False
    env.curriculum_order = None

    first_x, first_y = 1120.0, 2250.0
    state = {
        "global_state": {"funds": 6000.0},
        "own_entities": [
            {"entity_id": 23, "unit_type": "commandCenter", "x": 990.0, "y": 2010.0,
             "is_building": True, "is_factory": True, "is_factory_building": True,
             "is_operational": True, "build_progress": 1.0, "hp": 4000.0},
            {"entity_id": 24, "unit_type": "builder", "x": first_x, "y": first_y,
             "is_builder": True, "hp": 170.0, "is_operational": True,
             "build_progress": 1.0},
            {"entity_id": 25, "unit_type": "extractorT1", "x": 1130.0, "y": 2050.0,
             "is_building": True, "is_operational": True, "build_progress": 1.0,
             "hp": 800.0},
            {"entity_id": 55, "unit_type": "landFactory", "x": first_x, "y": first_y,
             "is_building": True, "is_factory": True, "is_factory_building": True,
             "is_operational": True, "build_progress": 1.0, "hp": 1200.0},
        ],
        "visible_enemy_entities": [],
        "resource_slots": [],
        "builders": [{"entity_id": 24, "can_build": ["landFactory"]}],
        "factories": [],
    }
    env.state = state
    env.obs_state = state

    orders = env._issue_opening_factory()
    assert orders, "the second factory must be ordered"
    order = orders[0]
    assert order["kind"] == "build_structure"
    # The latched grid cell must resolve to a point that is NOT the first factory.
    gx, gy = env.factory_site
    wx, wy = env.micro.grid_to_world(gx, gy)
    assert (wx - first_x) ** 2 + (wy - first_y) ** 2 > 260.0 ** 2, (
        f"second factory site {wx:.0f},{wy:.0f} overlaps the first at "
        f"{first_x:.0f},{first_y:.0f}"
    )


def test_factory_site_attribute_is_the_one_that_is_latched() -> None:
    """The typo guard: there must be no competing ``_factory_site`` attribute."""

    from ironzero.rwpp.env import RwppDuelEnv

    env = RwppDuelEnv.__new__(RwppDuelEnv)
    env.factory_site = None
    assert not hasattr(env, "_factory_site")


# ---------------------------------------------------------------------------
# 7. Reward: the two new terms exist and are one-off
# ---------------------------------------------------------------------------
def test_upgrade_and_defence_bonuses_are_configurable() -> None:
    game = RwppGameConfig()
    assert game.upgrade_bonus > 0.0
    assert game.defence_bonus > 0.0
    # Both must stay well under a decided game, or the agent farms structures
    # instead of winning. ``win_reward`` is 1.0 in the micro config.
    assert game.upgrade_bonus < 1.0
    assert game.defence_bonus < 1.0


@pytest.mark.parametrize("flag", ["economy_upgrade", "economy_defence"])
def test_new_curriculum_stages_are_on_by_default(flag: str) -> None:
    assert getattr(RwppGameConfig(), flag) is True


# ---------------------------------------------------------------------------
# 8. ``bonus_upgrade`` must pay for upgrades, not for production
# ---------------------------------------------------------------------------
def _tier_state(units: list[dict]) -> dict:
    return {"own_entities": units, "visible_enemy_entities": [], "global_state": {}}


def test_tier_total_counts_buildings_only_not_units() -> None:
    """A produced unit must never look like an upgraded building.

    Every unit reports ``tech_level`` 1 -- a scout is a T1 scout -- so summing
    ``tech_level`` over all entities made "made a scout" indistinguishable from
    "upgraded an extractor". Measured on the two-slot layout at difficulty 1, the
    unfiltered sum paid ``bonus_upgrade`` on **609 of 609 decisions** while no
    building ever left T1; the policy spent 15 straight updates massing scouts and
    never built an extractor (``.runtime/tier_reward_audit.json``).
    """

    from ironzero.rwpp.reward_shaping import _building_tier_total

    units = [
        {"unit_type": "commandCenter", "is_building": True, "tech_level": 1},
        {"unit_type": "builder", "is_building": False, "tech_level": 1},
        {"unit_type": "scout", "is_building": False, "tech_level": 1},
        {"unit_type": "scout", "is_building": False, "tech_level": 1},
    ]
    assert _building_tier_total(_tier_state(units)) == 0

    # An upgraded extractor is the only thing that can move the number.
    units.append({"unit_type": "extractorT1", "is_building": True, "tech_level": 2})
    assert _building_tier_total(_tier_state(units)) == 1


def test_tier_total_ignores_the_starting_tier() -> None:
    """A building begins at T1, so T1 is not an achievement."""

    from ironzero.rwpp.reward_shaping import _building_tier_total

    units = [
        {"unit_type": "commandCenter", "is_building": True, "tech_level": 1},
        {"unit_type": "landFactory", "is_building": True, "tech_level": 1},
    ]
    assert _building_tier_total(_tier_state(units)) == 0

    units[1]["tech_level"] = 3
    # T3 counts as two tiers gained, not three.
    assert _building_tier_total(_tier_state(units)) == 2


def test_tier_total_counts_nothing_when_the_bridge_omits_the_field() -> None:
    """An older bridge must leave the term identically zero, not crash."""

    from ironzero.rwpp.reward_shaping import _building_tier_total

    assert _building_tier_total(_tier_state([])) == 0
    assert _building_tier_total({}) == 0
    assert (
        _building_tier_total(
            _tier_state([{"unit_type": "extractorT1", "is_building": True}])
        )
        == 0
    )
    assert (
        _building_tier_total(
            _tier_state([{"unit_type": "extractorT1", "is_building": True,
                          "tech_level": "garbage"}])
        )
        == 0
    )


def test_env_and_reward_shaping_agree_on_the_tier_total() -> None:
    """The two implementations must not drift.

    ``rwpp.env`` keeps its own snapshot for ``previous_own_tier_total`` and
    ``reward_shaping`` recomputes the current one; if they disagree the difference
    is paid every decision, which is exactly the 609/609 failure. The duplicate
    exists only to avoid an import cycle, so a test is what holds them together.
    """

    from ironzero.rwpp.env import _building_tier_total as env_total
    from ironzero.rwpp.reward_shaping import _building_tier_total as shaping_total

    states = [
        _tier_state([]),
        _tier_state([{"unit_type": "scout", "is_building": False, "tech_level": 1}]),
        _tier_state([{"unit_type": "commandCenter", "is_building": True, "tech_level": 1}]),
        _tier_state([
            {"unit_type": "commandCenter", "is_building": True, "tech_level": 1},
            {"unit_type": "extractorT2", "is_building": True, "tech_level": 3},
            {"unit_type": "scout", "is_building": False, "tech_level": 2},
        ]),
    ]
    for state in states:
        assert env_total(state.get("own_entities")) == shaping_total(state)
