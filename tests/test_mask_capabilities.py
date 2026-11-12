# SPDX-License-Identifier: Apache-2.0
"""The action mask must agree with what the game says the builder can queue.

Regression guard for a real training failure: on stock skirmish maps the
built-in ``builder`` reports ``can_build: ["landFactory"]`` only, but the mask
let the policy pick ``build_extractor`` anyway. Every such attempt came back
rejected and paid ``invalid_penalty``, so 105 PPO updates spent the whole
reward budget on "do not press the forbidden button" and never learned to fight.

The invariant: an economy kind is selectable only if the engine reports the
matching buildable. When the bridge does not report capabilities at all
(offline mock engine, older bridge), fall back to the funding/builder gate so
the economy is not silently disabled everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.mock_engine import ACTION_KINDS, BUILDER_TYPE_ID
from ironzero.observations import build_masks

KIND_INDEX = {kind: index for index, kind in enumerate(ACTION_KINDS)}

BUILDER = {"entity_id": 7, "entity_type_id": BUILDER_TYPE_ID, "hp": 170.0,
           "max_hp": 170.0, "x": 100.0, "y": 100.0, "is_builder": True}


def _state(builders=None, funds: float = 4000.0, enemies: int = 0,
           buildings: int = 0, upgradeable: bool = False):
    own = [dict(BUILDER)]
    for index in range(buildings):
        building = {
            "entity_id": 40 + index, "entity_type_id": 10,
            "hp": 600.0, "max_hp": 600.0, "x": 200.0, "y": 200.0,
            "is_building": True,
        }
        if upgradeable:
            building["can_upgrade"] = True
        own.append(building)
    state = {
        "own_entities": own,
        "visible_enemy_entities": [
            {"entity_id": 100 + i, "hp": 210.0, "max_hp": 210.0, "x": 500.0, "y": 500.0}
            for i in range(enemies)
        ],
        "global_state": {"funds": funds},
    }
    if builders is not None:
        state["builders"] = builders
    return state


def _mask(state):
    return build_masks(state, DEFAULT_CONFIG)["kind"]


def _selectable(mask) -> set[str]:
    return {kind for kind, index in KIND_INDEX.items() if bool(mask[index])}


def test_extractor_masked_when_engine_only_offers_factory() -> None:
    """The exact stock-map case: builder can only build landFactory."""

    mask = _mask(_state(builders=[{"entity_id": 7, "unit_type": "builder",
                                   "can_build": ["landFactory"]}]))
    selectable = _selectable(mask)
    assert "build_structure" in selectable
    assert "build_extractor" not in selectable
    assert "upgrade_building" not in selectable
    assert "produce_unit" not in selectable


def test_all_economy_kinds_open_when_engine_offers_everything() -> None:
    """With a builder, an upgradable building and money, everything is open."""

    mask = _mask(_state(builders=[{"entity_id": 7, "unit_type": "builder",
                                   "can_build": ["extractor", "landFactory",
                                                 "tank", "turret"]}],
                        buildings=1, upgradeable=True))
    selectable = _selectable(mask)
    assert {"build_extractor", "build_structure",
            "produce_unit", "upgrade_building"} <= selectable


def test_upgrade_needs_an_upgradable_building_not_just_a_builder() -> None:
    """``upgrade_building`` upgrades a *building*; a builder alone is not enough.

    The kind used to be gated on the builder's ``can_build`` list containing
    ``extractor``, which is the wrong question: an upgrade targets a building
    that already exists. Gating it that way both masked a legal order on a map
    where extractors are not yet buildable and left the kind selectable with
    nothing to upgrade.
    """

    no_building = _selectable(_mask(_state(
        builders=[{"entity_id": 7, "can_build": ["extractor"]}])))
    assert "upgrade_building" not in no_building

    with_building = _selectable(_mask(_state(
        builders=[{"entity_id": 7, "can_build": ["extractor"]}],
        buildings=1, upgradeable=True)))
    assert "upgrade_building" in with_building


def test_unreported_capabilities_fall_back_to_funding_gate() -> None:
    """No ``builders`` key at all (mock engine) must not disable the economy."""

    mask = _mask(_state())
    selectable = _selectable(mask)
    assert "build_extractor" in selectable
    assert "build_structure" in selectable


def test_empty_can_build_masks_every_economy_kind() -> None:
    """A builder that reports nothing is a real answer, not missing data."""

    mask = _mask(_state(builders=[{"entity_id": 7, "unit_type": "builder",
                                   "can_build": []}]))
    assert _selectable(mask) == {"move_group", "attack_move", "attack_target",
                                 "stop", "hold", "noop"}


def test_affordability_still_gates_even_when_buildable() -> None:
    mask = _mask(_state(builders=[{"entity_id": 7, "unit_type": "builder",
                                   "can_build": ["extractor", "landFactory"]}],
                        funds=0.0))
    selectable = _selectable(mask)
    assert "build_extractor" not in selectable
    assert "build_structure" not in selectable
    # Tactical orders are never gated on money.
    assert {"attack_move", "move_group"} <= selectable


def test_no_builder_masks_every_economy_kind() -> None:
    state = _state(builders=[{"entity_id": 7, "unit_type": "builder",
                              "can_build": ["extractor"]}])
    # entity_type_id 1 (tank) and no is_builder: nothing here can queue builds.
    state["own_entities"] = [{"entity_id": 1, "entity_type_id": 1, "hp": 210.0,
                              "max_hp": 210.0}]
    selectable = _selectable(_mask(state))
    assert "build_extractor" not in selectable
    assert "build_structure" not in selectable


def test_a_lone_command_center_is_not_a_builder() -> None:
    """Regression: ``entity_type_id == 2`` used to mean "builder" in the mask.

    The bridge reported every *building* as ``entity_type_id = 2``, so a base
    with no builder on the map made all four economy kinds selectable. The
    policy then ordered ``build_extractor``, the engine rejected it, and the
    step paid ``invalid_penalty`` -- measured as the dominant negative term over
    a 105-update run. Only ``is_builder`` may open the economy.
    """

    state = _state(builders=[{"entity_id": 7, "unit_type": "builder",
                              "can_build": ["extractor", "landFactory",
                                            "tank", "turret"]}])
    state["own_entities"] = [
        {"entity_id": 23, "entity_type_id": 2, "hp": 4000.0, "max_hp": 4000.0,
         "is_building": True, "is_factory": True}
    ]
    selectable = _selectable(_mask(state))
    assert selectable == {"move_group", "attack_move", "attack_target",
                          "stop", "hold", "noop"}


def test_mask_never_all_false() -> None:
    """Every kind gated off must still leave the tactical orders selectable."""

    state = _state(builders=[{"entity_id": 7, "unit_type": "builder",
                              "can_build": []}], funds=0.0)
    mask = _mask(state)
    assert mask.any()
    # 5 tactical orders + noop (always available). ``repair_target`` used to add
    # a seventh slot; it is gone because the bridge never implemented it.
    assert int(np.count_nonzero(mask)) == 6


@pytest.mark.parametrize("kind", ["build_extractor", "build_structure",
                                  "produce_unit", "upgrade_building"])
def test_each_economy_kind_respects_its_capability(kind: str) -> None:
    """Dropping one buildable from the report closes exactly its kind."""

    full = ["extractor", "landFactory", "tank", "turret"]

    def selectable_for(can_build: list[str]) -> set[str]:
        # An upgradable building is present in every case: ``upgrade_building``
        # is gated on that, not on the builder's menu, so without it the kind
        # would be masked for a reason unrelated to the parametrised one.
        return _selectable(_mask(_state(
            builders=[{"entity_id": 7, "can_build": can_build}],
            buildings=1, upgradeable=True)))

    assert kind in selectable_for(full)

    # ``upgrade_building`` no longer depends on the build menu (it targets a
    # building that already exists), so there is no buildable to remove for it.
    if kind == "upgrade_building":
        return

    necessary = {"build_extractor": "extractor",
                 "build_structure": "landFactory", "produce_unit": "tank"}[kind]
    reduced = [name for name in full if name != necessary]
    assert kind not in selectable_for(reduced)
