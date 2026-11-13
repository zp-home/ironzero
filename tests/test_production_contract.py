# SPDX-License-Identifier: Apache-2.0
"""``produce_unit`` must reach a factory that can actually make a combat unit.

Four independent defects made "build a factory, then make tanks" impossible on a
real map. Each one alone was enough to keep the agent from ever fielding a
combat unit, and each looked like "training is bad" rather than a bug:

1. the mask gated production on the **builder's** menu, which on every stock
   skirmish map is ``['landFactory']`` -- it never contains a unit type;
2. the decoder hard-coded ``unit_type = "tank"``, and a Lake ``landFactory``
   offers ``['builder', 'hoverTank']`` -- no ``tank`` at all;
3. the mask required the literal name ``"tank"`` in the factory's menu, so even
   after (1) and (2) the kind stayed masked;
4. the decoder took ``target_index`` raw over the factory list, whose first
   entry is the command center (it makes only builders), so production was
   spent on builders.

Measured end-to-end on the real game after the fixes (Lake, difficulty 0)::

    reset          own = {commandCenter: 1, builder: 1}
    build factory  own = {commandCenter: 1, builder: 1, landFactory: 1}
    produce_unit   own = {..., hoverTank: 1}      <- first combat unit ever built
"""

from __future__ import annotations

import numpy as np

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.mock_engine import ACTION_KINDS
from ironzero.observations import build_masks
from ironzero.rwpp.decoding import decode_action

KIND = {kind: index for index, kind in enumerate(ACTION_KINDS)}

BUILDER = {
    "entity_id": 7, "entity_type_id": 2, "hp": 170.0, "max_hp": 170.0,
    "x": 100.0, "y": 100.0, "is_builder": True,
}
COMMAND_CENTER = {
    "entity_id": 23, "entity_type_id": 11, "hp": 4000.0, "max_hp": 4000.0,
    "x": 200.0, "y": 200.0, "is_building": True, "is_factory": True,
}
LAND_FACTORY = {
    "entity_id": 25, "entity_type_id": 11, "hp": 1200.0, "max_hp": 1200.0,
    "x": 300.0, "y": 300.0, "is_building": True, "is_factory": True,
}


def _lake_state(*, with_factory: bool = True, with_builder: bool = True) -> dict:
    """The exact shape a Lake start streams: no ``tank`` anywhere in a menu."""

    own = [dict(COMMAND_CENTER)]
    if with_builder:
        own.append(dict(BUILDER))
    if with_factory:
        own.append(dict(LAND_FACTORY))
    state = {
        "own_entities": own,
        "visible_enemy_entities": [],
        "global_state": {"funds": 4000.0},
        # The builder menu really is factory-only on a stock map.
        "builders": [{"entity_id": 7, "unit_type": "builder",
                      "can_build": ["landFactory"]}],
        # The real menus, measured on Lake.
        "factories": [
            {"entity_id": 23, "unit_type": "commandCenter",
             "can_produce": ["builder"]},
            {"entity_id": 25, "unit_type": "landFactory",
             "can_produce": ["builder", "hoverTank"]},
        ],
    }
    return state


def _action(kind: str, target: int = 0) -> np.ndarray:
    return np.array([KIND[kind], DEFAULT_CONFIG.groups, 3, 3, target], dtype=np.int64)


def test_mask_unlocks_production_on_a_factory_without_a_tank_in_its_menu() -> None:
    """No ``tank`` exists on Lake, yet production must be selectable."""

    masks = build_masks(_lake_state(), DEFAULT_CONFIG)
    assert bool(masks["kind"][KIND["produce_unit"]])


def test_mask_still_blocks_production_without_a_factory() -> None:
    """Only buildings count: a builder and a plain building are not factories."""

    state = _lake_state(with_factory=False)
    state["factories"] = []
    state["own_entities"] = [dict(BUILDER)]
    for unit in state["own_entities"]:
        unit.pop("is_factory", None)
    masks = build_masks(state, DEFAULT_CONFIG)
    assert not bool(masks["kind"][KIND["produce_unit"]])


def test_decoder_never_picks_a_builder_to_produce() -> None:
    """The command center makes only builders; a factory must win."""

    action = decode_action(_action("produce_unit", target=0), _lake_state(), DEFAULT_CONFIG)
    assert action[0]["kind"] == "produce_unit"
    assert action[0]["unit_type"] == "hoverTank"
    # And it names the factory, not the command center.
    assert action[0]["factory_entity_id"] == 25


def test_decoder_produces_the_real_combat_unit_name() -> None:
    """``tank`` is not universal: the decoder must use what the engine reports."""

    action = decode_action(_action("produce_unit", target=1), _lake_state(), DEFAULT_CONFIG)
    assert action[0]["unit_type"] not in {"tank", "builder"}
    assert action[0]["unit_type"] == "hoverTank"


def test_decoder_falls_back_to_the_only_menu_when_it_is_builders_only() -> None:
    """A Lake start has no factory: the command center can still make a builder."""

    state = _lake_state(with_factory=False)
    state["factories"] = [
        {"entity_id": 23, "unit_type": "commandCenter", "can_produce": ["builder"]}
    ]
    action = decode_action(_action("produce_unit"), state, DEFAULT_CONFIG)
    assert action[0]["kind"] == "produce_unit"
    assert action[0]["unit_type"] == "builder"


def test_decoder_holds_when_the_factory_menu_is_empty() -> None:
    """An explicit empty menu is a real answer: do not invent an order."""

    state = _lake_state()
    state["factories"] = [
        {"entity_id": 25, "unit_type": "landFactory", "can_produce": []}
    ]
    action = decode_action(_action("produce_unit"), state, DEFAULT_CONFIG)
    assert action[0]["kind"] == "hold"


def test_decoder_falls_back_to_tank_when_capabilities_are_unreported() -> None:
    """Offline mock engine / older bridge: keep the historical behaviour."""

    state = _lake_state()
    state.pop("factories")
    action = decode_action(_action("produce_unit"), state, DEFAULT_CONFIG)
    assert action[0]["kind"] == "produce_unit"
    assert action[0]["factory_entity_id"] in {23, 25}
    assert action[0]["unit_type"] == "tank"
