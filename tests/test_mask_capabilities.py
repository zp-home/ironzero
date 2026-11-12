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
           "max_hp": 170.0, "x": 100.0, "y": 100.0}


def _state(builders=None, funds: float = 4000.0, enemies: int = 0):
    state = {
        "own_entities": [dict(BUILDER)],
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
    mask = _mask(_state(builders=[{"entity_id": 7, "unit_type": "builder",
                                   "can_build": ["extractor", "landFactory",
                                                 "tank", "turret"]}]))
    selectable = _selectable(mask)
    assert {"build_extractor", "build_structure",
            "produce_unit", "upgrade_building"} <= selectable


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
                                 "stop", "hold", "repair_target", "noop"}


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


def test_mask_never_all_false() -> None:
    """Every kind gated off must still leave the tactical orders selectable."""

    state = _state(builders=[{"entity_id": 7, "unit_type": "builder",
                              "can_build": []}], funds=0.0)
    mask = _mask(state)
    assert mask.any()
    # 5 tactical orders + repair_target (builder present, no capability needed)
    # + noop (always available).
    assert int(np.count_nonzero(mask)) == 7


@pytest.mark.parametrize("kind", ["build_extractor", "build_structure",
                                  "produce_unit", "upgrade_building"])
def test_each_economy_kind_respects_its_capability(kind: str) -> None:
    """Dropping one buildable from the report closes exactly its kind."""

    full = ["extractor", "landFactory", "tank", "turret"]
    with_all = _selectable(_mask(_state(builders=[{"entity_id": 7, "can_build": full}])))
    assert kind in with_all

    necessary = {"build_extractor": "extractor", "upgrade_building": "extractor",
                 "build_structure": "landFactory", "produce_unit": "tank"}[kind]
    reduced = [name for name in full if name != necessary]
    without = _selectable(_mask(_state(builders=[{"entity_id": 7, "can_build": reduced}])))
    assert kind not in without
