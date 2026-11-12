# SPDX-License-Identifier: Apache-2.0
"""经济动作链路：资源池必须真的流到策略手里。

这一组锁住一个会让"建造者跑到对方家里送死"的真实缺陷。三个环节各自独立地坏了，
合起来的表现是：建造者从不造资源提取器，一路走向敌方基地，然后在原地待着。

1. ``observation_to_state`` 把桥接发来的 ``resource_slots`` **整个丢掉**。于是
   :func:`decode_action` 的 ``slots`` 退化成 ``[{}]``，``target_index % 1``
   永远是 0 —— 不管策略选哪个目标，都只会发出 ``slot_id = 0``；编码器也永远
   认为一个提取器都没建。
2. 桥接的槽位排序是 **tile-row major**（y 小的在前）。在南北向地图上，敌方那半边
   排在前面：``[p2]Lake (2p)`` 实测自家基地 (990, 2010)、敌方 (1230, 210)，而
   ``slot 0`` 是 (1130, 390) —— 敌方门口。于是"永远 slot 0"直接把建造者送上死路。
   修复后按"离自家基地远近"排序，``slot 0`` 是家门口那口井。
3. ``target_mask`` 只给可见敌人开槽，策略**无法表达**"去用第 5 口井"，因为资源池
   数量常常多于可见敌人数量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ironzero.micro_config import MicroDuelConfig  # noqa: E402
from ironzero.observation.masks import build_masks  # noqa: E402
from ironzero.rwpp.decoding import decode_action, merge_done_message, observation_to_state  # noqa: E402

# ``[p2]Lake (2p)`` pools, in the order the fixed bridge reports them
# (nearest to our base at (990, 2010) first).
LAKE_SLOTS = [
    {"slot_id": 0, "x": 1170.0, "y": 2050.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 1, "x": 130.0, "y": 1910.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 2, "x": 2370.0, "y": 1890.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 3, "x": 1250.0, "y": 1190.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 4, "x": 230.0, "y": 1190.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 5, "x": 2190.0, "y": 1190.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 6, "x": 150.0, "y": 530.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 7, "x": 2370.0, "y": 510.0, "occupied": False, "extractor_level": 0},
    {"slot_id": 8, "x": 1130.0, "y": 390.0, "occupied": False, "extractor_level": 0},
]

BUILD_EXTRACTOR = 5  # index into ACTION_KINDS


def _obs(slots, *, enemies=None, funds=5000.0):
    return {
        "tick": 100,
        "game_time": 1.7,
        "controlled_team": 0,
        "global_state": {"funds": funds, "income": 1.0},
        "own_entities": [
            {
                "entity_id": 24,
                "entity_type_id": 1,
                "team_id": 0,
                "x": 990.0,
                "y": 2070.0,
                "hp": 170.0,
                "max_hp": 170.0,
                "is_builder": True,
                "group_id": 0,
            }
        ],
        "visible_enemy_entities": enemies or [],
        "resource_slots": slots,
        "info": {"builders": [{"unit_type": "builder", "can_build": ["extractor"]}]},
    }


@pytest.fixture()
def config() -> MicroDuelConfig:
    return MicroDuelConfig()


# ---------------------------------------------------------------------------
# 1. resource_slots must survive the conversion
# ---------------------------------------------------------------------------
def test_observation_keeps_resource_slots() -> None:
    state = observation_to_state(_obs(LAKE_SLOTS))
    assert len(state["resource_slots"]) == len(LAKE_SLOTS), (
        "dropping resource_slots collapses every extractor order onto slot 0"
    )
    assert state["resource_slots"][0]["x"] == 1170.0


def test_missing_resource_slots_degrades_to_empty_list() -> None:
    message = _obs(LAKE_SLOTS)
    message.pop("resource_slots")
    state = observation_to_state(message)
    assert state["resource_slots"] == [], "older bridges omit the key; never KeyError"


def test_done_message_keeps_resource_slots() -> None:
    state = observation_to_state(_obs(LAKE_SLOTS))
    terminal = merge_done_message(
        state, {"type": "done", "resource_slots": [dict(s, occupied=True) for s in LAKE_SLOTS[:2]]}
    )
    assert len(terminal["resource_slots"]) == 2, "terminal observation must keep the pools"


# ---------------------------------------------------------------------------
# 2. the target head must actually select a pool
# ---------------------------------------------------------------------------
def test_target_index_selects_the_named_pool(config) -> None:
    state = observation_to_state(_obs(LAKE_SLOTS))
    seen = []
    for target_index in range(4):
        action = decode_action(
            np.array([BUILD_EXTRACTOR, 4, 0, 0, target_index]), state, config
        )
        seen.append(action[0]["slot_id"])
    assert seen == [0, 1, 2, 3], f"target head is ignored; got {seen}"
    assert len(set(seen)) > 1, "every target index collapsing onto one slot is the bug"


def test_occupied_pools_are_skipped(config) -> None:
    slots = [dict(slot) for slot in LAKE_SLOTS]
    slots[0].update(occupied=True, extractor_level=1)
    slots[1].update(occupied=True, extractor_level=1)
    state = observation_to_state(_obs(slots))

    picks = [
        decode_action(np.array([BUILD_EXTRACTOR, 4, 0, 0, index]), state, config)[0]["slot_id"]
        for index in range(3)
    ]
    assert picks == [2, 3, 4], f"occupied pools must not be re-ordered; got {picks}"


def test_all_pools_occupied_still_emits_a_valid_action(config) -> None:
    slots = [dict(slot, occupied=True, extractor_level=1) for slot in LAKE_SLOTS]
    state = observation_to_state(_obs(slots))
    action = decode_action(np.array([BUILD_EXTRACTOR, 4, 0, 0, 2]), state, config)
    assert action[0]["kind"] == "build_extractor"
    assert 0 <= action[0]["slot_id"] < len(slots), "must stay in range for the bridge"


def test_no_reported_pools_still_emits_slot_zero(config) -> None:
    state = observation_to_state(_obs([]))
    action = decode_action(np.array([BUILD_EXTRACTOR, 4, 0, 0, 7]), state, config)
    assert action[0]["slot_id"] == 0, "legacy bridge: let Java resolve slot 0 itself"


# ---------------------------------------------------------------------------
# 3. the mask must let the policy express "use this pool"
# ---------------------------------------------------------------------------
def test_target_mask_opens_resource_pool_slots(config) -> None:
    state = observation_to_state(_obs(LAKE_SLOTS))
    masks = build_masks(state, config)

    # One visible enemy, so the enemy-only mask would open a single slot.
    assert int(masks["target"].sum()) >= len(LAKE_SLOTS), (
        "with fewer enemies than pools the policy cannot name a pool at all"
    )


def test_target_mask_still_opens_enemy_slots(config) -> None:
    enemies = [
        {"entity_id": 21 + index, "team_id": 1, "x": 1230.0, "y": 210.0, "hp": 4000, "max_hp": 4000}
        for index in range(4)
    ]
    state = observation_to_state(_obs(LAKE_SLOTS, enemies=enemies))
    masks = build_masks(state, config)
    assert all(bool(masks["target"][index]) for index in range(4))


def test_target_mask_always_keeps_the_fallback_slot(config) -> None:
    state = observation_to_state(_obs([]))
    masks = build_masks(state, config)
    assert bool(masks["target"][-1]), "the 'no target' slot keeps sampling valid"


# ---------------------------------------------------------------------------
# 4. the bridge orders pools nearest-to-own-base first
# ---------------------------------------------------------------------------
def test_bridge_orders_pools_nearest_to_our_base_first() -> None:
    """Tile-row order put the enemy's half first and sent builders to their death."""

    source = (
        REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
    ).read_text(encoding="utf-8")

    assert "ownBaseAnchor()" in source, (
        "slot 0 must be the pool nearest our base, not the smallest tile row"
    )
    # The anchor must prefer the command centre over an arbitrary unit.
    assert "isFactoryBuilding()" in source
    # And the sort must be distance-based rather than row-major.
    anchor_block = source[source.index("private java.util.List<double[]> resourceSlots()") :]
    anchor_block = anchor_block[: anchor_block.index("private static double distanceSquared")]
    assert "distanceSquared" in anchor_block, "resourceSlots() must sort by distance to base"
