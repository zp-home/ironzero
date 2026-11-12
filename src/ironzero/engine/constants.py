# SPDX-License-Identifier: Apache-2.0
"""Shared constants of the deterministic micro-duel engine.

Order names, action kinds, entity type ids, terrain tuning and the rejection
reason codes that mirror ``ironzero.protocol.ErrorCode``. Values are contractual
(the action ids are positional) and must not change.
"""

from __future__ import annotations

ORDER_NAMES: dict[int, str] = {
    0: "idle",
    1: "move",
    2: "attack_move",
    3: "attack_target",
    4: "hold",
}

ORDER_IDS: dict[str, int] = {name: order for order, name in ORDER_NAMES.items()}

ACTION_KINDS: tuple[str, ...] = (
    "move_group",
    "attack_move",
    "attack_target",
    "stop",
    "hold",
    "build_extractor",
    "build_structure",
    "produce_unit",
    "upgrade_building",
    # ``repair_target`` used to sit here (index 9). It was removed rather than
    # fixed: ``rwpp/decoding.py`` looked the repair target up in
    # ``visible_enemy_entities`` -- you cannot repair an enemy -- and the Java
    # bridge had no repair branch at all, so on the real game the kind could
    # only ever produce a rejected order. Removing it keeps the head honest:
    # every kind the policy can sample is a kind the bridge can execute.
    # ``noop`` issues no order at all. It exists because construction and unit
    # production take hundreds of ticks of *uninterrupted* progress, and every
    # real order cancels the in-flight one. Without a no-op the policy has to
    # issue ``hold``/``stop`` between build orders and the factory is cancelled
    # before it finishes -- measured: a landFactory completes in 191 ticks when
    # the next decisions are no-ops, and never when they are ``hold``.
    # Appended last so the existing head indices (and their weights) stay put.
    "noop",
)

TANK_TYPE_ID = 1
BUILDER_TYPE_ID = 2
EXTRACTOR_TYPE_ID = 10
LAND_FACTORY_TYPE_ID = 11
TURRET_TYPE_ID = 12

# Terrain mode (real map) tuning. Arena mode never touches these values: with
# ``terrain=None`` the engine keeps the exact original code path and numbers.
TERRAIN_SPAWN_MARGIN_RATIO = 0.12  # spawn columns sit at 12% / 88% of map width
TERRAIN_REPATH_TICKS = 20  # at most one A* re-plan per unit per 20 ticks
TERRAIN_STUCK_LIMIT_TICKS = 30  # no progress for 0.5 s -> give up the order
TERRAIN_MAX_PATH_NODES = 20000  # A* expansion cap per query

# Rejection reasons mirror ironzero.protocol.ErrorCode on purpose.
STALE_TICK = "STALE_TICK"
UNKNOWN_UNIT = "UNKNOWN_UNIT"
UNIT_NOT_OWNED = "UNIT_NOT_OWNED"
TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"
POSITION_BLOCKED = "POSITION_BLOCKED"
INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
ACTION_MASKED = "ACTION_MASKED"
INVALID_MESSAGE = "INVALID_MESSAGE"
