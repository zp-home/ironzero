# SPDX-License-Identifier: Apache-2.0
"""Deterministic micro-duel engine, split into focused mixins.

The public surface is unchanged from the historical ``ironzero.mock_engine``
module: :class:`MicroDuelEngine` composes spawn / action / movement / combat /
economy / opponent / observation / reward mixins, and :class:`MicroDuelAdapter`
wraps it for the env and the TCP bridge.
"""

from __future__ import annotations

from .actions import ActionMixin
from .adapter import MicroDuelAdapter
from .combat import CombatMixin
from .config import make_terrain_config
from .constants import (
    ACTION_KINDS,
    ACTION_MASKED,
    BUILDER_TYPE_ID,
    EXTRACTOR_TYPE_ID,
    INSUFFICIENT_FUNDS,
    INVALID_MESSAGE,
    LAND_FACTORY_TYPE_ID,
    ORDER_IDS,
    ORDER_NAMES,
    POSITION_BLOCKED,
    STALE_TICK,
    TANK_TYPE_ID,
    TARGET_NOT_VISIBLE,
    TERRAIN_MAX_PATH_NODES,
    TERRAIN_REPATH_TICKS,
    TERRAIN_SPAWN_MARGIN_RATIO,
    TERRAIN_STUCK_LIMIT_TICKS,
    TURRET_TYPE_ID,
    UNIT_NOT_OWNED,
    UNKNOWN_UNIT,
)
from .economy import EconomyMixin
from .engine import MicroDuelEngine
from .entities import MockResourceSlot, Unit
from .movement import MovementMixin
from .observation import ObservationMixin
from .opponents import OpponentMixin
from .reward import RewardMixin
from .spawn import SpawnMixin
from .util import _clamp

__all__ = [
    # constants
    "ORDER_NAMES",
    "ORDER_IDS",
    "ACTION_KINDS",
    "TANK_TYPE_ID",
    "BUILDER_TYPE_ID",
    "EXTRACTOR_TYPE_ID",
    "LAND_FACTORY_TYPE_ID",
    "TURRET_TYPE_ID",
    "TERRAIN_SPAWN_MARGIN_RATIO",
    "TERRAIN_REPATH_TICKS",
    "TERRAIN_STUCK_LIMIT_TICKS",
    "TERRAIN_MAX_PATH_NODES",
    "STALE_TICK",
    "UNKNOWN_UNIT",
    "UNIT_NOT_OWNED",
    "TARGET_NOT_VISIBLE",
    "POSITION_BLOCKED",
    "INSUFFICIENT_FUNDS",
    "ACTION_MASKED",
    "INVALID_MESSAGE",
    # entities and config
    "MockResourceSlot",
    "Unit",
    "make_terrain_config",
    # engine / adapter
    "MicroDuelEngine",
    "MicroDuelAdapter",
    # mixins (exposed for typing / introspection)
    "SpawnMixin",
    "ActionMixin",
    "MovementMixin",
    "CombatMixin",
    "EconomyMixin",
    "OpponentMixin",
    "ObservationMixin",
    "RewardMixin",
    # helpers
    "_clamp",
]
