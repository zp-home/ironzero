# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: the implementation now lives in ``ironzero.engine``.

This module is a pure re-export so every historical import path keeps working
(``from ironzero.mock_engine import MicroDuelEngine`` and friends). Do not add
implementation logic here -- put it in the matching ``ironzero.engine`` module.
"""

from __future__ import annotations

# Re-exported stdlib names the old module also exposed, purely for backward
# compatibility of ``ironzero.mock_engine.<name>`` attribute access.
import math  # noqa: F401
import random  # noqa: F401
from dataclasses import dataclass, field, replace  # noqa: F401
from typing import Any, Iterable, Sequence  # noqa: F401

from .engine import *  # noqa: F401,F403
from .engine import (  # noqa: F401
    ACTION_KINDS,
    ACTION_MASKED,
    ActionMixin,
    BUILDER_TYPE_ID,
    CombatMixin,
    EconomyMixin,
    EXTRACTOR_TYPE_ID,
    INSUFFICIENT_FUNDS,
    INVALID_MESSAGE,
    LAND_FACTORY_TYPE_ID,
    MicroDuelAdapter,
    MicroDuelEngine,
    MockResourceSlot,
    MovementMixin,
    ObservationMixin,
    OpponentMixin,
    ORDER_IDS,
    ORDER_NAMES,
    POSITION_BLOCKED,
    RewardMixin,
    SpawnMixin,
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
    Unit,
    _clamp,
    make_terrain_config,
)
from .engine import __all__ as _engine_all
from .micro_config import DEFAULT_CONFIG, MicroDuelConfig  # noqa: F401
from .terrain import TerrainGrid  # noqa: F401

# ``*`` on a module without ``__all__`` would skip underscore names, so the
# explicit list in ``ironzero.engine`` is the authoritative one; mirror it here.
__all__ = list(_engine_all) + [
    "DEFAULT_CONFIG",
    "MicroDuelConfig",
    "TerrainGrid",
    "Unit",
    "MockResourceSlot",
    "make_terrain_config",
    "math",
    "random",
    "dataclass",
    "field",
    "replace",
    "Any",
    "Iterable",
    "Sequence",
]

# Drop duplicates while keeping first-seen order.
__all__ = list(dict.fromkeys(__all__))
del _engine_all
