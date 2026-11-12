# SPDX-License-Identifier: Apache-2.0
"""Terrain-aware configuration rescaling for real maps."""

from __future__ import annotations

import math
from dataclasses import replace

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..terrain import TerrainGrid
from .constants import TERRAIN_SPAWN_MARGIN_RATIO


def make_terrain_config(
    terrain: TerrainGrid,
    config: MicroDuelConfig | None = None,
) -> MicroDuelConfig:
    """Return ``config`` rescaled to a real map's pixel dimensions.

    The engine, the env encoder and the recorder all share this helper so the
    observation normalisation, action grid projection and spawn logic agree on
    the terrain map size. Arena configs are never modified.
    """

    base = config or DEFAULT_CONFIG
    margin = round(terrain.width_px * TERRAIN_SPAWN_MARGIN_RATIO, 3)
    # Real maps are much larger than the 1000 px arena: give the episode enough
    # time for both sides to cross the map (2x the diagonal travel time).
    diagonal = math.hypot(terrain.width_px, terrain.height_px)
    travel_seconds = diagonal / max(1.0, base.unit_speed)
    episode_ticks = max(base.episode_ticks, int(travel_seconds * 2.0 * base.tick_rate))
    return replace(
        base,
        map_width=float(terrain.width_px),
        map_height=float(terrain.height_px),
        spawn_margin=margin,
        episode_ticks=episode_ticks,
    )
