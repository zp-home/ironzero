# SPDX-License-Identifier: Apache-2.0
"""The static terrain passability channel of the observation grid.

Channel 3 of the coarse control grid is a downsampled land/water prior read from
the scenario's ``terrain_path``. The dimension of the observation vector never
depends on whether the file exists: when the terrain is missing or unparsable the
channel simply stays zero and a ``RuntimeWarning`` is emitted.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..terrain import TerrainGrid

# This module lives at ``src/ironzero/observation/terrain_channel.py``; the
# repository root is three levels up. It is only used to resolve a relative
# ``terrain_path`` that does not exist relative to the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "REPO_ROOT",
    "_downsample_passable",
    "_resolve_terrain_path",
    "_terrain_passable_channel",
    "terrain_passable_channel",
]


def terrain_passable_channel(config: MicroDuelConfig = DEFAULT_CONFIG) -> np.ndarray | None:
    """Downsampled terrain passability prior ``(map_grid, map_grid)`` in ``[0, 1]``.

    Returns ``None`` when the config has no terrain channel or the terrain file
    cannot be read (callers then leave the channel zeroed; the observation
    dimension is unaffected because it was fixed when the config was built).
    """

    if not config.terrain_channel:
        return None
    path = _resolve_terrain_path(config.terrain_path)
    if path is None:
        warnings.warn(
            f"terrain file not found: {config.terrain_path!r}; the terrain observation "
            "channel stays zero",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    channel = _terrain_passable_channel(str(path), int(config.map_grid))
    if channel is None:
        warnings.warn(
            f"terrain file could not be parsed: {path}; the terrain observation "
            "channel stays zero",
            RuntimeWarning,
            stacklevel=2,
        )
    return channel


def _resolve_terrain_path(path: str | None) -> Path | None:
    """Resolve a terrain path against cwd first, then the repository root."""

    if not path:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        fallback = (REPO_ROOT / candidate).resolve()
        if fallback.is_file():
            return fallback
    return None


@lru_cache(maxsize=16)
def _terrain_passable_channel(path: str, resolution: int) -> np.ndarray | None:
    try:
        grid = TerrainGrid.from_file(path)
    except (OSError, ValueError):
        return None
    return _downsample_passable(grid, resolution)


def _downsample_passable(grid: TerrainGrid, resolution: int) -> np.ndarray:
    """Average the blocked mask into a ``resolution x resolution`` passable grid.

    Each output cell covers the same *fraction* of the map regardless of the
    terrain tile size, so this works for any map resolution.
    """

    resolution = max(1, int(resolution))
    out = np.ones((resolution, resolution), dtype=np.float32)
    if grid.width <= 0 or grid.height <= 0:
        return out
    for cell_y in range(resolution):
        y0 = grid.height_px * cell_y / resolution
        y1 = grid.height_px * (cell_y + 1) / resolution
        ty0 = min(grid.height - 1, max(0, int(math.floor(y0 / grid.tile_height))))
        ty1 = min(grid.height - 1, max(ty0, int(math.ceil(y1 / grid.tile_height)) - 1))
        for cell_x in range(resolution):
            x0 = grid.width_px * cell_x / resolution
            x1 = grid.width_px * (cell_x + 1) / resolution
            tx0 = min(grid.width - 1, max(0, int(math.floor(x0 / grid.tile_width))))
            tx1 = min(grid.width - 1, max(tx0, int(math.ceil(x1 / grid.tile_width)) - 1))
            total = 0
            passable = 0
            for ty in range(ty0, ty1 + 1):
                base = ty * grid.width
                for tx in range(tx0, tx1 + 1):
                    total += 1
                    if not grid.blocked[base + tx]:
                        passable += 1
            out[cell_y, cell_x] = (passable / total) if total else 1.0
    return out
