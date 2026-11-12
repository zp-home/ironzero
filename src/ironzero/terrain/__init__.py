# SPDX-License-Identifier: Apache-2.0
"""Terrain grid + deterministic A* path finding for real Rusted Warfare maps.

The platform backend parses a real ``.tmx`` map (see ``web/backend/maps.py``)
and writes a compact ``map.json`` next to each inference job:

    {
      "id": "<map id>", "name": "[p2]Lake (2p)",
      "width": 130, "height": 130,
      "tile_width": 20, "tile_height": 20,
      "width_px": 2600, "height_px": 2600,
      "blocked": [0, 0, 1, ...]      # width*height, row-major, 1 = water/blocked
    }

:class:`TerrainGrid` turns that payload into an immutable collision structure
with world (pixel) coordinate queries, and provides a deterministic
4-neighbour A* that returns world-space waypoints. Everything here is standard
library only and fully deterministic: the same grid + query always yields the
same path, which the engine relies on for reproducible episodes.

Package layout:

============================  ====================================================
``grid.py``                    :class:`TerrainGrid`: payload/file parsing,
                               coordinate helpers, collision queries.
``passable.py``                ``PassableMixin``: ``nearest_passable_world`` and
                               the expanding-ring shoreline scan.
``pathing.py``                 ``PathingMixin``: ``find_path`` (A*),
                               ``_compress_waypoints``, ``path_length``.
``utils.py``                   numeric guards / clamping helpers (no state).
============================  ====================================================
"""

from __future__ import annotations

from .grid import TerrainGrid
from .passable import PassableMixin
from .pathing import DEFAULT_MAX_NODES, PathingMixin
from .utils import _clamp, _clamp_bounds, _clamp_int, _positive_int

__all__ = [
    "DEFAULT_MAX_NODES",
    "PassableMixin",
    "PathingMixin",
    "TerrainGrid",
    "_clamp",
    "_clamp_bounds",
    "_clamp_int",
    "_positive_int",
]
