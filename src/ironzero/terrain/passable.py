# SPDX-License-Identifier: Apache-2.0
"""Nearest-passable snapping, composed into :class:`~ironzero.terrain.grid.TerrainGrid`.

The mixin only *reads* ``self.width`` / ``self.height`` / ``self.width_px`` /
``self.height_px`` / ``self.tile_width`` / ``self.tile_height`` / ``self.blocked``
and the coordinate helpers declared in ``grid.py``; it adds no instance state, so
it can be mixed into the frozen ``slots`` dataclass without changing its API.
"""

from __future__ import annotations

import math
from typing import Iterable

from .utils import _clamp, _clamp_bounds, _clamp_int

__all__ = ["PassableMixin"]


class PassableMixin:
    """``nearest_passable_world`` and its expanding-ring scan helpers."""

    __slots__ = ()

    def nearest_passable_world(
        self,
        x: float,
        y: float,
        *,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float]:
        """Closest passable point (tile centre) to ``(x, y)``.

        If the requested point is already passable it is returned unchanged, so
        callers keep exact targets on land. ``bounds`` is an optional inclusive
        world-space box ``(min_x, min_y, max_x, max_y)`` used to keep spawn
        points on the correct half of the map. The search is a deterministic
        expanding-ring scan (tie-break: lower y then lower x tile), and falls
        back to the whole map when the bounded region has no passable tile.
        """

        if not (math.isfinite(x) and math.isfinite(y)):
            x = y = 0.0
        box = _clamp_bounds(bounds, self.width_px, self.height_px)
        if box is not None and not self._inside_box(x, y, box):
            point = self._nearest_passable_in_box(x, y, box)
            if point is None:
                point = self._nearest_passable_in_box(x, y, None)
            if point is None:  # fully blocked map: stay put (deterministic fallback)
                return (
                    float(_clamp(x, 0.0, self.width_px - 1.0)),
                    float(_clamp(y, 0.0, self.height_px - 1.0)),
                )
            return point
        if not self.is_blocked_world(x, y):
            return (float(x), float(y))
        point = self._nearest_passable_in_box(x, y, box)
        if point is None and box is not None:
            point = self._nearest_passable_in_box(x, y, None)
        if point is None:  # fully blocked map: stay put (deterministic fallback)
            return (
                float(_clamp(x, 0.0, self.width_px - 1.0)),
                float(_clamp(y, 0.0, self.height_px - 1.0)),
            )
        return point

    def _nearest_passable_in_box(
        self,
        x: float,
        y: float,
        box: tuple[float, float, float, float] | None,
    ) -> tuple[float, float] | None:
        if box is None:
            tx_lo, ty_lo, tx_hi, ty_hi = 0, 0, self.width - 1, self.height - 1
        else:
            min_x, min_y, max_x, max_y = box
            tx_lo = _clamp_int(int(math.floor(min_x / self.tile_width)), 0, self.width - 1)
            ty_lo = _clamp_int(int(math.floor(min_y / self.tile_height)), 0, self.height - 1)
            tx_hi = _clamp_int(int(math.floor(max_x / self.tile_width)), 0, self.width - 1)
            ty_hi = _clamp_int(int(math.floor(max_y / self.tile_height)), 0, self.height - 1)
        if tx_lo > tx_hi or ty_lo > ty_hi:
            return None
        cx = _clamp_int(int(math.floor(x / self.tile_width)), tx_lo, tx_hi)
        cy = _clamp_int(int(math.floor(y / self.tile_height)), ty_lo, ty_hi)
        max_radius = max(self.width, self.height) + 1
        for radius in range(0, max_radius):
            best: tuple[float, int, int] | None = None
            for ty in range(cy - radius, cy + radius + 1):
                if ty < ty_lo or ty > ty_hi:
                    continue
                if ty != cy - radius and ty != cy + radius:
                    xs: Iterable[int] = (cx - radius, cx + radius)
                else:
                    xs = range(cx - radius, cx + radius + 1)
                for tx in xs:
                    if tx < tx_lo or tx > tx_hi:
                        continue
                    if self.blocked[ty * self.width + tx]:
                        continue
                    center_x, center_y = self.tile_center(tx, ty)
                    key = ((center_x - x) ** 2 + (center_y - y) ** 2, ty, tx)
                    if best is None or key < best:
                        best = key
            if best is not None:
                return self.tile_center(best[2], best[1])
        return None

    @staticmethod
    def _inside_box(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
        min_x, min_y, max_x, max_y = box
        return min_x <= x <= max_x and min_y <= y <= max_y
