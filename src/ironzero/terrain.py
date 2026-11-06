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
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Hard cap on A* node expansions. Lake-sized maps (130x130 = 16900 tiles) stay
# well below this; the cap only protects against pathological / giant maps.
DEFAULT_MAX_NODES = 30000

__all__ = ["DEFAULT_MAX_NODES", "TerrainGrid"]


@dataclass(frozen=True, slots=True)
class TerrainGrid:
    """Immutable blocked-tile mask with world<->tile coordinate helpers."""

    map_id: str
    name: str
    width: int  # tiles
    height: int  # tiles
    tile_width: int
    tile_height: int
    width_px: int
    height_px: int
    blocked: tuple[bool, ...]  # row-major, length width*height

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TerrainGrid":
        """Build a grid from the ``map.json`` payload written by the backend."""

        if isinstance(payload, TerrainGrid):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("terrain payload must be a mapping")
        width = _positive_int(payload.get("width"), "width")
        height = _positive_int(payload.get("height"), "height")
        tile_width = _positive_int(payload.get("tile_width", 20), "tile_width")
        tile_height = _positive_int(payload.get("tile_height", 20), "tile_height")
        width_px = _positive_int(payload.get("width_px") or width * tile_width, "width_px")
        height_px = _positive_int(payload.get("height_px") or height * tile_height, "height_px")
        raw = payload.get("blocked")
        if raw is None:
            raise ValueError("terrain payload missing 'blocked' mask")
        values = [1 if int(value) != 0 else 0 for value in raw]
        expected = width * height
        if len(values) < expected:
            values.extend([0] * (expected - len(values)))
        elif len(values) > expected:
            values = values[:expected]
        return cls(
            map_id=str(payload.get("id") or ""),
            name=str(payload.get("name") or ""),
            width=width,
            height=height,
            tile_width=tile_width,
            tile_height=tile_height,
            width_px=width_px,
            height_px=height_px,
            blocked=tuple(bool(value) for value in values),
        )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def tile_count(self) -> int:
        return self.width * self.height

    @property
    def land_ratio(self) -> float:
        if not self.blocked:
            return 0.0
        return 1.0 - (sum(1 for value in self.blocked if value) / len(self.blocked))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def world_to_tile(self, x: float, y: float) -> tuple[int, int]:
        """World (pixel) point -> tile indices, clamped into the map."""

        tx = int(math.floor(x / self.tile_width))
        ty = int(math.floor(y / self.tile_height))
        return (
            _clamp_int(tx, 0, self.width - 1),
            _clamp_int(ty, 0, self.height - 1),
        )

    def _tile_unclamped(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor(x / self.tile_width)),
            int(math.floor(y / self.tile_height)),
        )

    def tile_center(self, tx: int, ty: int) -> tuple[float, float]:
        return (
            (tx + 0.5) * self.tile_width,
            (ty + 0.5) * self.tile_height,
        )

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------
    def is_blocked_tile(self, tx: int, ty: int) -> bool:
        if tx < 0 or ty < 0 or tx >= self.width or ty >= self.height:
            return True
        return self.blocked[ty * self.width + tx]

    def is_blocked_world(self, x: float, y: float) -> bool:
        """True when the world point is outside the map or on blocked ground."""

        if not (math.isfinite(x) and math.isfinite(y)):
            return True
        tx, ty = self._tile_unclamped(x, y)
        return self.is_blocked_tile(tx, ty)

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
                return (float(_clamp(x, 0.0, self.width_px - 1.0)), float(_clamp(y, 0.0, self.height_px - 1.0)))
            return point
        if not self.is_blocked_world(x, y):
            return (float(x), float(y))
        point = self._nearest_passable_in_box(x, y, box)
        if point is None and box is not None:
            point = self._nearest_passable_in_box(x, y, None)
        if point is None:  # fully blocked map: stay put (deterministic fallback)
            return (float(_clamp(x, 0.0, self.width_px - 1.0)), float(_clamp(y, 0.0, self.height_px - 1.0)))
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

    # ------------------------------------------------------------------
    # A*
    # ------------------------------------------------------------------
    def find_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        exact_goal: bool = False,
    ) -> list[tuple[float, float]] | None:
        """Deterministic 4-neighbour A* between two world points.

        Returns a list of world-space waypoints, or ``None`` when the goal is
        unreachable (or the node budget is exhausted). Intermediate waypoints
        are tile centres; with ``exact_goal=True`` the final waypoint is the
        exact goal world point when the goal itself stands on passable ground.
        Consecutive collinear tile centres are merged into a single waypoint so
        the follower only has to walk to the next turn (a straight run through
        passable tiles) instead of every tile centre.

        Ties in the open set are broken deterministically by
        ``(f, h, tile_y, tile_x)``; the 4-neighbour order is up/down/left/right.
        """

        if not (math.isfinite(start[0]) and math.isfinite(start[1])):
            return None
        if not (math.isfinite(goal[0]) and math.isfinite(goal[1])):
            return None
        start_tile = self.world_to_tile(start[0], start[1])
        goal_tile = self.world_to_tile(goal[0], goal[1])
        goal_point = (float(goal[0]), float(goal[1]))
        if self.is_blocked_tile(*start_tile):
            start_tile = self.world_to_tile(*self.nearest_passable_world(start[0], start[1]))
        if self.is_blocked_tile(*goal_tile):
            goal_point = self.nearest_passable_world(goal[0], goal[1])
            goal_tile = self.world_to_tile(*goal_point)
        if goal_tile == start_tile:
            return [goal_point] if exact_goal else [self.tile_center(*goal_tile)]

        width, height = self.width, self.height
        blocked = self.blocked
        start_index = start_tile[1] * width + start_tile[0]
        goal_index = goal_tile[1] * width + goal_tile[0]
        inf = float("inf")
        g_score = [inf] * (width * height)
        came_from = [-1] * (width * height)
        closed = bytearray(width * height)
        g_score[start_index] = 0.0
        start_h = abs(start_tile[0] - goal_tile[0]) + abs(start_tile[1] - goal_tile[1])
        open_heap: list[tuple[float, int, int, int, float, int]] = [
            (float(start_h), start_h, start_tile[1], start_tile[0], 0.0, start_index)
        ]
        found = False
        expanded = 0
        while open_heap:
            _, _, ty, tx, g_value, index = heapq.heappop(open_heap)
            if closed[index]:
                continue
            closed[index] = 1
            expanded += 1
            if index == goal_index:
                found = True
                break
            if expanded > max_nodes:
                return None
            for dtx, dty in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                ntx = tx + dtx
                nty = ty + dty
                if ntx < 0 or nty < 0 or ntx >= width or nty >= height:
                    continue
                nindex = nty * width + ntx
                if closed[nindex] or blocked[nindex]:
                    continue
                candidate_g = g_value + 1.0
                if candidate_g < g_score[nindex]:
                    g_score[nindex] = candidate_g
                    came_from[nindex] = index
                    h_value = abs(ntx - goal_tile[0]) + abs(nty - goal_tile[1])
                    heapq.heappush(
                        open_heap,
                        (candidate_g + h_value, h_value, nty, ntx, candidate_g, nindex),
                    )
        if not found:
            return None

        path_tiles: list[tuple[int, int]] = []
        node = goal_index
        while node != start_index and node != -1:
            path_tiles.append((node % width, node // width))
            node = came_from[node]
        if node != start_index:  # pragma: no cover - defensive, cannot happen when found
            return None
        path_tiles.reverse()
        waypoints = self._compress_waypoints(path_tiles)
        if exact_goal:
            last_x, last_y = waypoints[-1]
            last_tile = self.world_to_tile(last_x, last_y)
            if last_tile == goal_tile and not self.is_blocked_world(*goal_point):
                waypoints[-1] = goal_point
            else:
                waypoints.append(goal_point)
        return waypoints

    def _compress_waypoints(self, path_tiles: Sequence[tuple[int, int]]) -> list[tuple[float, float]]:
        """Tile centres with collinear runs merged down to their turn points."""

        if not path_tiles:
            return []
        centres = [self.tile_center(tx, ty) for tx, ty in path_tiles]
        if len(centres) <= 2:
            return centres
        waypoints: list[tuple[float, float]] = [centres[0]]
        previous_direction: tuple[float, float] | None = None
        for index in range(1, len(centres)):
            direction = (
                centres[index][0] - centres[index - 1][0],
                centres[index][1] - centres[index - 1][1],
            )
            if previous_direction is None or direction != previous_direction:
                waypoints.append(centres[index])
                previous_direction = direction
            elif waypoints:
                # Same heading: extend the run instead of adding a new waypoint.
                waypoints[-1] = centres[index]
        return waypoints

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def path_length(
        self,
        path: Sequence[tuple[float, float]],
        start: tuple[float, float] | None = None,
    ) -> float:
        """Euclidean length of a waypoint list (optionally including ``start``)."""

        total = 0.0
        previous = start
        for point in path:
            if previous is not None:
                total += math.hypot(point[0] - previous[0], point[1] - previous[1])
            previous = point
        return total


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _positive_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"terrain payload field {field!r} is not an int: {value!r}") from exc
    if number <= 0:
        raise ValueError(f"terrain payload field {field!r} must be positive, got {number}")
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _clamp_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def _clamp_bounds(
    bounds: tuple[float, float, float, float] | None,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float, float] | None:
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    min_x = _clamp(min_x, 0.0, float(width_px))
    max_x = _clamp(max_x, 0.0, float(width_px))
    min_y = _clamp(min_y, 0.0, float(height_px))
    max_y = _clamp(max_y, 0.0, float(height_px))
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y
    return (min_x, min_y, max_x, max_y)
