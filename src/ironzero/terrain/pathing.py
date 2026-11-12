# SPDX-License-Identifier: Apache-2.0
"""Deterministic 4-neighbour A* over the terrain grid.

Extracted from ``TerrainGrid`` as a mixin: A* is the largest single block of the
original 414-line module, and it only needs the public surface declared in
``grid.py`` (``width`` / ``height`` / ``blocked`` / ``tile_center`` /
``world_to_tile`` / ``is_blocked_tile`` / ``is_blocked_world`` /
``nearest_passable_world``). It adds no instance state, so the frozen dataclass
contract of :class:`~ironzero.terrain.grid.TerrainGrid` is untouched.

The tie-break order, the up/down/left/right neighbour order and the node budget
are byte-for-byte identical to the original implementation — the engine relies on
run-to-run determinism for reproducible episodes.
"""

from __future__ import annotations

import heapq
import math
from typing import Sequence

from .utils import _clamp_int  # noqa: F401  (re-exported for parity with the old module)

# Hard cap on A* node expansions. Lake-sized maps (130x130 = 16900 tiles) stay
# well below this; the cap only protects against pathological / giant maps.
DEFAULT_MAX_NODES = 30000

__all__ = ["DEFAULT_MAX_NODES", "PathingMixin"]


class PathingMixin:
    """``find_path`` / ``_compress_waypoints`` / ``path_length``."""

    __slots__ = ()

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
