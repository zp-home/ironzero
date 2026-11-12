# SPDX-License-Identifier: Apache-2.0
"""MovementMixin: arena straight-line movement and terrain A* waypoint following.

Arena mode (``terrain is None``) keeps the original straight-line stepper.
Terrain mode follows a cached A* path, re-plans lazily, slides along walls when
the direct heading is blocked and drops the order after being stuck too long.
"""

from __future__ import annotations

import math
from typing import Sequence

from .constants import (
    TERRAIN_MAX_PATH_NODES,
    TERRAIN_REPATH_TICKS,
    TERRAIN_STUCK_LIMIT_TICKS,
)
from .entities import Unit
from .util import _clamp


class MovementMixin:
    """Per-unit movement, path planning and collision handling."""

    def _move(self, unit: Unit) -> None:
        if unit.is_building:
            return
        if self.terrain is not None:
            self._move_terrain(unit)
            return
        cfg = self.config
        speed = cfg.unit_speed / cfg.tick_rate
        if unit.order == 0:
            return
        if unit.order == 4:
            return
        target = None
        if unit.order == 3 and unit.target_id is not None:
            target = self.unit_by_id(unit.target_id)
            if target is None or not target.alive:
                unit.order = 0
                return
        goal_x = target.x if target is not None else unit.tx
        goal_y = target.y if target is not None else unit.ty
        dx, dy = goal_x - unit.x, goal_y - unit.y
        distance = math.hypot(dx, dy)
        if unit.order == 3 and target is not None:
            stop_at = cfg.attack_range * 0.85
        else:
            stop_at = 2.0
        if distance <= stop_at:
            if unit.order == 2 and target is None:
                unit.order = 0
            return
        step = min(speed, distance - stop_at + 1e-9)
        unit.x += dx / distance * step
        unit.y += dy / distance * step
        unit.x = _clamp(unit.x, 0.0, cfg.map_width)
        unit.y = _clamp(unit.y, 0.0, cfg.map_height)

    # ------------------------------------------------------------------
    # Terrain-mode movement (A* waypoint following with water collision)
    # ------------------------------------------------------------------
    def _move_terrain(self, unit: Unit) -> None:
        cfg = self.config
        terrain = self.terrain
        assert terrain is not None
        speed = cfg.unit_speed / cfg.tick_rate
        if unit.order == 0 or unit.order == 4:
            return

        target: Unit | None = None
        if unit.order == 3:
            if unit.target_id is None:
                unit.order = 0
                return
            target = self.unit_by_id(unit.target_id)
            if target is None or not target.alive:
                unit.order = 0
                unit.target_id = None
                self._reset_unit_path(unit)
                return
            stop_at = cfg.attack_range * 0.85
            if self._distance(unit, target) <= stop_at:
                self._reset_unit_path(unit)
                return
            goal = (target.x, target.y)
        else:
            stop_at = 2.0
            goal = (unit.tx, unit.ty)
            if math.hypot(goal[0] - unit.x, goal[1] - unit.y) <= stop_at:
                # Mirrors arena semantics: attack_move completes, plain move
                # keeps its order (plain move never auto-engages).
                if unit.order == 2:
                    unit.order = 0
                self._reset_unit_path(unit)
                return

        # (Re)plan lazily: at order time a path is already installed, this only
        # fires when the path was consumed, became stale, or failed to exist.
        need_path = unit.path is None or unit.path_index >= len(unit.path)
        if not need_path and self._path_stale(unit, goal):
            need_path = True
        if need_path and self.tick >= unit.repath_at:
            self._repath_terrain(unit, goal)

        if unit.path is not None and unit.path_index < len(unit.path):
            waypoint = unit.path[unit.path_index]
            moved, arrived = self._step_toward(unit, waypoint[0], waypoint[1], speed)
            if arrived:
                unit.path_index += 1
        else:
            # A* failed (no route): straight-line fallback with wall sliding.
            moved, arrived = self._step_toward(unit, goal[0], goal[1], speed)
            if arrived and unit.order == 3:
                self._reset_unit_path(unit)

        if moved:
            unit.stuck_ticks = 0
        else:
            unit.stuck_ticks += 1
            if unit.stuck_ticks > TERRAIN_STUCK_LIMIT_TICKS:
                # No walkable route for half a second: stop and drop the order.
                unit.order = 0
                unit.target_id = None
                unit.tx = unit.x
                unit.ty = unit.y
                unit.stuck_ticks = 0
                self._reset_unit_path(unit)

    def _assign_group_path(
        self,
        members: Sequence[Unit],
        goal: tuple[float, float],
    ) -> None:
        """Plan once for the squad anchor and let the group share the waypoints.

        Group members start within a tile or two of each other, so one A* per
        group keeps terrain pathing cheap (4 A* per broadcast order at most)
        while each tank still walks the path with its own collision checks.
        """

        assert self.terrain is not None
        anchor = members[0]
        path = self.terrain.find_path(
            (anchor.x, anchor.y),
            goal,
            max_nodes=TERRAIN_MAX_PATH_NODES,
            exact_goal=True,
        )
        goal_tile = self.terrain.world_to_tile(*goal)
        for member in members:
            member.path = path
            member.path_index = 0
            member.path_goal = goal
            member.path_goal_tile = goal_tile
            member.repath_at = self.tick + TERRAIN_REPATH_TICKS
            member.stuck_ticks = 0

    def _repath_terrain(self, unit: Unit, goal: tuple[float, float]) -> None:
        assert self.terrain is not None
        path = self.terrain.find_path(
            (unit.x, unit.y),
            goal,
            max_nodes=TERRAIN_MAX_PATH_NODES,
            exact_goal=True,
        )
        unit.path = path
        unit.path_index = 0
        unit.path_goal = goal
        unit.path_goal_tile = self.terrain.world_to_tile(*goal)
        unit.repath_at = self.tick + TERRAIN_REPATH_TICKS

    def _path_stale(self, unit: Unit, goal: tuple[float, float]) -> bool:
        assert self.terrain is not None
        if unit.path_goal is None:
            return True
        if unit.order == 3:
            return unit.path_goal_tile != self.terrain.world_to_tile(*goal)
        return math.hypot(goal[0] - unit.path_goal[0], goal[1] - unit.path_goal[1]) > 0.5

    @staticmethod
    def _reset_unit_path(unit: Unit) -> None:
        unit.path = None
        unit.path_index = 0
        unit.path_goal = None
        unit.path_goal_tile = None
        unit.repath_at = 0
        unit.stuck_ticks = 0

    def _step_toward(
        self,
        unit: Unit,
        target_x: float,
        target_y: float,
        step: float,
    ) -> tuple[bool, bool]:
        """Move ``unit`` at most ``step`` px toward a point, never into water.

        Returns ``(moved, arrived)``. The full step is tried first, then the
        dominant axis alone (wall sliding), then a half step and a quarter step.
        Every accepted position is validated against the blocked mask, so a unit
        can never walk into a water tile even with a large per-tick step.
        """

        assert self.terrain is not None
        dx = target_x - unit.x
        dy = target_y - unit.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return (False, True)
        step = min(step, distance)
        head_x = dx / distance
        head_y = dy / distance
        if self._try_offset(unit, head_x * step, head_y * step):
            return (True, step >= distance - 1e-9)
        # Slide along the dominant axis at full speed (capped by the remaining
        # axis distance so it cannot oscillate around the goal): the classic
        # "walk along the shore" behaviour when the direct heading is blocked.
        axis_step_x = math.copysign(min(abs(dx), step), dx) if dx != 0.0 else 0.0
        axis_step_y = math.copysign(min(abs(dy), step), dy) if dy != 0.0 else 0.0
        offsets: list[tuple[float, float]] = []
        if abs(head_x) >= abs(head_y):
            offsets.append((axis_step_x, 0.0))
            offsets.append((0.0, axis_step_y))
        else:
            offsets.append((0.0, axis_step_y))
            offsets.append((axis_step_x, 0.0))
        offsets.append((head_x * step * 0.5, head_y * step * 0.5))
        offsets.append((head_x * step * 0.25, head_y * step * 0.25))
        for offset_x, offset_y in offsets:
            if offset_x == 0.0 and offset_y == 0.0:
                continue
            if self._try_offset(unit, offset_x, offset_y):
                return (True, False)
        return (False, False)

    def _try_offset(self, unit: Unit, offset_x: float, offset_y: float) -> bool:
        assert self.terrain is not None
        new_x = unit.x + offset_x
        new_y = unit.y + offset_y
        if self.terrain.is_blocked_world(new_x, new_y):
            return False
        unit.x = _clamp(new_x, 0.0, self.config.map_width)
        unit.y = _clamp(new_y, 0.0, self.config.map_height)
        return True

    @staticmethod
    def _distance(a: Unit, b: Unit) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)
