# SPDX-License-Identifier: Apache-2.0
"""Deterministic micro-duel engine used as the reference game adapter.

This module plays the role of the in-game side of the bridge: it owns the
simulation, validates and executes actions, advances ticks and produces raw
observation payloads that are shaped like ``ironzero/1`` messages.

It is intentionally dependency-free (standard library only) and fully
deterministic given ``(scenario, seed)``. That determinism is what lets the
project test the *whole* pipeline before the real RWPP headless runtime is
available, and it doubles as the fixed-baseline environment for CI.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .terrain import TerrainGrid

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
    "repair_target",
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


@dataclass(slots=True)
class MockResourceSlot:
    slot_id: int
    x: float
    y: float
    occupied: bool = False
    building_entity_id: int | None = None
    owner_team: int | None = None
    extractor_level: int = 0


@dataclass(slots=True)
class Unit:
    entity_id: int
    team: int
    group: int
    x: float
    y: float
    max_hp: float
    hp: float
    order: int = 0
    tx: float = 0.0
    ty: float = 0.0
    target_id: int | None = None
    cooldown: int = 0
    alive: bool = True
    is_building: bool = False
    entity_type_id: int = TANK_TYPE_ID
    build_progress: float = 1.0
    extractor_level: int = 0
    production_queue: list[str] = field(default_factory=list)
    production_timer: int = 0
    # Terrain-mode only state (ignored in arena mode).
    path: list[tuple[float, float]] | None = None
    path_index: int = 0
    path_goal: tuple[float, float] | None = None
    path_goal_tile: tuple[int, int] | None = None
    repath_at: int = 0
    stuck_ticks: int = 0


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


class MicroDuelEngine:
    """Two teams of tanks on an open map, micro control only."""

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        terrain: TerrainGrid | None = None,
    ) -> None:
        self.terrain = terrain
        base_config = config or DEFAULT_CONFIG
        if terrain is not None:
            # Terrain mode: the real map's pixel size replaces the arena size.
            self.config = make_terrain_config(terrain, base_config)
        else:
            self.config = base_config
        self.opponent = opponent or self.config.opponent
        if self.opponent not in {"scripted", "idle", "random"}:
            raise ValueError(f"unknown opponent: {self.opponent}")
        self._rng = random.Random(0)
        self.units: list[Unit] = []
        self.tick = 0
        self.done = False
        self.result: str | None = None
        self.events: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self._phi_prev = 0.0
        self._pending_reward = 0.0
        self.reset(seed=0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        cfg = self.config
        self._rng = random.Random(seed)
        self.tick = 0
        self.done = False
        self.result = None
        self.events = []
        self.units = []
        self.funds = {
            team: float(cfg.starting_funds if cfg.enable_economy else 0.0)
            for team in range(cfg.teams)
        }
        self.resource_slots = []
        if cfg.enable_economy:
            cx, cy = cfg.map_width / 2.0, cfg.map_height / 2.0
            margin = cfg.spawn_margin
            self.resource_slots = [
                MockResourceSlot(slot_id=0, x=margin * 0.8, y=cy, occupied=False),
                MockResourceSlot(slot_id=1, x=cfg.map_width - margin * 0.8, y=cy, occupied=False),
                MockResourceSlot(slot_id=2, x=cx, y=max(50.0, cy - 200.0), occupied=False),
                MockResourceSlot(slot_id=3, x=cx, y=min(cfg.map_height - 50.0, cy + 200.0), occupied=False),
            ]
        self.stats = {
            "damage_dealt": {team: 0.0 for team in range(cfg.teams)},
            "damage_taken": {team: 0.0 for team in range(cfg.teams)},
            "kills": {team: 0 for team in range(cfg.teams)},
            "losses": {team: 0 for team in range(cfg.teams)},
        }
        per_team = cfg.units_per_team
        if self.terrain is not None:
            self._spawn_terrain_units()
        else:
            rows = max(1, math.ceil(per_team / 2))
            spacing_y = min(72.0, (cfg.map_height - 2 * cfg.spawn_margin) / max(1, rows - 1)) if rows > 1 else 0.0
            entity_id = 1
            for team in range(cfg.teams):
                side_x = cfg.spawn_margin if team == 0 else cfg.map_width - cfg.spawn_margin
                for index in range(per_team):
                    column = index % 2
                    row = index // 2
                    jitter = self._rng.uniform(-10.0, 10.0)
                    x = side_x + (18.0 if team == 0 else -18.0) * column + jitter
                    y = cfg.map_height / 2.0 + (row - (rows - 1) / 2.0) * spacing_y + jitter
                    group = min(index // max(1, per_team // cfg.groups), cfg.groups - 1)
                    self.units.append(
                        Unit(
                            entity_id=entity_id,
                            team=team,
                            group=group,
                            x=x,
                            y=y,
                            max_hp=cfg.unit_hp,
                            hp=cfg.unit_hp,
                            tx=x,
                            ty=y,
                        )
                    )
                    entity_id += 1
        self._phi_prev = self._potential()
        self._pending_reward = 0.0

    def _spawn_terrain_units(self) -> None:
        """Spawn both teams on real passable ground, left vs right of the map.

        Each unit gets a desired position (mirroring the arena layout, scaled to
        map size) which is then snapped to the nearest passable tile *inside its
        own half* of the map, so nobody starts in water and the two armies stay
        on opposite sides. Jitter still comes from the seeded RNG, which keeps
        spawns reproducible for a fixed ``(seed, terrain)``.
        """

        cfg = self.config
        terrain = self.terrain
        per_team = cfg.units_per_team
        rows = max(1, math.ceil(per_team / 2))
        margin = cfg.spawn_margin
        vertical_span = max(1.0, cfg.map_height - 2.0 * margin)
        spacing_y = min(cfg.map_height * 0.06, vertical_span / max(1, rows - 1)) if rows > 1 else 0.0
        y_low = min(margin, cfg.map_height * 0.25)
        y_high = max(cfg.map_height - margin, cfg.map_height * 0.75)
        entity_id = 1
        for team in range(cfg.teams):
            left = team == 0
            side_x = margin if left else cfg.map_width - margin
            bounds = (
                0.0 if left else cfg.map_width * 0.5,
                y_low,
                cfg.map_width * 0.5 if left else cfg.map_width,
                y_high,
            )
            for index in range(per_team):
                column = index % 2
                row = index // 2
                jitter = self._rng.uniform(-10.0, 10.0)
                desired_x = side_x + (18.0 if left else -18.0) * column + jitter
                desired_y = (
                    cfg.map_height / 2.0
                    + (row - (rows - 1) / 2.0) * spacing_y
                    + jitter
                )
                x, y = terrain.nearest_passable_world(desired_x, desired_y, bounds=bounds)
                group = min(index // max(1, per_team // cfg.groups), cfg.groups - 1)
                self.units.append(
                    Unit(
                        entity_id=entity_id,
                        team=team,
                        group=group,
                        x=x,
                        y=y,
                        max_hp=cfg.unit_hp,
                        hp=cfg.unit_hp,
                        tx=x,
                        ty=y,
                    )
                )
                entity_id += 1

    def alive_units(self, team: int | None = None) -> list[Unit]:
        return [u for u in self.units if u.alive and (team is None or u.team == team)]

    def unit_by_id(self, entity_id: int) -> Unit | None:
        for unit in self.units:
            if unit.entity_id == entity_id:
                return unit
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def apply_actions(
        self,
        actions: Sequence[dict[str, Any]],
        team: int = 0,
    ) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        if self.done:
            return [
                {"accepted": False, "rejected_reason": STALE_TICK, "action_index": i}
                for i, _ in enumerate(actions)
            ]
        for index, action in enumerate(actions[:64]):
            reason = self._apply_one(action, team)
            receipts.append(
                {
                    "action_index": index,
                    "kind": action.get("kind"),
                    "accepted": reason is None,
                    "rejected_reason": reason,
                }
            )
        return receipts

    def _apply_one(self, action: dict[str, Any], team: int) -> str | None:
        kind = action.get("kind")
        if kind not in ACTION_KINDS:
            return INVALID_MESSAGE

        if kind == "build_extractor":
            slot_id = action.get("slot_id")
            if not isinstance(slot_id, int) or not (0 <= slot_id < len(self.resource_slots)):
                return INVALID_MESSAGE
            slot = self.resource_slots[slot_id]
            if slot.occupied:
                return ACTION_MASKED
            if self.funds.get(team, 0.0) < self.config.extractor_cost:
                return INSUFFICIENT_FUNDS
            self.funds[team] -= self.config.extractor_cost
            new_id = max((u.entity_id for u in self.units), default=0) + 1
            extractor = Unit(
                entity_id=new_id,
                team=team,
                group=0,
                x=slot.x,
                y=slot.y,
                max_hp=600.0,
                hp=600.0,
                is_building=True,
                entity_type_id=EXTRACTOR_TYPE_ID,
                build_progress=1.0,
                extractor_level=1,
            )
            self.units.append(extractor)
            slot.occupied = True
            slot.building_entity_id = new_id
            slot.owner_team = team
            slot.extractor_level = 1
            return None

        if kind == "build_structure":
            stype = action.get("structure_type", "land_factory")
            cost = self.config.factory_cost if stype == "land_factory" else self.config.turret_cost
            if self.funds.get(team, 0.0) < cost:
                return INSUFFICIENT_FUNDS
            x = float(action.get("x", math.nan))
            y = float(action.get("y", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                return INVALID_MESSAGE
            if not (0.0 <= x <= self.config.map_width and 0.0 <= y <= self.config.map_height):
                return POSITION_BLOCKED
            self.funds[team] -= cost
            new_id = max((u.entity_id for u in self.units), default=0) + 1
            type_id = LAND_FACTORY_TYPE_ID if stype == "land_factory" else TURRET_TYPE_ID
            hp = 1200.0 if stype == "land_factory" else 450.0
            building = Unit(
                entity_id=new_id,
                team=team,
                group=0,
                x=x,
                y=y,
                max_hp=hp,
                hp=hp,
                is_building=True,
                entity_type_id=type_id,
                build_progress=1.0,
            )
            self.units.append(building)
            return None

        if kind == "produce_unit":
            unit_type = action.get("unit_type", "tank")
            cost = self.config.tank_cost
            if self.funds.get(team, 0.0) < cost:
                return INSUFFICIENT_FUNDS
            factory_id = action.get("factory_entity_id")
            factory: Unit | None = None
            if factory_id is not None:
                u = self.unit_by_id(factory_id)
                if u is not None and u.alive and u.team == team and u.entity_type_id == LAND_FACTORY_TYPE_ID:
                    factory = u
            if factory is None:
                for u in self.alive_units(team):
                    if u.is_building and u.entity_type_id == LAND_FACTORY_TYPE_ID:
                        factory = u
                        break
            if factory is None:
                return ACTION_MASKED
            self.funds[team] -= cost
            factory.production_queue.append(unit_type)
            return None

        if kind == "upgrade_building":
            building_id = action.get("building_entity_id")
            building = self.unit_by_id(building_id) if isinstance(building_id, int) else None
            if building is None or not building.alive or building.team != team or not building.is_building:
                return INVALID_MESSAGE
            if building.entity_type_id == EXTRACTOR_TYPE_ID and building.extractor_level == 1:
                cost = self.config.extractor_upgrade_cost
                if self.funds.get(team, 0.0) < cost:
                    return INSUFFICIENT_FUNDS
                self.funds[team] -= cost
                building.extractor_level = 2
                for slot in self.resource_slots:
                    if slot.building_entity_id == building.entity_id:
                        slot.extractor_level = 2
                return None
            return ACTION_MASKED

        if kind == "repair_target":
            target_id = action.get("target_entity_id")
            target = self.unit_by_id(target_id) if isinstance(target_id, int) else None
            if target is None or not target.alive or target.team != team:
                return INVALID_MESSAGE
            target.hp = min(target.max_hp, target.hp + 50.0)
            return None

        group = action.get("unit_group")
        if not isinstance(group, int) or not 0 <= group < self.config.groups:
            return UNKNOWN_UNIT
        members = [u for u in self.alive_units(team) if u.group == group]
        if not members:
            return UNIT_NOT_OWNED

        if kind in {"move_group", "attack_move"}:
            x = float(action.get("x", math.nan))
            y = float(action.get("y", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                return INVALID_MESSAGE
            if not (0.0 <= x <= self.config.map_width and 0.0 <= y <= self.config.map_height):
                return POSITION_BLOCKED
            if self.terrain is not None:
                # A target inside water means "move to the nearest shore".
                x, y = self.terrain.nearest_passable_world(x, y)
            order = 1 if kind == "move_group" else 2
            for member in members:
                member.order = order
                member.tx = x
                member.ty = y
                member.target_id = None
            if self.terrain is not None:
                self._assign_group_path(members, (x, y))
            return None

        if kind == "attack_target":
            target_id = action.get("target_entity_id")
            if not isinstance(target_id, int) or target_id < 0:
                return INVALID_MESSAGE
            target = self.unit_by_id(target_id)
            if target is None or not target.alive or target.team == team:
                return TARGET_NOT_VISIBLE
            for member in members:
                member.order = 3
                member.target_id = target_id
            if self.terrain is not None:
                self._assign_group_path(members, (target.x, target.y))
            return None

        if kind == "stop":
            for member in members:
                member.order = 0
                member.target_id = None
                member.tx = member.x
                member.ty = member.y
                self._reset_unit_path(member)
            return None

        # hold
        for member in members:
            member.order = 4
            member.target_id = None
            member.tx = member.x
            member.ty = member.y
            self._reset_unit_path(member)
        return None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def advance(self, ticks: int | None = None) -> None:
        steps = self.config.decision_interval if ticks is None else ticks
        for _ in range(steps):
            if self.done:
                break
            self._tick()
        self._pending_reward = self._reward_after_window()

    def step_decision(
        self,
        actions: Sequence[dict[str, Any]],
        team: int = 0,
    ) -> dict[str, Any]:
        """Apply one decision, run the opponent, advance a decision window."""
        self.events = []
        receipts = self.apply_actions(actions, team=team)
        opponent_actions = self._opponent_actions(team=team)
        if opponent_actions:
            self.apply_actions(opponent_actions, team=1)
        self.advance()
        state = self.observation(team=team)
        state.update(
            {
                "reward": self._pending_reward,
                "terminated": self.done and self.result in {"victory", "defeat"},
                "truncated": self.done and self.result == "timeout",
                "result": self.result,
                "receipts": receipts,
                "events": list(self.events),
                "decision_id": None,
            }
        )
        return state

    def _tick(self) -> None:
        cfg = self.config
        self.tick += 1
        if cfg.enable_economy:
            for team in range(cfg.teams):
                self.funds[team] += cfg.base_income_per_second / cfg.tick_rate
            for unit in list(self.units):
                if not unit.alive or not unit.is_building:
                    continue
                if unit.entity_type_id == EXTRACTOR_TYPE_ID and unit.build_progress >= 1.0:
                    inc = (
                        cfg.extractor_income_per_second
                        if unit.extractor_level <= 1
                        else cfg.extractor_t2_income_per_second
                    )
                    self.funds[unit.team] += inc / cfg.tick_rate
                elif unit.entity_type_id == LAND_FACTORY_TYPE_ID and unit.build_progress >= 1.0:
                    if unit.production_queue:
                        unit.production_timer += 1
                        if unit.production_timer >= 60:
                            unit.production_timer = 0
                            unit.production_queue.pop(0)
                            new_id = max((u.entity_id for u in self.units), default=0) + 1
                            offset_x = 35.0 if unit.team == 0 else -35.0
                            new_unit = Unit(
                                entity_id=new_id,
                                team=unit.team,
                                group=0,
                                x=unit.x + offset_x,
                                y=unit.y,
                                max_hp=cfg.unit_hp,
                                hp=cfg.unit_hp,
                                entity_type_id=TANK_TYPE_ID,
                            )
                            self.units.append(new_unit)

        for unit in self.alive_units():
            if unit.cooldown > 0:
                unit.cooldown -= 1
            target = self._select_target(unit)
            if (
                target is not None
                and unit.cooldown == 0
                and self._distance(unit, target) <= cfg.attack_range
            ):
                unit.cooldown = cfg.attack_cooldown_ticks
                self._damage(target, cfg.unit_damage, source=unit)
            self._move(unit)
        self._check_done()

    def _select_target(self, unit: Unit) -> Unit | None:
        if unit.is_building and unit.entity_type_id in {EXTRACTOR_TYPE_ID, LAND_FACTORY_TYPE_ID}:
            return None
        if unit.order == 1:  # plain move never auto-engages
            return None
        if unit.order == 3 and unit.target_id is not None:
            target = self.unit_by_id(unit.target_id)
            if target is not None and target.alive and self._visible_to(target, unit.team):
                return target
            unit.order = 0
            unit.target_id = None
            self._reset_unit_path(unit)
        best: Unit | None = None
        best_dist = math.inf
        for enemy in self.alive_units(1 - unit.team):
            if not self._visible_to(enemy, unit.team):
                continue
            dist = self._distance(unit, enemy)
            if dist < best_dist:
                best, best_dist = enemy, dist
        return best

    def _visible_to(self, target: Unit, team: int) -> bool:
        """Fog of war check: is ``target`` within vision of any unit of ``team``?"""

        radius = self.config.vision_radius
        if radius <= 0.0:
            return True
        return any(self._distance(observer, target) <= radius for observer in self.alive_units(team))

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

    def _damage(self, target: Unit, amount: float, source: Unit) -> None:
        if not target.alive:
            return
        actual = min(target.hp, amount)
        target.hp -= actual
        self.stats["damage_dealt"][source.team] += actual
        self.stats["damage_taken"][target.team] += actual
        self.events.append(
            {
                "type": "shot",
                "source": source.entity_id,
                "target": target.entity_id,
                "team": source.team,
                "amount": round(actual, 3),
                "tick": self.tick,
            }
        )
        if target.hp <= 0.0:
            target.alive = False
            target.hp = 0.0
            self.stats["kills"][source.team] += 1
            self.stats["losses"][target.team] += 1
            self.events.append(
                {
                    "type": "kill",
                    "source": source.entity_id,
                    "target": target.entity_id,
                    "team": source.team,
                    "x": round(target.x, 3),
                    "y": round(target.y, 3),
                    "tick": self.tick,
                }
            )

    def _check_done(self) -> None:
        cfg = self.config
        if self.done:
            return
        if not self.alive_units(1):
            self.done = True
            self.result = "victory"
        elif not self.alive_units(0):
            self.done = True
            self.result = "defeat"
        elif self.tick >= cfg.episode_ticks:
            self.done = True
            self.result = "timeout"

    # ------------------------------------------------------------------
    # Scripted opponents
    # ------------------------------------------------------------------
    def scripted_actions(self, team: int = 0) -> list[dict[str, Any]]:
        """Deterministic "attack the nearest enemy" orders for ``team``.

        Under fog of war the scripted AI cannot target what it cannot see, so it
        advances toward the nearest enemy position with an attack-move order and
        switches to an explicit attack once the target is visible.
        """

        if self.done:
            return []
        own = self.alive_units(team)
        enemies = self.alive_units(1 - team)
        if not own or not enemies:
            return []
        actions: list[dict[str, Any]] = []
        fog = self.config.vision_radius > 0.0
        for group in sorted({unit.group for unit in own}):
            members = [unit for unit in own if unit.group == group]
            anchor = members[0]
            nearest = min(enemies, key=lambda target: self._distance(anchor, target))
            if fog and not self._visible_to(nearest, team):
                actions.append(
                    {
                        "kind": "attack_move",
                        "unit_group": group,
                        "x": nearest.x,
                        "y": nearest.y,
                    }
                )
            else:
                actions.append(
                    {
                        "kind": "attack_target",
                        "unit_group": group,
                        "target_entity_id": nearest.entity_id,
                    }
                )
        return actions

    def _opponent_actions(self, team: int = 0) -> list[dict[str, Any]]:
        if self.done:
            return []
        enemy_team = 1 - team
        own = self.alive_units(enemy_team)
        if not own:
            return []
        targets = self.alive_units(team)
        if not targets:
            return []
        if self.opponent == "idle":
            return []
        if self.opponent == "random":
            actions: list[dict[str, Any]] = []
            for group in sorted({u.group for u in own}):
                member = next(u for u in own if u.group == group)
                kind = self._rng.choice(ACTION_KINDS)
                if kind in {"move_group", "attack_move"}:
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "x": self._rng.uniform(0.0, self.config.map_width),
                            "y": self._rng.uniform(0.0, self.config.map_height),
                        }
                    )
                elif kind == "attack_target":
                    actions.append(
                        {
                            "kind": kind,
                            "unit_group": group,
                            "target_entity_id": self._rng.choice(targets).entity_id,
                        }
                    )
                else:
                    actions.append({"kind": kind, "unit_group": group})
                _ = member
            return actions
        # scripted aggressor: advance/target exactly like scripted_actions()
        return self.scripted_actions(enemy_team)

    # ------------------------------------------------------------------
    # Observation payload (shape-compatible with ironzero/1 messages)
    # ------------------------------------------------------------------
    def _calculate_income(self, team: int) -> float:
        if not self.config.enable_economy:
            return 0.0
        income = self.config.base_income_per_second
        for u in self.alive_units(team):
            if u.is_building and u.entity_type_id == EXTRACTOR_TYPE_ID and u.build_progress >= 1.0:
                income += (
                    self.config.extractor_income_per_second
                    if u.extractor_level <= 1
                    else self.config.extractor_t2_income_per_second
                )
        return income

    def observation(self, team: int = 0) -> dict[str, Any]:
        cfg = self.config
        own = self.alive_units(team)
        enemies = self.alive_units(1 - team)
        if cfg.vision_radius > 0.0:
            enemies = [unit for unit in enemies if self._visible_to(unit, team)]
        own_hp = sum(u.hp for u in own)
        enemy_hp = sum(u.hp for u in enemies)
        own_buildings = [u for u in own if u.is_building]
        return {
            "tick": self.tick,
            "game_time": self.tick / cfg.tick_rate,
            "controlled_team": team,
            "fog": cfg.vision_radius > 0.0,
            "vision_radius": cfg.vision_radius,
            "global_state": {
                "unit_count": len([u for u in own if not u.is_building]),
                "building_count": len(own_buildings),
                "army_value": round(own_hp, 3),
                "visible_enemy_value": round(enemy_hp, 3),
                "kills": self.stats["kills"][team],
                "losses": self.stats["losses"][team],
                "income": round(self._calculate_income(team), 3),
                "funds": round(self.funds.get(team, 0.0), 3),
            },
            "own_entities": [self._entity_payload(u) for u in own],
            "visible_enemy_entities": [self._entity_payload(u) for u in enemies],
            "remembered_enemies": [],
            "local_maps": {},
            "resource_slots": [
                {
                    "slot_id": s.slot_id,
                    "x": s.x,
                    "y": s.y,
                    "occupied": s.occupied,
                    "building_entity_id": s.building_entity_id,
                    "owner_team": s.owner_team,
                    "extractor_level": s.extractor_level,
                }
                for s in self.resource_slots
            ],
            "target_ids": [u.entity_id for u in enemies],
            "tick_interval": cfg.decision_interval,
        }

    def _entity_payload(self, unit: Unit) -> dict[str, Any]:
        return {
            "entity_id": unit.entity_id,
            "entity_type_id": unit.entity_type_id,
            "team_id": unit.team,
            "x": round(unit.x, 4),
            "y": round(unit.y, 4),
            "hp": round(unit.hp, 4),
            "max_hp": unit.max_hp,
            "is_building": unit.is_building,
            "is_visible": True,
            "is_alive": unit.alive,
            "order_type": ORDER_NAMES[unit.order],
            "attack_target_id": unit.target_id,
            "group_id": unit.group,
            "cooldown_frac": round(unit.cooldown / self.config.attack_cooldown_ticks, 4),
            "build_progress": unit.build_progress,
            "production_queue_length": len(unit.production_queue) if unit.is_building else 0,
        }

    # ------------------------------------------------------------------
    # Rewards and diagnostics
    # ------------------------------------------------------------------
    def _potential(self) -> float:
        cfg = self.config
        total = float(cfg.units_per_team * cfg.unit_hp)
        own = sum(u.hp for u in self.alive_units(0)) / total
        enemy = sum(u.hp for u in self.alive_units(1)) / total
        return own - enemy

    def _reward_after_window(self) -> float:
        cfg = self.config
        phi = self._potential()
        reward = cfg.potential_coef * (phi - self._phi_prev)
        self._phi_prev = phi
        if self.done:
            if self.result == "victory":
                reward += cfg.win_reward
            elif self.result == "defeat":
                reward += cfg.loss_reward
        return reward

    def snapshot(self) -> tuple[Any, ...]:
        """Deterministic fingerprint used by replay/regression tests."""
        return (
            self.tick,
            self.result,
            tuple(
                (
                    u.entity_id,
                    u.team,
                    round(u.x, 3),
                    round(u.y, 3),
                    round(u.hp, 3),
                    u.order,
                    u.target_id,
                    u.alive,
                )
                for u in self.units
            ),
        )

    def evaluate(self, team: int = 0) -> dict[str, Any]:
        cfg = self.config
        total = float(cfg.units_per_team * cfg.unit_hp)
        return {
            "tick": self.tick,
            "result": self.result,
            "own_hp_frac": sum(u.hp for u in self.alive_units(team)) / total,
            "enemy_hp_frac": sum(u.hp for u in self.alive_units(1 - team)) / total,
            "own_alive": len(self.alive_units(team)),
            "enemy_alive": len(self.alive_units(1 - team)),
            "damage_dealt": self.stats["damage_dealt"][team],
            "damage_taken": self.stats["damage_taken"][team],
            "kills": self.stats["kills"][team],
            "losses": self.stats["losses"][team],
        }


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class MicroDuelAdapter:
    """Game-side adapter interface shared by the env and the TCP bridge."""

    def __init__(
        self,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        terrain: TerrainGrid | None = None,
    ) -> None:
        self.engine = MicroDuelEngine(config=config, opponent=opponent, terrain=terrain)
        self.terrain = terrain
        self.opponent = self.engine.opponent
        self.episode_id: str = "ep-0"

    def reset(self, *, episode_id: str, seed: int, scenario: str = "micro_duel") -> dict[str, Any]:
        if scenario != "micro_duel":
            raise ValueError(f"unsupported scenario: {scenario}")
        self.episode_id = episode_id
        self.engine.reset(seed=seed)
        return self.engine.observation()

    def step(self, actions: Iterable[dict[str, Any]], *, team: int = 0) -> dict[str, Any]:
        return self.engine.step_decision(list(actions), team=team)

    def evaluate(self, team: int = 0) -> dict[str, Any]:
        return self.engine.evaluate(team)

    def close(self) -> None:  # pragma: no cover - symmetry with TCP adapters
        return None
