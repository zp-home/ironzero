# SPDX-License-Identifier: Apache-2.0
"""Entity data classes of the micro-duel simulation.

``Unit`` is used for tanks, builders and every building (extractor, factory,
turret) — the ``is_building`` / ``entity_type_id`` pair decides the behaviour.
The terrain fields at the bottom are only touched in terrain mode and stay at
their defaults in arena mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import TANK_TYPE_ID


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
