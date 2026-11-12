# SPDX-License-Identifier: Apache-2.0
"""SpawnMixin: episode reset and unit placement.

``reset()`` re-seeds the RNG, rebuilds funds / resource slots / stats and then
places both armies either on the arena grid or on passable terrain ground.
"""

from __future__ import annotations

import math
import random

from .entities import MockResourceSlot, Unit


class SpawnMixin:
    """Episode lifecycle: reset and spawn placement."""

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
            # Slot layout is driven by ``resource_slots_count`` (4 by default):
            # the first two flank each base, the remainder ring the map centre.
            count = max(1, int(cfg.resource_slots_count))
            for index in range(count):
                if index == 0:
                    x, y = margin * 0.8, cy
                elif index == 1:
                    x, y = cfg.map_width - margin * 0.8, cy
                else:
                    angle = (index - 2) * (math.pi / max(1, count - 2))
                    radius = min(cx, cy) * 0.45
                    x = cx + radius * math.cos(angle)
                    y = cy + radius * math.sin(angle)
                    x = min(max(x, 50.0), cfg.map_width - 50.0)
                    y = min(max(y, 50.0), cfg.map_height - 50.0)
                self.resource_slots.append(
                    MockResourceSlot(slot_id=index, x=x, y=y, occupied=False)
                )
        self.stats = {
            "damage_dealt": {team: 0.0 for team in range(cfg.teams)},
            "damage_taken": {team: 0.0 for team in range(cfg.teams)},
            "kills": {team: 0 for team in range(cfg.teams)},
            "losses": {team: 0 for team in range(cfg.teams)},
        }
        self.economic_events = {
            team: {
                "extractor_built": 0,
                "extractor_upgraded": 0,
                "factory_built": 0,
                "unit_produced": 0,
                "building_destroyed": 0,
            }
            for team in range(cfg.teams)
        }
        self.latest_reward_components = {}
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
