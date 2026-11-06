# SPDX-License-Identifier: Apache-2.0
"""Tests for terrain-aware pathing, spawning and water collision.

These tests are fully self-contained: they synthesise small blocked masks
instead of depending on the local ``.game_cache`` game assets.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ironzero.mock_engine import (
    MicroDuelAdapter,
    MicroDuelEngine,
    make_terrain_config,
)
from ironzero.terrain import TerrainGrid


# ----------------------------------------------------------------------
# Synthetic terrain helpers
# ----------------------------------------------------------------------
def make_payload(
    width: int,
    height: int,
    *,
    tile_width: int = 10,
    tile_height: int = 10,
    is_water,
    map_id: str = "synthetic",
    name: str = "Synthetic Test Map",
) -> dict:
    blocked = [
        1 if is_water(tx, ty) else 0
        for ty in range(height)
        for tx in range(width)
    ]
    return {
        "id": map_id,
        "name": name,
        "width": width,
        "height": height,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "width_px": width * tile_width,
        "height_px": height * tile_height,
        "blocked": blocked,
    }


def band_map(*, with_gap: bool = True) -> TerrainGrid:
    """40x20 map (400x200 px) with a vertical water band and an optional gap.

    The band covers tiles x=18..21 for every row except rows y=2..4 when
    ``with_gap`` is true, so a path from the left to the right side either
    detours through the northern gap or does not exist at all.
    """

    def is_water(tx: int, ty: int) -> bool:
        if not 18 <= tx <= 21:
            return False
        if with_gap and 2 <= ty <= 4:
            return False
        return True

    return TerrainGrid.from_payload(
        make_payload(40, 20, is_water=is_water, map_id="band-gap" if with_gap else "band-solid")
    )


def lake_map() -> TerrainGrid:
    """60x40 map (600x400 px) with water on both edges and a central lake.

    Spawn bands at 12% / 88% of the width are inside the edge water strips, so
    the engine must snap spawn points onto the nearest passable shore.
    """

    def is_water(tx: int, ty: int) -> bool:
        if tx < 9 or tx > 50:
            return True
        if 20 <= tx <= 39 and 8 <= ty <= 31:
            return True
        return False

    return TerrainGrid.from_payload(make_payload(60, 40, is_water=is_water, map_id="mini-lake"))


class TerrainGridTests(unittest.TestCase):
    def test_from_payload_round_trip(self) -> None:
        payload = make_payload(10, 8, is_water=lambda tx, ty: tx == 4)
        grid = TerrainGrid.from_payload(payload)
        self.assertEqual(grid.width, 10)
        self.assertEqual(grid.height, 8)
        self.assertEqual(grid.width_px, 100)
        self.assertEqual(grid.height_px, 80)
        self.assertEqual(grid.tile_count, 80)
        self.assertTrue(grid.is_blocked_tile(4, 0))
        self.assertFalse(grid.is_blocked_tile(5, 0))
        self.assertTrue(grid.is_blocked_world(-1.0, 5.0))
        self.assertTrue(grid.is_blocked_world(200.0, 5.0))
        self.assertFalse(grid.is_blocked_world(50.0, 45.0))
        self.assertAlmostEqual(grid.land_ratio, 0.9)

    def test_from_payload_validation(self) -> None:
        with self.assertRaises(ValueError):
            TerrainGrid.from_payload({"width": 4, "height": 4})
        with self.assertRaises(ValueError):
            TerrainGrid.from_payload({"width": 0, "height": 4, "blocked": []})

    def test_from_payload_accepts_bool_and_short_masks(self) -> None:
        payload = {
            "id": "x",
            "name": "x",
            "width": 2,
            "height": 2,
            "tile_width": 10,
            "tile_height": 10,
            "blocked": [True, False],
        }
        grid = TerrainGrid.from_payload(payload)
        self.assertEqual(grid.blocked, (True, False, False, False))

    def test_nearest_passable_world_returns_shore(self) -> None:
        grid = band_map()
        x, y = grid.nearest_passable_world(190.0, 120.0)  # inside the band
        self.assertFalse(grid.is_blocked_world(x, y))
        self.assertLessEqual(math.hypot(x - 190.0, y - 120.0), 20.0)

    def test_nearest_passable_world_respects_bounds(self) -> None:
        grid = lake_map()
        x, y = grid.nearest_passable_world(60.0, 200.0, bounds=(0.0, 0.0, 300.0, 400.0))
        self.assertFalse(grid.is_blocked_world(x, y))
        self.assertLess(x, 300.0)

    def test_a_star_detours_through_gap(self) -> None:
        grid = band_map(with_gap=True)
        path = grid.find_path((50.0, 175.0), (350.0, 175.0), exact_goal=True)
        self.assertIsNotNone(path)
        assert path is not None
        for waypoint in path:
            self.assertFalse(grid.is_blocked_world(*waypoint))
        # It must cross the band through the northern gap (tiles y=2..4).
        crossed_gap = [
            (wx, wy)
            for wx, wy in path
            if 170.0 <= wx <= 230.0 and 15.0 <= wy <= 55.0
        ]
        self.assertTrue(crossed_gap, msg=f"path did not use the gap: {path}")
        straight = math.hypot(350.0 - 50.0, 175.0 - 175.0)
        self.assertGreater(grid.path_length(path, (50.0, 175.0)), straight * 1.2)

    def test_a_star_returns_none_when_no_route(self) -> None:
        grid = band_map(with_gap=False)
        self.assertIsNone(grid.find_path((50.0, 25.0), (350.0, 25.0)))

    def test_a_star_is_deterministic(self) -> None:
        grid = band_map(with_gap=True)
        first = grid.find_path((50.0, 25.0), (350.0, 175.0))
        second = grid.find_path((50.0, 25.0), (350.0, 175.0))
        self.assertEqual(first, second)
        self.assertIsNotNone(first)

    def test_a_star_exact_goal_keeps_requested_point(self) -> None:
        grid = band_map(with_gap=True)
        goal = (352.0, 27.0)
        path = grid.find_path((50.0, 25.0), goal, exact_goal=True)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[-1], goal)

    def test_a_star_path_length(self) -> None:
        grid = band_map(with_gap=True)
        path = grid.find_path((50.0, 25.0), (350.0, 25.0))
        assert path is not None
        self.assertGreater(grid.path_length(path), 0.0)
        self.assertAlmostEqual(
            grid.path_length(path, (50.0, 25.0)),
            grid.path_length(path) + math.hypot(path[0][0] - 50.0, path[0][1] - 25.0),
        )


class TerrainSpawnTests(unittest.TestCase):
    def test_terrain_config_replaces_map_size_only(self) -> None:
        grid = lake_map()
        base = MicroDuelConfig()
        config = make_terrain_config(grid, base)
        self.assertEqual(config.map_width, float(grid.width_px))
        self.assertEqual(config.map_height, float(grid.height_px))
        self.assertAlmostEqual(
            config.spawn_margin, round(grid.width_px * 0.12, 3)
        )
        # Unit stats / clock are untouched.
        self.assertEqual(config.unit_hp, base.unit_hp)
        self.assertEqual(config.unit_speed, base.unit_speed)
        self.assertEqual(config.episode_ticks, base.episode_ticks)
        # The original config object is immutable and unchanged.
        self.assertEqual(base.map_width, 1000.0)
        self.assertEqual(base.spawn_margin, 220.0)

    def test_engine_uses_terrain_dimensions(self) -> None:
        grid = lake_map()
        engine = MicroDuelEngine(terrain=grid)
        self.assertEqual(engine.config.map_width, 600.0)
        self.assertEqual(engine.config.map_height, 400.0)
        self.assertAlmostEqual(engine.config.spawn_margin, 72.0)

    def test_spawn_points_are_on_land_and_split_by_side(self) -> None:
        grid = lake_map()
        engine = MicroDuelEngine(terrain=grid, opponent="idle")
        engine.reset(seed=42)
        self.assertEqual(len(engine.units), 16)
        for unit in engine.units:
            self.assertFalse(
                grid.is_blocked_world(unit.x, unit.y),
                msg=f"unit {unit.entity_id} spawned in water at ({unit.x}, {unit.y})",
            )
        own = [unit for unit in engine.units if unit.team == 0]
        enemy = [unit for unit in engine.units if unit.team == 1]
        self.assertTrue(all(unit.x < engine.config.map_width / 2.0 for unit in own))
        self.assertTrue(all(unit.x > engine.config.map_width / 2.0 for unit in enemy))
        # Vertical dispersion: the squad is not stacked on one row.
        own_ys = [unit.y for unit in own]
        self.assertGreater(max(own_ys) - min(own_ys), 40.0)

    def test_spawn_is_deterministic_for_a_seed(self) -> None:
        grid = lake_map()
        first = MicroDuelEngine(terrain=grid, opponent="idle")
        second = MicroDuelEngine(terrain=grid, opponent="idle")
        first.reset(seed=7)
        second.reset(seed=7)
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_adapter_forwards_terrain(self) -> None:
        grid = lake_map()
        adapter = MicroDuelAdapter(config=MicroDuelConfig(), opponent="idle", terrain=grid)
        state = adapter.reset(episode_id="ep-0", seed=3)
        self.assertIs(adapter.terrain, grid)
        self.assertIs(adapter.engine.terrain, grid)
        for entity in state["own_entities"] + state["visible_enemy_entities"]:
            self.assertFalse(grid.is_blocked_world(entity["x"], entity["y"]))


class TerrainMovementTests(unittest.TestCase):
    def test_units_never_enter_water_and_cross_via_the_gap(self) -> None:
        grid = band_map(with_gap=True)
        engine = MicroDuelEngine(terrain=grid, opponent="idle")
        engine.reset(seed=11)
        actions = [
            {"kind": "attack_move", "unit_group": group, "x": 350.0, "y": 150.0}
            for group in range(engine.config.groups)
        ]
        receipts = engine.apply_actions(actions)
        self.assertTrue(all(receipt["accepted"] for receipt in receipts))
        max_x = 0.0
        for _ in range(900):
            engine.advance(1)
            for unit in engine.alive_units():
                self.assertFalse(
                    grid.is_blocked_world(unit.x, unit.y),
                    msg=f"unit {unit.entity_id} entered water at tick {engine.tick}",
                )
                max_x = max(max_x, unit.x)
                if 180.0 <= unit.x <= 220.0:
                    # The only passable band tiles are the northern gap (y=20..50).
                    self.assertGreaterEqual(unit.y, 20.0)
                    self.assertLessEqual(unit.y, 50.0)
            if all(unit.order == 0 for unit in engine.alive_units()):
                break
        self.assertGreater(max_x, 250.0, "squad never crossed the water band")

    def test_move_order_into_water_snaps_to_shore(self) -> None:
        grid = band_map(with_gap=True)
        engine = MicroDuelEngine(terrain=grid, opponent="idle")
        engine.reset(seed=5)
        receipts = engine.apply_actions(
            [{"kind": "move_group", "unit_group": 0, "x": 190.0, "y": 120.0}]
        )
        self.assertIsNone(receipts[0]["rejected_reason"])
        for member in engine.alive_units(0):
            if member.group != 0:
                continue
            self.assertFalse(grid.is_blocked_world(member.tx, member.ty))
            self.assertLessEqual(math.hypot(member.tx - 190.0, member.ty - 120.0), 20.0)

    def test_solid_wall_falls_back_to_slide_then_stops(self) -> None:
        grid = band_map(with_gap=False)
        engine = MicroDuelEngine(terrain=grid, opponent="idle")
        engine.reset(seed=9)
        engine.apply_actions(
            [
                {"kind": "attack_move", "unit_group": group, "x": 350.0, "y": 100.0}
                for group in range(engine.config.groups)
            ]
        )
        for _ in range(400):
            engine.advance(1)
        for unit in engine.units:
            self.assertFalse(grid.is_blocked_world(unit.x, unit.y))
        for unit in engine.alive_units(0):
            self.assertLess(unit.x, 185.0)  # wall starts at tile 18 -> x=180
            self.assertEqual(unit.order, 0, "stuck unit should drop its order")

    def test_attack_target_closes_in_and_fires(self) -> None:
        grid = band_map(with_gap=True)
        engine = MicroDuelEngine(terrain=grid, opponent="idle")
        engine.reset(seed=3)
        target = engine.alive_units(1)[0]
        engine.apply_actions(
            [
                {"kind": "attack_target", "unit_group": group, "target_entity_id": target.entity_id}
                for group in range(engine.config.groups)
            ]
        )
        damage_before = engine.stats["damage_dealt"][0]
        decisions = 0
        while not engine.done and decisions < 120:
            engine.step_decision([])
            decisions += 1
        self.assertGreater(engine.stats["damage_dealt"][0], damage_before)
        self.assertTrue(engine.done)
        for unit in engine.alive_units():
            self.assertFalse(grid.is_blocked_world(unit.x, unit.y))

    def test_same_seed_and_actions_are_deterministic(self) -> None:
        grid = band_map(with_gap=True)

        def run() -> MicroDuelEngine:
            engine = MicroDuelEngine(terrain=grid, opponent="scripted")
            engine.reset(seed=21)
            for _ in range(40):
                if engine.done:
                    break
                enemies = engine.alive_units(1)
                if not enemies:
                    break
                engine.step_decision(
                    [
                        {
                            "kind": "attack_target",
                            "unit_group": group,
                            "target_entity_id": enemies[0].entity_id,
                        }
                        for group in range(engine.config.groups)
                    ]
                )
            return engine

        first = run()
        second = run()
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.tick, second.tick)
        self.assertEqual(first.result, second.result)

    def test_decision_window_performance(self) -> None:
        grid = band_map(with_gap=True)
        engine = MicroDuelEngine(terrain=grid, opponent="scripted")
        engine.reset(seed=13)
        enemies = engine.alive_units(1)
        actions = [
            {"kind": "attack_target", "unit_group": group, "target_entity_id": enemies[0].entity_id}
            for group in range(engine.config.groups)
        ]
        start = time.perf_counter()
        for _ in range(10):
            engine.advance(engine.config.decision_interval)
        elapsed = time.perf_counter() - start
        # 10 decision windows = 300 ticks for 16 units; requirement: 60 tick/s.
        self.assertLess(elapsed, 1.0, msg=f"terrain simulation too slow: {elapsed:.3f}s")


class RecordTerrainIntegrationTests(unittest.TestCase):
    """The recorder worker is what the platform backend actually spawns."""

    def test_record_terrain_mode_meta_and_pixel_coordinates(self) -> None:
        from ironzero.training.record import main as record_main

        grid = band_map(with_gap=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_path = root / "map.json"
            map_path.write_text(
                json.dumps(
                    {
                        "id": grid.map_id,
                        "name": grid.name,
                        "width": grid.width,
                        "height": grid.height,
                        "tile_width": grid.tile_width,
                        "tile_height": grid.tile_height,
                        "width_px": grid.width_px,
                        "height_px": grid.height_px,
                        "blocked": list(grid.blocked),
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "replays"
            stream_path = root / "frames.jsonl"
            code = record_main(
                [
                    "--episodes",
                    "1",
                    "--mode",
                    "random",
                    "--max-decisions",
                    "3",
                    "--sim-mode",
                    "terrain",
                    "--map-json",
                    str(map_path),
                    "--out-dir",
                    str(out_dir),
                    "--stream",
                    str(stream_path),
                    "--seed",
                    "5",
                ]
            )
            self.assertEqual(code, 0)
            replay_files = sorted(out_dir.glob("*.json"))
            self.assertEqual(len(replay_files), 1)
            replay = json.loads(replay_files[0].read_text(encoding="utf-8"))
            meta = replay["meta"]
            self.assertEqual(meta["sim_mode"], "terrain")
            self.assertEqual(meta["map"]["id"], grid.map_id)
            self.assertEqual(meta["map"]["width_px"], grid.width_px)
            self.assertEqual(meta["map"]["tile_width"], grid.tile_width)
            self.assertEqual(meta["vision_radius"], 260)
            self.assertFalse(meta["fog"])
            frames = replay["frames"]
            self.assertTrue(frames)
            for frame in frames:
                self.assertIn("result", frame)
                for unit in frame["units"]:
                    self.assertFalse(grid.is_blocked_world(unit["x"], unit["y"]))
            stream_types = [
                json.loads(line)["type"] for line in stream_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(stream_types[0], "start")
            self.assertIn("frame", stream_types)
            self.assertEqual(stream_types[-1], "done")


class ArenaRegressionTests(unittest.TestCase):
    def test_default_config_is_unchanged(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.map_width, 1000.0)
        self.assertEqual(DEFAULT_CONFIG.map_height, 1000.0)
        self.assertEqual(DEFAULT_CONFIG.spawn_margin, 220.0)
        self.assertEqual(DEFAULT_CONFIG.episode_ticks, 3600)
        self.assertEqual(DEFAULT_CONFIG.unit_speed, 66.0)
        self.assertEqual(DEFAULT_CONFIG.attack_range, 130.0)

    def test_no_terrain_keeps_arena_engine_identical(self) -> None:
        first = MicroDuelEngine()
        second = MicroDuelEngine(terrain=None)
        first.reset(seed=123)
        second.reset(seed=123)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertIsNone(first.terrain)
        self.assertEqual(first.config, DEFAULT_CONFIG)
        for unit in first.units:
            self.assertTrue(0.0 <= unit.x <= 1000.0)
            self.assertTrue(0.0 <= unit.y <= 1000.0)

    def test_arena_move_behaviour_is_unchanged(self) -> None:
        engine = MicroDuelEngine(opponent="idle")
        engine.reset(seed=0)
        before = {unit.entity_id: (unit.x, unit.y) for unit in engine.units}
        engine.step_decision(
            [{"kind": "move_group", "unit_group": 0, "x": 700.0, "y": 500.0}]
        )
        moved = [
            unit.entity_id
            for unit in engine.units
            if (unit.x, unit.y) != before[unit.entity_id]
        ]
        self.assertEqual(len(moved), 2)
        for unit in engine.units:
            self.assertGreaterEqual(unit.x, 0.0)
            self.assertLessEqual(unit.x, 1000.0)
            self.assertGreaterEqual(unit.y, 0.0)
            self.assertLessEqual(unit.y, 1000.0)

    def test_terrain_grid_is_reusable_as_payload(self) -> None:
        grid = band_map()
        round_tripped = TerrainGrid.from_payload(grid)
        self.assertIs(round_tripped, grid)
        minimal = {"width": 2, "height": 2, "blocked": [1, 0, 0, 1]}
        parsed = TerrainGrid.from_payload(minimal)
        self.assertTrue(parsed.is_blocked_tile(0, 0))
        self.assertTrue(parsed.is_blocked_tile(1, 1))
        self.assertEqual(parsed.width_px, 40)  # defaults to width * tile_width


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
