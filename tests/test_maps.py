# SPDX-License-Identifier: Apache-2.0
"""Tests for the real-map parser used by the sandbox renderer."""

from __future__ import annotations

from pathlib import Path
import unittest

from web.backend import maps as gamemaps

FIXTURE = Path(__file__).parent / "fixtures" / "mini_game_cache"


class MapParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._original_root = gamemaps.ASSETS_ROOT
        gamemaps.ASSETS_ROOT = FIXTURE.resolve()

    @classmethod
    def tearDownClass(cls) -> None:
        gamemaps.ASSETS_ROOT = cls._original_root

    def test_assets_available(self) -> None:
        self.assertTrue(gamemaps.assets_available())

    def test_map_id_roundtrip(self) -> None:
        relative = "maps/skirmish/[p1]tiny map.tmx"
        map_id = gamemaps.map_id_for(relative)
        self.assertEqual(gamemaps.relative_for(map_id), relative)
        self.assertNotIn("/", map_id)

    def test_list_maps(self) -> None:
        items = gamemaps.list_maps()
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["name"], "[p1]tiny map")
        self.assertEqual(item["width_tiles"], 6)
        self.assertEqual(item["tile_width"], 20)
        self.assertEqual(item["width_px"], 120)
        self.assertFalse(item["broken"])

    def test_load_map_layers_and_tilesets(self) -> None:
        map_id = gamemaps.map_id_for("maps/skirmish/[p1]tiny map.tmx")
        payload = gamemaps.load_map(map_id)
        self.assertEqual(payload["width"], 6)
        self.assertEqual(len(payload["layers"]), 2)
        ground = next(layer for layer in payload["layers"] if layer["name"] == "Ground")
        self.assertEqual(len(ground["data"]), 36)
        self.assertEqual(ground["data"][0], 1)  # left column terrain
        self.assertEqual(ground["data"][3], 3)  # water column
        self.assertEqual(ground["data"][7], 2)  # external tsx grass
        # Embedded tileset image with a bogus "terrain/" prefix must still resolve.
        images = {tileset["name"]: tileset["image"] for tileset in payload["tilesets"]}
        self.assertIn("Tiny Terrain", images)
        self.assertTrue(images["Tiny Terrain"].endswith("tiny_terrain.png"))
        self.assertTrue(images["Tiny Grass"].endswith("tiny_grass.png"))
        for tileset in payload["tilesets"]:
            self.assertTrue((FIXTURE / tileset["image"]).is_file(), tileset["image"])

    def test_water_detection_and_battle_crop(self) -> None:
        map_id = gamemaps.map_id_for("maps/skirmish/[p1]tiny map.tmx")
        payload = gamemaps.load_map(map_id)
        water = next(t for t in payload["tilesets"] if t["name"] == "Water")
        self.assertTrue(water["water"])
        self.assertGreater(payload["land_ratio"], 0.0)
        self.assertLess(payload["land_ratio"], 1.0)
        crop = payload["crop"]
        self.assertGreaterEqual(crop["x"], 0)
        self.assertGreaterEqual(crop["y"], 0)
        self.assertLessEqual(crop["x"] + crop["width"], payload["width_px"])
        self.assertLessEqual(crop["y"] + crop["height"], payload["height_px"])

    def test_unit_sprites(self) -> None:
        sprites = gamemaps.unit_sprites()
        self.assertIn("tank", sprites)
        self.assertIn("tank_turret", sprites)
        self.assertTrue((FIXTURE / sprites["tank"]).is_file())

    def test_resolve_asset_blocks_traversal(self) -> None:
        with self.assertRaises(gamemaps.MapError):
            gamemaps.resolve_asset("../outside.png")
        self.assertTrue(gamemaps.resolve_asset("units/tanks/tank.png").is_file())

    def test_unknown_map_raises(self) -> None:
        with self.assertRaises(gamemaps.MapError):
            gamemaps.load_map(gamemaps.map_id_for("maps/nope.tmx"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
