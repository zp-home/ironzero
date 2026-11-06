# SPDX-License-Identifier: Apache-2.0
"""Tests for the real-game asset reader and calibration."""

from __future__ import annotations

from pathlib import Path
import unittest

from ironzero.assets import (
    AssetError,
    GAME_TICK_RATE,
    GameAssets,
    parse_tmx_header,
    parse_unit_ini,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_game_assets"


class UnitParserTests(unittest.TestCase):
    def test_parse_tank_ini(self) -> None:
        unit = parse_unit_ini(FIXTURE / "assets" / "units" / "tanks" / "tank.ini", key="tank")
        self.assertEqual(unit.key, "tank")
        self.assertEqual(unit.display_name, "tank")
        self.assertEqual(unit.price, 350.0)
        self.assertEqual(unit.max_hp, 210.0)
        self.assertEqual(unit.radius, 11.0)
        self.assertEqual(unit.attack_range, 130.0)
        self.assertEqual(unit.shoot_delay_ticks, 75)
        self.assertEqual(unit.direct_damage, 25.0)
        self.assertEqual(unit.move_speed_per_tick, 1.1)
        self.assertAlmostEqual(unit.move_speed_per_second, 1.1 * GAME_TICK_RATE)
        self.assertAlmostEqual(unit.dps, 25.0 * GAME_TICK_RATE / 75)
        payload = unit.to_dict()
        self.assertEqual(payload["source"], str(FIXTURE / "assets" / "units" / "tanks" / "tank.ini"))

    def test_parse_map_header(self) -> None:
        info = parse_tmx_header(FIXTURE / "assets" / "maps" / "skirmish" / "[p2]Lake (2p).tmx")
        self.assertEqual(info.width_tiles, 130)
        self.assertEqual(info.height_tiles, 130)
        self.assertEqual(info.tile_width, 20)
        self.assertEqual(info.width_px, 2600)
        self.assertEqual(len(info.sha256), 64)


class GameAssetsTests(unittest.TestCase):
    def test_game_info_and_lookup(self) -> None:
        assets = GameAssets(FIXTURE)
        self.assertEqual(assets.info.version, "1.15")
        self.assertEqual(assets.info.game_code, 176)
        self.assertEqual(assets.info.build_number, "28")
        self.assertIn("tank", assets.list_units())
        self.assertTrue(assets.unit_ini_path("tank").is_file())
        self.assertEqual(assets.map_info().name, "[p2]Lake (2p).tmx")

    def test_calibrated_config_matches_real_tank(self) -> None:
        assets = GameAssets(FIXTURE)
        config = assets.calibrated_micro_config()
        self.assertEqual(config.tick_rate, 60)
        self.assertEqual(config.decision_interval, 30)
        self.assertEqual(config.episode_ticks, 3600)
        self.assertEqual(config.unit_hp, 210.0)
        self.assertEqual(config.unit_damage, 25.0)
        self.assertEqual(config.attack_range, 130.0)
        self.assertEqual(config.attack_cooldown_ticks, 75)
        self.assertAlmostEqual(config.unit_speed, 66.0)
        self.assertEqual(config.unit_price, 350.0)
        self.assertEqual(config.map_width, 1000.0)

    def test_manifest_sections(self) -> None:
        manifest = GameAssets(FIXTURE).manifest()
        self.assertEqual(manifest["schema"], "ironzero.game_assets/1")
        self.assertEqual(manifest["game"]["version"], "1.15")
        self.assertEqual(manifest["game"]["build_number"], "28")
        self.assertEqual(manifest["unit"]["max_hp"], 210.0)
        self.assertEqual(manifest["map"]["width_px"], 2600)
        self.assertEqual(manifest["micro_config"]["attack_cooldown_ticks"], 75)

    def test_missing_assets_raise(self) -> None:
        with self.assertRaises(AssetError):
            GameAssets(Path(__file__).parent / "does_not_exist")

    def test_missing_unit_raises(self) -> None:
        with self.assertRaises(AssetError):
            GameAssets(FIXTURE).unit("no_such_unit")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
