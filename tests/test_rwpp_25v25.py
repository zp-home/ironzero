# SPDX-License-Identifier: Apache-2.0
"""Tests for the 25-vs-25 real-game observation contract, terrain channel,
evaluation confidence intervals and best-checkpoint bookkeeping.

Everything here runs offline: the terrain grid comes from a temporary JSON file
written in the same ``TerrainGrid``/``map.json`` format the real map pipeline
produces, and the game-side observations are synthesised bridge messages.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ironzero.mock_engine import ACTION_KINDS
from ironzero.observations import build_masks, encode_observation
from ironzero.rwpp_env import (
    RwppGameConfig,
    army_hp_reference,
    decode_action,
    launch_rwpp_process,
    observation_to_state,
)
from ironzero.terrain import TerrainGrid
from ironzero.training.evaluate import summarize, wilson_interval
from ironzero.training.policy import MaskedActorCritic
from ironzero.training.ppo import PPOConfig, PPOTrainer, load_policy
from ironzero.training.train_rwpp import build_game, build_parser

# 25 tanks per side on the fixed IronZero25 map (135x125 tiles of 20 px).
MAP_PX_W = 2700.0
MAP_PX_H = 2500.0

FULL_25 = dict(
    map_width=MAP_PX_W,
    map_height=MAP_PX_H,
    units_per_team=25,
    max_entities=25,
    target_slots=26,
)


def cfg_25(**overrides) -> MicroDuelConfig:
    fields = dict(FULL_25)
    fields.update(overrides)
    return replace(DEFAULT_CONFIG, **fields)


def bridge_state(
    own_count: int = 25,
    enemy_count: int = 25,
    *,
    own_id_base: int = 100,
    enemy_id_base: int = 200,
    map_width: float = MAP_PX_W,
    map_height: float = MAP_PX_H,
    own_hp: float = 210.0,
    enemy_hp: float = 210.0,
) -> dict:
    """Bridge-shaped state without ``group_id`` (the real bridge does not send it)."""

    def entity(entity_id: int, team: int, index: int) -> dict:
        return {
            "entity_id": entity_id,
            "team_id": team,
            "x": 200.0 + (index % 5) * 120.0,
            "y": 200.0 + (index // 5) * 200.0,
            "hp": 210.0 if team == 0 else enemy_hp,
            "max_hp": 210.0,
            "is_alive": True,
        }

    own = [entity(own_id_base + i, 0, i) for i in range(own_count)]
    enemies = [entity(enemy_id_base + i, 1, i) for i in range(enemy_count)]
    return observation_to_state(
        {
            "protocol": "ironzero/1",
            "type": "obs",
            "tick": 0,
            "game_time": 0.0,
            "controlled_team": 0,
            "global_state": {"unit_count": len(own) + len(enemies)},
            "own_entities": own,
            "visible_enemy_entities": enemies,
            "result": None,
            "terminated": False,
            "truncated": False,
        }
    )


def write_terrain_file(
    directory: Path,
    *,
    width_tiles: int = 8,
    height_tiles: int = 8,
    tile_px: int = 10,
    blocked_columns: int = 4,
) -> Path:
    """Left ``blocked_columns`` tile columns blocked (water), the rest passable."""

    blocked = [
        1 if tx < blocked_columns else 0
        for ty in range(height_tiles)
        for tx in range(width_tiles)
    ]
    payload = {
        "id": "test-terrain",
        "name": "Test Terrain",
        "width": width_tiles,
        "height": height_tiles,
        "tile_width": tile_px,
        "tile_height": tile_px,
        "width_px": width_tiles * tile_px,
        "height_px": height_tiles * tile_px,
        "blocked": blocked,
    }
    path = directory / "IronZero25.terrain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Observation / action contract for 25 units
# ----------------------------------------------------------------------
class MicroConfig25Tests(unittest.TestCase):
    def test_25v25_observation_and_action_contract(self) -> None:
        cfg = cfg_25()
        # 15 global (12 legacy + 3 economy) + 2 * (25 * 10) entity + 3 * 8 * 8 grid.
        self.assertEqual(cfg.global_dim, 15)
        self.assertEqual(
            cfg.flat_obs_dim,
            cfg.global_dim
            + 2 * cfg.max_entities * cfg.entity_features
            + cfg.map_channels * cfg.map_grid**2,
        )
        self.assertEqual(cfg.flat_obs_dim, 15 + 500 + 192)
        # Five heads; the kind head spans the whole ACTION_KINDS vocabulary.
        self.assertEqual(len(cfg.action_nvec), 5)
        self.assertEqual(cfg.action_nvec, (len(ACTION_KINDS), cfg.groups + 1, 8, 8, 26))
        self.assertEqual(cfg.action_nvec[0], 10)
        self.assertEqual(cfg.max_entities, 25)
        self.assertEqual(cfg.target_slots, 26)

    def test_terrain_channel_adds_one_map_grid(self) -> None:
        cfg = cfg_25(map_channels=4, terrain_path="configs/maps/IronZero25.terrain.json")
        self.assertEqual(cfg.flat_obs_dim, 15 + 500 + 4 * 64)
        self.assertEqual(
            cfg.flat_obs_dim,
            cfg_25().flat_obs_dim + cfg.map_grid**2,  # exactly one extra map grid
        )
        self.assertTrue(cfg.terrain_channel)

    def test_defaults_stay_legacy_8_units(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.units_per_team, 8)
        self.assertEqual(DEFAULT_CONFIG.max_entities, 8)
        self.assertEqual(DEFAULT_CONFIG.target_slots, 9)
        self.assertEqual(DEFAULT_CONFIG.map_channels, 3)
        self.assertFalse(DEFAULT_CONFIG.terrain_channel)
        self.assertEqual(DEFAULT_CONFIG.global_dim, 15)
        self.assertEqual(DEFAULT_CONFIG.flat_obs_dim, 15 + 2 * 80 + 192)


class Observation25Tests(unittest.TestCase):
    def test_encoder_fills_every_entity_slot(self) -> None:
        cfg = cfg_25()
        state = bridge_state()
        obs = encode_observation(state, cfg)
        self.assertEqual(obs.shape, (cfg.flat_obs_dim,))
        offset = cfg.global_dim
        own_block = obs[offset : offset + 250].reshape(25, cfg.entity_features)
        offset += 250
        enemy_block = obs[offset : offset + 250].reshape(25, cfg.entity_features)
        self.assertEqual(int((own_block[:, 0] > 0.5).sum()), 25)
        self.assertEqual(int((enemy_block[:, 0] > 0.5).sum()), 25)
        # Global army fractions see all 25 units on both sides.
        self.assertAlmostEqual(float(obs[1]), 1.0)
        self.assertAlmostEqual(float(obs[2]), 1.0)
        self.assertAlmostEqual(float(obs[3]), 1.0)
        self.assertAlmostEqual(float(obs[4]), 1.0)

    def test_masks_without_group_id_expose_all_groups(self) -> None:
        cfg = cfg_25()
        masks = build_masks(bridge_state(), cfg)
        self.assertEqual(masks["group"].shape, (cfg.groups + 1,))
        self.assertTrue(bool(masks["group"].all()))
        self.assertEqual(masks["target"].shape, (26,))
        self.assertEqual(int(masks["target"].sum()), 26)  # 25 enemies + no-target
        self.assertTrue(bool(masks["target"][-1]))
        # Bridge units carry no builder and no funds, so the kind head exposes the
        # five tactical orders and gates the four economy kinds. The mask must
        # still be sized to the full action vocabulary.
        self.assertEqual(masks["kind"].shape, (cfg.action_nvec[0],))
        self.assertEqual(masks["kind"].shape, (len(ACTION_KINDS),))
        self.assertEqual(int(masks["kind"].sum()), 5)
        self.assertTrue(bool(masks["kind"][:5].all()))
        self.assertFalse(bool(masks["kind"][5:].any()))

    def test_masks_from_mock_engine_still_derive_groups_from_group_id(self) -> None:
        # Offline mock units carry group_id: the mask must reflect only the
        # groups that actually exist (0 and 2 here), plus broadcast.
        state = {
            "own_entities": [
                {"entity_id": 1, "hp": 210.0, "max_hp": 210.0, "group_id": 0, "x": 10.0, "y": 10.0},
                {"entity_id": 2, "hp": 210.0, "max_hp": 210.0, "group_id": 2, "x": 20.0, "y": 20.0},
            ],
            "visible_enemy_entities": [],
        }
        masks = build_masks(state, cfg_25())
        self.assertEqual(list(masks["group"]), [True, False, True, False, True])

    def test_entity_group_feature_infers_id_mod_groups(self) -> None:
        cfg = cfg_25()
        state = bridge_state(own_count=4)
        obs = encode_observation(state, cfg)
        own_block = obs[cfg.global_dim : cfg.global_dim + 250].reshape(25, cfg.entity_features)
        groups = [int(round(own_block[index, 4] * (cfg.groups - 1))) for index in range(4)]
        self.assertEqual(groups, [index % cfg.groups for index in range(4)])

    def test_target_mask_covers_all_25_enemy_slots(self) -> None:
        cfg = cfg_25()
        masks = build_masks(bridge_state(enemy_count=25), cfg)
        self.assertEqual(int(masks["target"][:25].sum()), 25)
        self.assertEqual(int(masks["target"].sum()), 26)
        # A stale slot beyond the visible enemies stays masked.
        fewer = build_masks(bridge_state(enemy_count=3), cfg)
        self.assertEqual(int(fewer["target"][:3].sum()), 3)
        self.assertFalse(bool(fewer["target"][3]))


class DecodeAction25Tests(unittest.TestCase):
    def test_25_enemy_target_slots_decode_to_entity_ids(self) -> None:
        cfg = cfg_25()
        state = bridge_state()
        ids = [int(entity["entity_id"]) for entity in state["visible_enemy_entities"]]
        self.assertEqual(len(ids), 25)
        first = decode_action(np.array([2, 4, 0, 0, 0]), state, cfg)
        self.assertEqual(first[0]["target_entity_id"], ids[0])
        last = decode_action(np.array([2, 4, 0, 0, 24]), state, cfg)
        self.assertEqual(last[0]["target_entity_id"], ids[24])
        # Slot 25 is the "no target" fallback -> safe hold.
        no_target = decode_action(np.array([2, 4, 0, 0, 25]), state, cfg)
        self.assertEqual(no_target[0]["kind"], "hold")

    def test_move_action_projects_onto_full_map(self) -> None:
        cfg = cfg_25()
        state = bridge_state()
        move = decode_action(np.array([0, 4, 7, 7, 0]), state, cfg)
        self.assertEqual(move[0]["kind"], "move_group")
        self.assertGreaterEqual(move[0]["x"], 0.0)
        self.assertLessEqual(move[0]["x"], MAP_PX_W)
        self.assertLessEqual(move[0]["y"], MAP_PX_H)


# ----------------------------------------------------------------------
# Terrain observation channel
# ----------------------------------------------------------------------
class TerrainChannelTests(unittest.TestCase):
    def test_passable_channel_is_downsampled_into_channel_3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_terrain_file(Path(tmp))
            cfg = cfg_25(
                map_width=80.0,
                map_height=80.0,
                map_channels=4,
                terrain_path=str(path),
            )
            obs = encode_observation(bridge_state(own_count=0, enemy_count=0), cfg)
            offset = cfg.global_dim + 2 * cfg.max_entities * cfg.entity_features
            grid = obs[offset:].reshape(cfg.map_channels, cfg.map_grid, cfg.map_grid)
            # Left 4 of 8 tile columns are blocked -> left 4 observation cells 0,
            # right 4 cells fully passable.
            np.testing.assert_allclose(grid[3][:, :4], 0.0)
            np.testing.assert_allclose(grid[3][:, 4:], 1.0)
            # Army channels stay normalised by army size (not by the terrain scale).
            self.assertAlmostEqual(float(grid[0].sum()), 0.0)

    def test_army_channels_remain_normalised_with_terrain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_terrain_file(Path(tmp))
            cfg = cfg_25(
                map_width=80.0,
                map_height=80.0,
                map_channels=4,
                terrain_path=str(path),
            )
            state = {
                "own_entities": [
                    {"entity_id": 1, "x": 10.0, "y": 10.0, "hp": 210.0, "max_hp": 210.0}
                ],
                "visible_enemy_entities": [],
            }
            obs = encode_observation(state, cfg)
            offset = cfg.global_dim + 2 * cfg.max_entities * cfg.entity_features
            grid = obs[offset:].reshape(cfg.map_channels, cfg.map_grid, cfg.map_grid)
            self.assertAlmostEqual(float(grid[0].sum()), 1.0 / 25.0)
            self.assertAlmostEqual(float(grid[0].max()), 1.0 / 25.0)

    def test_missing_terrain_warns_and_keeps_channel_zero(self) -> None:
        cfg = cfg_25(
            map_channels=4,
            terrain_path="does/not/exist/IronZero25.terrain.json",
        )
        state = bridge_state(own_count=0, enemy_count=0)
        with self.assertWarns(RuntimeWarning):
            obs = encode_observation(state, cfg)
        offset = cfg.global_dim + 2 * cfg.max_entities * cfg.entity_features
        grid = obs[offset:].reshape(cfg.map_channels, cfg.map_grid, cfg.map_grid)
        self.assertEqual(float(np.abs(grid[3]).sum()), 0.0)

    def test_legacy_three_channel_config_has_no_terrain_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_terrain_file(Path(tmp))
            cfg = cfg_25(terrain_path=str(path))  # map_channels stays 3
            self.assertFalse(cfg.terrain_channel)
            obs = encode_observation(bridge_state(own_count=0, enemy_count=0), cfg)
            self.assertEqual(obs.shape, (cfg.flat_obs_dim,))
            self.assertEqual(cfg.flat_obs_dim, 15 + 500 + 192)

    def test_terrain_from_file_accepts_nested_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.terrain.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "nested",
                        "width_px": 80,
                        "height_px": 80,
                        "terrain": {
                            "width": 8,
                            "height": 8,
                            "tile_width": 10,
                            "tile_height": 10,
                            "blocked": [1] * 8 + [0] * 56,
                        },
                    }
                ),
                encoding="utf-8",
            )
            grid = TerrainGrid.from_file(str(path))
            self.assertEqual((grid.width, grid.height), (8, 8))
            self.assertEqual(grid.width_px, 80)
            self.assertEqual(grid.land_ratio, 56 / 64)

    def test_real_ironzero25_terrain_contract(self) -> None:
        """Lock the agreed map-engineer JSON schema (skip when the map is absent)."""

        path = Path(__file__).resolve().parents[1] / "configs" / "maps" / "IronZero25.terrain.json"
        if not path.is_file():
            self.skipTest("configs/maps/IronZero25.terrain.json not generated yet")
        grid = TerrainGrid.from_file(str(path))
        self.assertEqual((grid.width, grid.height), (135, 125))
        self.assertEqual((grid.tile_width, grid.tile_height), (20, 20))
        self.assertEqual((grid.width_px, grid.height_px), (2700, 2500))
        self.assertEqual(grid.tile_count, 16875)
        self.assertAlmostEqual(grid.land_ratio, 0.4718, places=3)
        game = RwppGameConfig(
            units_per_team=25,
            max_entities=25,
            target_slots=26,
            terrain_path=path,
            map_width_px=2700.0,
            map_height_px=2500.0,
        )
        micro = game.micro_config()
        self.assertEqual(micro.map_channels, 4)
        self.assertEqual(micro.flat_obs_dim, 771)
        self.assertEqual(
            micro.flat_obs_dim,
            15 + 2 * 25 * 10 + 4 * 8 * 8,  # 15 global + entity blocks + 4 map grids
        )
        # The passability prior describes the same land ratio as the raw map.
        obs = encode_observation(bridge_state(own_count=0, enemy_count=0), micro)
        offset = micro.global_dim + 2 * micro.max_entities * micro.entity_features
        terrain = obs[offset:].reshape(micro.map_channels, micro.map_grid, micro.map_grid)[3]
        self.assertAlmostEqual(float(terrain.mean()), 0.4718, places=2)


# ----------------------------------------------------------------------
# RwppGameConfig: 25-vs-25 + terrain resolution
# ----------------------------------------------------------------------
class RwppGameConfig25Tests(unittest.TestCase):
    def test_game_config_builds_25v25_terrain_micro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_terrain_file(Path(tmp))
            game = RwppGameConfig(
                units_per_team=25,
                max_entities=25,
                target_slots=26,
                terrain_path=path,
                map_width_px=MAP_PX_W,
                map_height_px=MAP_PX_H,
            )
            micro = game.micro_config()
            self.assertEqual(micro.units_per_team, 25)
            self.assertEqual(micro.max_entities, 25)
            self.assertEqual(micro.target_slots, 26)
            self.assertEqual(micro.map_channels, 4)
            self.assertEqual(micro.flat_obs_dim, 15 + 500 + 256)
            self.assertEqual(micro.action_nvec, (len(ACTION_KINDS), 5, 8, 8, 26))
            self.assertEqual(Path(str(micro.terrain_path)), path.resolve())
            # The contract survives a checkpoint asdict/rebuild round trip.
            rebuilt = MicroDuelConfig(**asdict(micro))
            self.assertEqual(rebuilt.flat_obs_dim, micro.flat_obs_dim)
            self.assertEqual(rebuilt.action_nvec, micro.action_nvec)

    def test_missing_terrain_warns_and_keeps_legacy_channels(self) -> None:
        game = RwppGameConfig(
            units_per_team=25,
            max_entities=25,
            terrain_path=Path("missing/IronZero25.terrain.json"),
        )
        with self.assertWarns(RuntimeWarning):
            micro = game.micro_config()
        self.assertEqual(micro.map_channels, 3)
        self.assertIsNone(micro.terrain_path)
        self.assertEqual(micro.flat_obs_dim, 15 + 500 + 192)

    def test_target_slots_auto_and_entity_floor(self) -> None:
        # target_slots=0 -> max_entities + 1; max_entities below the army is lifted.
        micro = RwppGameConfig(units_per_team=25, max_entities=1).micro_config()
        self.assertEqual(micro.max_entities, 25)
        self.assertEqual(micro.target_slots, 26)
        # Defaults still describe the legacy 8-unit scenario.
        legacy = RwppGameConfig().micro_config()
        self.assertEqual(legacy.flat_obs_dim, DEFAULT_CONFIG.flat_obs_dim)
        self.assertEqual(legacy.action_nvec, DEFAULT_CONFIG.action_nvec)


# ----------------------------------------------------------------------
# Fog reference helper
# ----------------------------------------------------------------------
class ArmyHpReferenceTests(unittest.TestCase):
    def test_uses_ground_truth_totals_not_visible_subset(self) -> None:
        cfg = replace(DEFAULT_CONFIG, vision_radius=260.0)
        state = bridge_state(own_count=2, enemy_count=3)
        own_ref, enemy_ref = army_hp_reference(state, cfg)
        self.assertAlmostEqual(own_ref, 420.0)
        self.assertAlmostEqual(enemy_ref, 630.0)

    def test_falls_back_to_configured_army_when_stream_is_empty(self) -> None:
        cfg = cfg_25()
        own_ref, enemy_ref = army_hp_reference(
            {"own_entities": [], "visible_enemy_entities": []}, cfg
        )
        self.assertAlmostEqual(own_ref, 25 * 210.0)
        self.assertAlmostEqual(enemy_ref, 25 * 210.0)


# ----------------------------------------------------------------------
# Evaluation confidence intervals
# ----------------------------------------------------------------------
class WilsonIntervalTests(unittest.TestCase):
    def test_interval_bounds_are_sane_for_small_samples(self) -> None:
        low, high = wilson_interval(0, 8)
        self.assertAlmostEqual(low, 0.0)
        self.assertGreater(high, 0.25)
        self.assertLess(high, 0.40)
        low, high = wilson_interval(8, 8)
        self.assertAlmostEqual(high, 1.0)
        self.assertLess(low, 0.75)
        self.assertGreater(low, 0.55)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_summarize_reports_ci_and_counts(self) -> None:
        report = summarize(
            ["victory", "victory", "defeat", "timeout"],
            [1.0, 1.0, -1.0, 0.0],
            [0.5, 0.4, 0.0, 0.2],
            [0.0, 0.0, 0.7, 0.5],
            [10, 12, 30, 20],
        )
        self.assertEqual(report["wins"], 2)
        self.assertEqual(report["losses"], 1)
        self.assertEqual(report["timeouts"], 1)
        self.assertAlmostEqual(report["win_rate"], 0.5)
        low, high = report["win_rate_ci95"]
        self.assertLess(low, high)
        self.assertLessEqual(low, report["win_rate"])
        self.assertGreaterEqual(high, report["win_rate"])
        self.assertTrue(0.0 <= low <= 1.0 and 0.0 <= high <= 1.0)


# ----------------------------------------------------------------------
# Training CLI
# ----------------------------------------------------------------------
class TrainCliTests(unittest.TestCase):
    def test_ironzero25_flags_build_the_25v25_terrain_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "--map",
                "maps/ironzero/IronZero25.tmx",
                "--map-name",
                "IronZero25",
                "--map-width",
                "2700",
                "--map-height",
                "2500",
                "--units-per-team",
                "25",
                "--max-entities",
                "25",
                "--target-slots",
                "26",
                "--terrain",
                "configs/maps/IronZero25.terrain.json",
                "--win-mode",
                "annihilation",
                "--unlock-ai",
                "--eval-episodes",
                "4",
            ]
        )
        game = build_game(args)
        micro = game.micro_config()
        self.assertEqual(micro.units_per_team, 25)
        self.assertEqual(micro.max_entities, 25)
        self.assertEqual(micro.target_slots, 26)
        self.assertEqual(micro.map_channels, 4)
        self.assertEqual(micro.flat_obs_dim, 771)
        self.assertEqual(micro.action_nvec, (len(ACTION_KINDS), 5, 8, 8, 26))
        self.assertEqual(game.win_mode, "annihilation")
        self.assertTrue(game.unlock_ai)
        self.assertTrue(micro.terrain_path)

    def test_default_cli_stays_legacy_8_unit_scenario(self) -> None:
        game = build_game(build_parser().parse_args([]))
        micro = game.micro_config()
        self.assertEqual(micro.units_per_team, 8)
        self.assertEqual(micro.max_entities, 8)
        self.assertEqual(micro.target_slots, 9)
        self.assertEqual(micro.map_channels, 3)
        self.assertEqual(micro.flat_obs_dim, DEFAULT_CONFIG.flat_obs_dim)
        self.assertEqual(game.win_mode, "")
        self.assertFalse(game.unlock_ai)

    def test_main_writes_summary_without_launching_games(self) -> None:
        """Exercise the CLI wiring end-to-end with a fake trainer (no real game)."""

        from types import SimpleNamespace

        from ironzero.training import train_rwpp

        captured: dict = {}

        class FakeVec:
            def __init__(self) -> None:
                self.envs = [SimpleNamespace(strategy=None)]

            def close(self) -> None:
                pass

        class FakeTrainer:
            def __init__(self, config, **kwargs):
                captured["config"] = config
                captured.update(kwargs)
                self.config = config
                self.device = "cpu"
                self.vec = FakeVec()

            def train(self):
                evaluation = {
                    "update": 1,
                    "win_rate": 0.5,
                    "loss_rate": 0.25,
                    "timeout_rate": 0.25,
                    "mean_return": 0.1,
                    "mean_own_hp_frac": 0.6,
                }
                return {
                    "updates": [{"update": 1, "episodes": 2, "win_rate": 0.5}],
                    "evaluations": [dict(evaluation)],
                    "final_eval": dict(evaluation),
                    "best_eval": dict(evaluation),
                    "best_checkpoint": "runs/fake/checkpoint_best.pt",
                }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            argv = [
                "--map",
                "maps/ironzero/IronZero25.tmx",
                "--map-name",
                "IronZero25",
                "--units-per-team",
                "25",
                "--max-entities",
                "25",
                "--target-slots",
                "26",
                "--terrain",
                "configs/maps/IronZero25.terrain.json",
                "--win-mode",
                "annihilation",
                "--unlock-ai",
                "--out",
                str(out),
                "--live",
                str(out / "live.jsonl"),
            ]
            with mock.patch.object(train_rwpp, "RwppPPOTrainer", FakeTrainer), mock.patch.object(
                train_rwpp, "_build_strategy", lambda _use_llm: None
            ):
                code = train_rwpp.main(argv)
            self.assertEqual(code, 0)
            summary = json.loads((out / "rwpp_summary.json").read_text(encoding="utf-8"))
            micro_payload = summary["micro"]
            self.assertEqual(micro_payload["units_per_team"], 25)
            self.assertEqual(micro_payload["map_channels"], 4)
            obs_dim = (
                micro_payload["global_dim"]
                + 2 * micro_payload["max_entities"] * micro_payload["entity_features"]
                + micro_payload["map_channels"] * micro_payload["map_grid"] ** 2
            )
            self.assertEqual(obs_dim, 771)
            self.assertEqual(summary["game"]["win_mode"], "annihilation")
            self.assertTrue(summary["game"]["unlock_ai"])
            self.assertEqual(summary["best_checkpoint"], "runs/fake/checkpoint_best.pt")
        self.assertIsNotNone(captured.get("game"))
        self.assertEqual(captured["game"].units_per_team, 25)
        self.assertEqual(captured["config"].save_best, True)


# ----------------------------------------------------------------------
# Best checkpoint
# ----------------------------------------------------------------------
class BestCheckpointTests(unittest.TestCase):
    def test_training_saves_best_checkpoint_and_loads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PPOConfig(
                total_updates=2,
                num_envs=2,
                rollout_steps=4,
                epochs=1,
                minibatch_size=8,
                hidden=64,
                eval_every=1,
                eval_episodes=1,
                device="cpu",
                out_dir=tmp,
                seed=3,
            )
            trainer = PPOTrainer(config)
            metrics = trainer.train()
            out = Path(tmp)
            self.assertTrue((out / "checkpoint.pt").exists())
            self.assertTrue((out / "checkpoint_best.pt").exists())
            self.assertIsNotNone(metrics["best_eval"])
            self.assertEqual(metrics["best_checkpoint"], str(out / "checkpoint_best.pt"))
            policy, micro = load_policy(out / "checkpoint_best.pt", device="cpu")
            # Hidden width comes from the checkpoint, not the 256 default.
            self.assertEqual(policy.trunk[0].out_features, 64)
            self.assertEqual(micro.flat_obs_dim, DEFAULT_CONFIG.flat_obs_dim)

    def test_terrain_checkpoint_rebuilds_the_771_dim_policy(self) -> None:
        terrain = Path(__file__).resolve().parents[1] / "configs" / "maps" / "IronZero25.terrain.json"
        if not terrain.is_file():
            self.skipTest("IronZero25 terrain not generated yet")
        micro = RwppGameConfig(
            units_per_team=25,
            max_entities=25,
            target_slots=26,
            terrain_path=terrain,
            map_width_px=2700.0,
            map_height_px=2500.0,
        ).micro_config()
        policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_best.pt"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "micro_config": asdict(micro),
                    "ppo_config": {"hidden": 128},
                    "update": 7,
                },
                path,
            )
            loaded, loaded_micro = load_policy(path, device="cpu")
        self.assertEqual(loaded_micro.flat_obs_dim, 771)
        self.assertEqual(loaded_micro.action_nvec, (len(ACTION_KINDS), 5, 8, 8, 26))
        self.assertEqual(loaded.trunk[0].in_features, 771)
        self.assertEqual(loaded.trunk[0].out_features, 128)
        self.assertEqual(loaded.nvec, (len(ACTION_KINDS), 5, 8, 8, 26))
        self.assertEqual(loaded_micro.terrain_path, str(terrain.resolve()))

    def test_legacy_768_dim_checkpoint_is_migrated_to_the_current_contract(self) -> None:
        """An older checkpoint must load, not crash, after the contract grew."""

        terrain = Path(__file__).resolve().parents[1] / "configs" / "maps" / "IronZero25.terrain.json"
        if not terrain.is_file():
            self.skipTest("IronZero25 terrain not generated yet")
        micro = RwppGameConfig(
            units_per_team=25,
            max_entities=25,
            target_slots=26,
            terrain_path=terrain,
            map_width_px=2700.0,
            map_height_px=2500.0,
        ).micro_config()
        # The pre-upgrade contract: 12 global features and a 5-wide kind head.
        # An old checkpoint does not carry the ``global_dim`` field at all, so
        # rebuilding it falls back to today's default (15).
        legacy_micro = replace(micro, global_dim=12)
        self.assertEqual(legacy_micro.flat_obs_dim, 768)
        legacy = MaskedActorCritic(
            legacy_micro.flat_obs_dim, nvec=(5, 5, 8, 8, 26), hidden=128
        )
        legacy_payload = asdict(legacy_micro)
        legacy_payload.pop("global_dim")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "model": legacy.state_dict(),
                    "micro_config": legacy_payload,
                    "ppo_config": {"hidden": 128},
                    "update": 3,
                },
                path,
            )
            # ``simplefilter("always")`` defeats Python's per-location
            # "once" de-duplication: another test in the session may already
            # have emitted this warning from the same source line.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded, loaded_micro = load_policy(path, device="cpu")
        messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("obs_dim=768" in message for message in messages),
            f"expected an obs_dim migration warning, got {messages}",
        )
        self.assertEqual(loaded_micro.flat_obs_dim, 771)
        self.assertEqual(loaded.trunk[0].in_features, 771)
        self.assertEqual(loaded.nvec, (len(ACTION_KINDS), 5, 8, 8, 26))
        # The migrated trunk keeps the legacy leading columns.
        self.assertTrue(
            torch.equal(
                loaded.trunk[0].weight[:, :768], legacy.trunk[0].weight
            )
        )

    def test_maybe_save_best_only_on_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = PPOTrainer(
                PPOConfig(total_updates=1, num_envs=1, rollout_steps=2, device="cpu", out_dir=tmp)
            )
            out = Path(tmp)
            first = trainer.maybe_save_best(out, 1, {"win_rate": 0.2, "mean_return": -0.5})
            self.assertTrue(first)
            payload = torch.load(out / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["update"], 1)
            self.assertEqual(payload["best_eval"]["win_rate"], 0.2)
            worse = trainer.maybe_save_best(out, 2, {"win_rate": 0.1, "mean_return": 9.0})
            self.assertFalse(worse)
            better = trainer.maybe_save_best(out, 3, {"win_rate": 0.3, "mean_return": -0.9})
            self.assertTrue(better)
            self.assertEqual(trainer.best_update, 3)


# ----------------------------------------------------------------------
# Bridge command line pass-through
# ----------------------------------------------------------------------
class LaunchCommandTests(unittest.TestCase):
    def _runtime(self, tmp: Path) -> Path:
        runtime = tmp / "runtime"
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin" / "rwpp-server.bat").write_text("@echo off\n", encoding="utf-8")
        return runtime

    def test_win_mode_and_unlock_ai_are_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runtime = self._runtime(base)
            game = RwppGameConfig(
                runtime=runtime,
                java_home=base,
                log_dir=base / "logs",
                win_mode="annihilation",
                unlock_ai=True,
                episode_ticks=900,
            )
            with mock.patch("ironzero.rwpp_env.subprocess.Popen") as popen:
                popen.return_value = mock.MagicMock()
                launch_rwpp_process(game, 5425)
            command = list(popen.call_args.args[0])
            self.assertIn("--win-mode", command)
            self.assertEqual(command[command.index("--win-mode") + 1], "annihilation")
            self.assertIn("--unlock-ai", command)
            self.assertEqual(command[command.index("--ticks") + 1], "900")

    def test_legacy_default_command_has_no_bridge_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runtime = self._runtime(base)
            game = RwppGameConfig(runtime=runtime, java_home=base, log_dir=base / "logs")
            with mock.patch("ironzero.rwpp_env.subprocess.Popen") as popen:
                popen.return_value = mock.MagicMock()
                launch_rwpp_process(game, 5426)
            command = list(popen.call_args.args[0])
            self.assertNotIn("--win-mode", command)
            self.assertNotIn("--unlock-ai", command)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
