# SPDX-License-Identifier: Apache-2.0
"""Tests for the evaluation / play CLI scripts (no real game is launched).

The scripts live outside the importable ``ironzero`` package, so they are loaded
by path. Everything tested here runs offline: only argument plumbing, checkpoint
loading and the observation-layout guard are exercised.
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from ironzero.micro_config import DEFAULT_CONFIG
from ironzero.rwpp_env import RwppGameConfig
from ironzero.training.policy import MaskedActorCritic

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAIN = REPO_ROOT / "configs" / "maps" / "IronZero25.terrain.json"


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_checkpoint(path: Path, micro, *, hidden: int = 32) -> None:
    policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=hidden)
    torch.save(
        {
            "model": policy.state_dict(),
            "micro_config": asdict(micro),
            "ppo_config": {"hidden": hidden},
            "update": 0,
        },
        path,
    )


class EvalScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_script = load_script("eval_rwpp_policy")

    def _args(self, **overrides) -> argparse.Namespace:
        values = {
            "checkpoint": "unused.pt",
            "terrain": "",
            "vision_radius": None,
            "episode_ticks": 300,
            "frame_ms": 1,
            "port_base": 5420,
            "java_home": r"E:\jdk25copy",
            "runtime": ".runtime/rwpp-runtime-1.15",
            "map": "maps/ironzero/IronZero25.tmx",
            "map_name": "IronZero25",
            "map_width": 2700.0,
            "map_height": 2500.0,
            "device": "cpu",
            "win_mode": "annihilation",
            "unlock_ai": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_terrain_override_keeps_checkpoint_obs_dim(self) -> None:
        micro = RwppGameConfig(
            units_per_team=25,
            max_entities=25,
            target_slots=26,
            terrain_path=TERRAIN,
            map_width_px=2700.0,
            map_height_px=2500.0,
        ).micro_config()
        policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=32)
        args = self._args(terrain=str(TERRAIN))
        resolved = self.eval_script._resolve_micro(policy, micro, args)
        self.assertEqual(resolved.flat_obs_dim, micro.flat_obs_dim)
        self.assertEqual(resolved.map_channels, 4)
        self.assertTrue(resolved.terrain_path)

    def test_terrain_override_rejected_when_checkpoint_has_no_channel(self) -> None:
        micro = RwppGameConfig(units_per_team=25, max_entities=25).micro_config()
        policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=32)
        args = self._args(terrain=str(TERRAIN))
        with self.assertRaises(SystemExit):
            self.eval_script._resolve_micro(policy, micro, args)

    def test_no_override_returns_checkpoint_contract_untouched(self) -> None:
        micro = RwppGameConfig(units_per_team=25, max_entities=25).micro_config()
        policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=32)
        self.assertIs(self.eval_script._resolve_micro(policy, micro, self._args()), micro)

    def test_record_replay_forwards_checkpoint_and_bridge_flags(self) -> None:
        captured: list[list[str]] = []

        def fake_main(argv: list[str]) -> int:
            captured.append(list(argv))
            return 0

        original = self.eval_script._load_play_module
        self.eval_script._load_play_module = lambda: SimpleNamespace(main=fake_main)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = self.eval_script._record_replay(self._args(), Path(tmp))
                self.assertTrue(str(result).startswith(str(Path(tmp))))
        finally:
            self.eval_script._load_play_module = original
        self.assertEqual(len(captured), 1)
        argv = captured[0]
        self.assertIn("--checkpoint", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "5421")
        self.assertEqual(argv[argv.index("--ticks") + 1], "300")
        self.assertIn("--win-mode", argv)
        self.assertIn("--unlock-ai", argv)


class PlayScriptTests(unittest.TestCase):
    def test_state_to_frame_uses_episode_start_hp_reference(self) -> None:
        play_script = load_script("play_rwpp_policy")
        state = {
            "own_entities": [
                {"entity_id": 100 + i, "team_id": 0, "hp": 210.0, "max_hp": 210.0, "x": 0.0, "y": 0.0}
                for i in range(25)
            ],
            # 10 of 25 enemies are dead and gone from the bridge stream.
            "visible_enemy_entities": [
                {"entity_id": 200 + i, "team_id": 1, "hp": 210.0, "max_hp": 210.0, "x": 0.0, "y": 0.0}
                for i in range(15)
            ],
        }
        frame = play_script.state_to_frame(
            state, 300, own_max_ref=25 * 210.0, enemy_max_ref=25 * 210.0
        )
        self.assertAlmostEqual(frame["stats"]["own_hp_frac"], 1.0)
        self.assertAlmostEqual(frame["stats"]["enemy_hp_frac"], 15 / 25)
        self.assertEqual(frame["stats"]["enemy_alive"], 15)
        # Without the episode-start reference the alive-unit maximum would hide
        # the losses and report a full-health enemy army.
        legacy = play_script.state_to_frame(state, 300)
        self.assertAlmostEqual(legacy["stats"]["enemy_hp_frac"], 1.0)

    def test_terrain_override_rejected_for_legacy_checkpoint(self) -> None:
        play_script = load_script("play_rwpp_policy")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            save_checkpoint(checkpoint, DEFAULT_CONFIG)
            with self.assertRaises(SystemExit):
                play_script.main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--terrain",
                        str(TERRAIN),
                    ]
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
