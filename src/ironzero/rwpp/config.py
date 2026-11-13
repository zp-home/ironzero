# SPDX-License-Identifier: Apache-2.0
"""Paths, bridge-contract constants and :class:`RwppGameConfig`.

Everything needed to start one headless real-game instance lives here: the
pinned runtime/java-home defaults, the map identity, the simulated-tick budget
and the reward-shaping coefficients. No I/O and no process handling lives in
this module so the config object stays trivially inspectable.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig

# ``src/ironzero/rwpp/config.py`` -> parents[0]=rwpp, [1]=ironzero, [2]=src,
# [3]=repository root (same value the single-file module computed from
# ``src/ironzero/rwpp_env.py``'s parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME = Path(os.environ.get("IRONZERO_RWPP_RUNTIME", r"E:\IronZeroBuild\rwpp-runtime-1.15"))
DEFAULT_JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))
DEFAULT_MAP = "maps/challenge/Artillery.tmx"
DEFAULT_MAP_NAME = "Artillery"
# The challenge map ships pre-placed armies (22 player units vs 25 AI units), so a
# fresh instance starts as a real micro battle instead of an economy skirmish.
DEFAULT_MAP_PX_W = 2700.0
DEFAULT_MAP_PX_H = 2500.0
# The bridge streams one observation per 10 simulated ticks (DECISION_INTERVAL in
# IronZeroBridge.java); this is part of the bridge contract, not a tunable here.
DECISION_TICKS = 10


@dataclass
class RwppGameConfig:
    """Everything needed to start one headless real-game instance."""

    runtime: Path = DEFAULT_RUNTIME
    java_home: Path = DEFAULT_JAVA_HOME
    map_relative: str = DEFAULT_MAP
    map_name: str = DEFAULT_MAP_NAME
    # Simulated-tick cap per episode. 3600 ticks = 60 s of game time: enough for a
    # 22-vs-25 battle to resolve, while the native game-over signal ends decided
    # battles early (a defeat typically arrives well before the cap).
    episode_ticks: int = 3600
    map_width_px: float = DEFAULT_MAP_PX_W
    map_height_px: float = DEFAULT_MAP_PX_H
    start_timeout: float = 180.0
    obs_timeout: float = 120.0
    # Wall-clock budget per simulated frame. 16 ms = real time (60 fps); 2 ms runs
    # faster on the wall clock. This never changes how many ticks an episode
    # simulates: that is ``episode_ticks`` below (60 ticks = 1 s of game time).
    frame_ms: int = 2
    vision_radius: float = 0.0  # >0 enables observation-level fog of war (enemy filtering)
    # Army size / observation contract. Defaults keep the legacy 8-unit scenario;
    # the fixed 25-vs-25 map sets units_per_team=25, max_entities>=25 and
    # target_slots=26 (or leaves target_slots=0 for the automatic value).
    units_per_team: int = DEFAULT_CONFIG.units_per_team
    max_entities: int = DEFAULT_CONFIG.max_entities
    target_slots: int = 0  # 0 = max_entities + 1 (25 enemies + "no target")
    # Terrain passability JSON (TerrainGrid ``map.json`` format). When it resolves
    # to a readable file the observation gains a static terrain channel and
    # ``obs_dim`` grows accordingly; a missing file falls back to the legacy
    # 3-channel observation with a warning.
    terrain_path: Path | None = None
    # Bridge pass-through options. Both are appended to the rwpp-server command
    # line only when set, so legacy invocations stay byte-identical.
    win_mode: str = ""  # e.g. "annihilation" -> --win-mode annihilation
    unlock_ai: bool = False  # -> --unlock-ai
    # Behaviour-cloning teacher mode. ``ai_pilot`` hands the *controlled* team to
    # the game's built-in AI (see ``IronZeroBridge.installAiPilot``); the AI's own
    # orders can then be read from ``record_ai_orders`` and used as expert labels.
    # Both are opt-in so a normal run's command line is unchanged.
    ai_pilot: bool = False
    record_ai_orders: bool = False
    # Built-in AI strength (-2..3, i.e. 非常简单/简单/正常/困难/非常困难/炼狱).
    #
    # The level is not cosmetic: it feeds ``Team.getAIDifficultyIncomeMultiplier()``,
    # the AI's resource cheat (x0.40 / x0.70 / x1.00 / x1.40 / x1.80 / x3.20), and
    # ``getAIDifficultyArmyValue()``, the army size the AI reaches before attacking.
    # A headless bridge run is neither networked nor a replay and sets no override,
    # so without this field the engine falls back to the global
    # ``SettingsEngine.aiDifficulty``, which defaults to **0 (正常, x1.00)** -- i.e.
    # the teacher plays with no bonus at all. ``None`` keeps that resolution order;
    # an int pins every AI team via ``aiDifficultyLocked``.
    ai_difficulty: int | None = None
    # Fixed engine frame delta, independent of ``frame_ms`` pacing. ``0`` keeps the
    # historical behaviour where ``frame_ms`` is both the delta and the sleep
    # budget (16 ms = real time). Set it (e.g. 16) to keep a real-time simulation
    # delta while ``frame_ms`` paces the wall clock faster.
    #
    # Default 16 (real time) rather than 0. This is a *bug fix*, not a tuning
    # preference: with ``frame_ms = 2`` the engine advanced 2 ms per tick, so the
    # default 3600-tick episode simulated only 3600 x 2 ms = **7.2 s** of game
    # time. A 25-vs-25 battle cannot resolve in 7 s (measured: 24 vs 24 alive at
    # the tick cap), so ``terminal`` was structurally unreachable -- it read
    # exactly 0.000 over 100 PPO updates. Pinning the physics delta to real time
    # makes ``episode_ticks`` mean what it says (3600 ticks = 57.6 s) and lets the
    # game-over signal actually fire. ``--frame-ms`` keeps pacing the wall clock,
    # so this costs no training throughput.
    physics_ms: int = 16
    # Dense shaping: reward closing the distance to the enemy army (the skirmish
    # AI often starts far away, so pure hp-diff reward has no gradient early on).
    engagement_coef: float = 2.0
    # Reward terms (docs 7.2/7.4): kills/own losses plus small penalties for
    # rejected orders and for idle decisions while enemies are in sight.
    # LLM/rule strategic layer (optional): the directive nudges reward shaping only,
    # it never emits low-level actions (doc 12.4 / stage 5).
    strategy_coef: float = 0.5
    kill_bonus: float = 0.25
    loss_penalty: float = 0.25
    invalid_penalty: float = 0.05
    idle_penalty: float = 0.02
    # One-off reward for an accepted economy build (extractor / factory). Kept
    # well under the +/-1.0 terminal reward: an accepted build used to be worth
    # a hard-coded +5.0, which -- once the bridge actually made building work --
    # would pay more than winning and turn the policy into an extractor farmer.
    build_bonus: float = 0.20
    log_dir: Path = REPO_ROOT / "runs" / "_rwpp_ppo"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def resolved_max_entities(self) -> int:
        """Entity slots per block; never fewer than one full army."""

        return max(1, int(self.max_entities), int(self.units_per_team))

    def resolved_target_slots(self) -> int:
        """Target head size: explicit value, else ``max_entities + 1``."""

        explicit = int(self.target_slots or 0)
        if explicit > 0:
            return explicit
        return self.resolved_max_entities() + 1

    def resolved_terrain(self) -> Path | None:
        """Resolve ``terrain_path`` relative to cwd (then repo root)."""

        if not self.terrain_path:
            return None
        candidate = Path(self.terrain_path)
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.is_absolute():
            fallback = (REPO_ROOT / candidate).resolve()
            if fallback.is_file():
                return fallback
        warnings.warn(
            f"terrain file not found: {self.terrain_path}; keeping the legacy "
            f"{DEFAULT_CONFIG.map_channels}-channel observation (no terrain channel)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def micro_config(self) -> MicroDuelConfig:
        """Observation/action contract aligned with the real map geometry."""

        terrain = self.resolved_terrain()
        channels = DEFAULT_CONFIG.map_channels + (1 if terrain is not None else 0)
        return replace(
            DEFAULT_CONFIG,
            map_width=float(self.map_width_px),
            map_height=float(self.map_height_px),
            episode_ticks=int(self.episode_ticks),
            vision_radius=float(self.vision_radius or 0.0),
            units_per_team=int(self.units_per_team),
            max_entities=self.resolved_max_entities(),
            target_slots=self.resolved_target_slots(),
            map_channels=channels,
            terrain_path=str(terrain) if terrain is not None else None,
        )


__all__ = [
    "DECISION_TICKS",
    "DEFAULT_JAVA_HOME",
    "DEFAULT_MAP",
    "DEFAULT_MAP_NAME",
    "DEFAULT_MAP_PX_H",
    "DEFAULT_MAP_PX_W",
    "DEFAULT_RUNTIME",
    "REPO_ROOT",
    "RwppGameConfig",
]
