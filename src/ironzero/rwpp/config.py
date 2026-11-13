# SPDX-License-Identifier: Apache-2.0
"""Paths, bridge-contract constants and :class:`RwppGameConfig`.

Everything needed to start one headless real-game instance lives here: the
pinned runtime/java-home defaults, the map identity, the simulated-tick budget
and the reward-shaping coefficients. No I/O and no process handling lives in
this module so the config object stays trivially inspectable.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig

# ``src/ironzero/rwpp/config.py`` -> parents[0]=rwpp, [1]=ironzero, [2]=src,
# [3]=repository root (same value the single-file module computed from
# ``src/ironzero/rwpp_env.py``'s parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[3]
#: RWPP runtime root. ``scripts/make_ironzero_map.py`` resolves the *same* value
#: (``resolve_repo_runtime``) because it installs generated arena maps into this
#: tree while the trainer reads them back; two different answers leave every
#: arena run stuck on "map missing".
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


# Rusted Warfare's fixed loop runs at 60 ticks/s, and a ground unit moves at
# ~66 px/s (``unit_speed`` in ``micro_config``: 1.1 units/tick * 60).
TICKS_PER_SECOND = 60
UNIT_SPEED_PX_PER_SECOND = 66.0

# How many "map traversals" an episode is given.
#
# One traversal is only the approach. A real match needs: build up, approach,
# fight, replace losses, and push the base -- so the budget is several
# traversals, not one. 6 was chosen so that on Lake (3677 px diagonal) the cap
# lands at ~20000 ticks (≈335 s of game time), which is comfortably longer than
# the measured 39000-tick run that never resolved *because the AI never attacked*
# -- the cap must not be the thing that decides a game the engine could decide.
TRAVERSALS_PER_EPISODE = 6.0


def recommended_episode_ticks(
    map_width_px: float,
    map_height_px: float,
    num_teams: int = 2,
) -> int:
    """Tick cap sized from the map and the number of players, not a constant.

    A fixed cap cannot be right for both a 1600 px duel and a 25-player epic:
    too small and a decided game is cut off mid-fight, too large and every
    timeout episode wastes minutes of simulation on a stalemate. The cap only has
    to be an *upper bound that a real game never reaches* -- the engine decides
    the result (see ``reward_shaping``); this is the safety fuse.

    Derivation: crossing the map takes ``diagonal / speed`` seconds; a match needs
    several traversals (approach, fight, rebuild, push); each additional team
    multiplies the number of pushes needed. Rounded to whole seconds, then
    clamped to the range the trainer and the web UI both accept (120..
    216000 ticks, i.e. 2 s .. 1 h of game time).
    """

    diagonal = float(np.hypot(float(map_width_px), float(map_height_px)))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        diagonal = 1600.0
    traversal_seconds = diagonal / UNIT_SPEED_PX_PER_SECOND
    teams = max(1, int(num_teams))
    # sqrt: 4 players do not need 3x the time of 2, because their pushes overlap.
    team_factor = float(np.sqrt(max(1.0, teams - 1)))
    seconds = traversal_seconds * TRAVERSALS_PER_EPISODE * team_factor
    ticks = int(round(seconds * TICKS_PER_SECOND))
    return max(120, min(216000, ticks))


@dataclass
class RwppGameConfig:
    """Everything needed to start one headless real-game instance."""

    runtime: Path = DEFAULT_RUNTIME
    java_home: Path = DEFAULT_JAVA_HOME
    map_relative: str = DEFAULT_MAP
    map_name: str = DEFAULT_MAP_NAME
    # Simulated-tick cap per episode: the **safety fuse**, not the win condition.
    #
    # The engine decides the match (``victoryScreenShown`` / ``defeatScreenShown``
    # / ``wipedOut``); this is only the upper bound past which the episode is cut
    # off and the policy pays ``timeout_penalty`` for failing to finish. It is
    # sized from the map geometry and player count via
    # :func:`recommended_episode_ticks` (see ``resolved_episode_ticks``) instead
    # of being a fixed constant, because no single number is right for both a
    # 1600 px duel and a large team map.
    #
    # ``0`` (the default) means "derive it from the map and player count".
    episode_ticks: int = 0
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
    # Enemy-only difficulty override. Set this when the teacher and the opponent
    # must differ: measured, two AIs at the *same* level mirror each other and never
    # fight, so the teacher only ever demonstrates "build up and wait". A stronger
    # enemy creates the pressure that makes the teacher actually manoeuvre.
    # Requires ``ai_difficulty`` to be set as well (it is the teacher's level).
    enemy_ai_difficulty: int | None = None
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
    #
    # Lowered 2.0 -> 0.3, but be precise about *why*, because the obvious
    # argument is wrong. Measured per-step magnitude on a 3680 px diagonal map:
    # a tank moves ~11 px per decision, so a head-on approach changes the
    # normalised distance by ~0.003 and this term paid ~0.006/step at coef 2.0 --
    # the same order as ``terminal`` (0.027). So the coefficient was *not*
    # numerically dominant step-by-step.
    #
    # The real reason to cut it is what it rewards across an episode: the two
    # bases are ~1800 px apart, so the very first army move towards the enemy
    # pays more than a kill (``kill_bonus = 0.25``) and it keeps paying every
    # decision until contact. Integrated over an episode that is a large,
    # one-directional gradient whose optimum is "walk at the enemy centroid" --
    # in an economy game that means abandoning production. At 0.3 the initial
    # gradient still exists but no longer competes with fighting.
    engagement_coef: float = 0.3
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
    # One-off reward for a *new combat unit* appearing (queued by ``produce_unit``
    # at one of our factories). ``build_bonus`` paid for the factory and the
    # income term paid for extractors, but nothing paid for actually using the
    # factory -- so "finish the factory, then idle" scored the same as "finish the
    # factory, then field an army", and the cheapest possible build dominated.
    # Kept below ``build_bonus`` * 2 so a single structure still outranks one
    # unit (the structure is the prerequisite), and far below a won game.
    produce_bonus: float = 0.10
    # One-off penalty for losing a completed building. The hp-based ``losses``
    # term cannot price this: a command center has more hp than an army but its
    # real value is the capability it unlocks (production / income). Small enough
    # that retreating to save a factory never outweighs fighting for the win.
    building_loss_penalty: float = 0.30
    # Penalty for hitting the tick cap without the engine deciding the match.
    #
    # This replaces the old *heuristic* adjudication (``_decide_by_army``
    # comparing surviving-unit ratios). Two measured facts made the heuristic the
    # wrong tool:
    #
    # 1. The engine already knows who won. It reports ``victoryScreenShown`` /
    #    ``defeatScreenShown`` / ``wipedOut`` and the bridge forwards that as
    #    ``result``; the Python side was second-guessing it with a "who has more
    #    units left" rule, which needed patch after patch (an absolute hp gap, then
    #    a unit-count ratio, then a disconnected carve-out) and still recorded a
    #    2-vs-8 loss as a draw.
    # 2. When the heuristic fired, it was almost always describing a *stalemate*,
    #    not a decided game: a 39000-tick Lake run ended with the enemy at 22
    #    units and us at 1, yet the engine never declared a result because the AI
    #    never actually attacked. Awarding "defeat" there was a guess; the honest
    #    statement is "you did not win in the time allowed".
    #
    # So: the engine decides the result. If the cap is reached first, the episode
    # is a timeout and pays this penalty. Negative so that "never finish" is worse
    # than a clean loss's *shape* on the shaping terms, but small enough that a
    # policy is never encouraged to lose faster than to keep playing.
    timeout_penalty: float = 0.20
    # Team count used by ``recommended_episode_ticks`` when ``episode_ticks`` is
    # left at 0 (derive the cap). Two on every stock skirmish map.
    num_teams: int = 2
    # Opening curriculum: queue an extractor build with the builder at reset.
    #
    # Without it the policy starts every episode facing an empty economy and a
    # ~2000-tick latency before its first combat unit exists (measured on
    # ``[p2]Lake (2p)``: extractor at 141, landFactory at 491, first unit at
    # 2021). A from-scratch policy samples ``build_extractor`` roughly
    # ``1/len(ACTION_KINDS)`` of the time and every other order cancels the
    # in-flight build, so it almost never finishes a structure by exploration --
    # ``income`` stayed at exactly 0.000 for 81 consecutive updates before the
    # mask bug was fixed, and even afterwards the economy only started on ~20% of
    # episodes.
    #
    # This is a *curriculum*, not a hidden policy: the opening order is issued
    # once through the normal bridge path, the reward is paid by the normal rules
    # (``bonus_build`` still requires a finished building), and the agent owns
    # every decision from then on. Set ``--no-opening-extractor`` to disable it and
    # measure the difference.
    opening_extractor: bool = True
    log_dir: Path = REPO_ROOT / "runs" / "_rwpp_ppo"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def resolved_episode_ticks(self) -> int:
        """The tick cap actually used: explicit ``episode_ticks`` or a derived one.

        Deriving it from :func:`recommended_episode_ticks` (map diagonal + player
        count) is what makes the fuse meaningful across maps instead of pretending
        one constant fits a 1600 px duel and a 25-player epic alike.
        """

        explicit = int(self.episode_ticks or 0)
        if explicit > 0:
            return explicit
        return recommended_episode_ticks(
            self.map_width_px, self.map_height_px, self.num_teams
        )

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
            episode_ticks=int(self.resolved_episode_ticks()),
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
