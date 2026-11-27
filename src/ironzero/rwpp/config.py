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

from ..micro_config import DEFAULT_CONFIG, WIDE_MAX_ENTITIES, MicroDuelConfig

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
    #
    # Why 0 and not a constant: measured on ``[p2]Lake (2p)``, the *earliest*
    # native ``defeat`` the engine ever produced (``player.wipedOut``) landed at
    # tick 46972; ``recommended_episode_ticks`` gives 20056 for that map. A fixed
    # 8000-tick cap therefore cut every episode off long before the engine could
    # decide anything, which is why 1724/1724 real games reported ``timeout`` and
    # ``terminal`` stayed at exactly 0.000 for 200 PPO updates. Deriving the cap
    # from the map is what makes a verdict *reachable* at all.
    episode_ticks: int = 0
    map_width_px: float = DEFAULT_MAP_PX_W
    map_height_px: float = DEFAULT_MAP_PX_H
    start_timeout: float = 180.0
    obs_timeout: float = 120.0
    # Wall-clock budget per simulated frame. 16 ms = real time (60 fps); 2 ms runs
    # faster on the wall clock. This never changes how many ticks an episode
    # simulates: that is ``episode_ticks`` below (60 ticks = 1 s of game time).
    #
    # Default 0 (unpaced) since ``physics_ms`` exists. With ``physics_ms=16`` the
    # engine's delta is pinned, so ``frame_ms`` is *only* pacing; sleeping 0 ms
    # cannot change the battle. Measured on 6000 ticks (Lake, ``physics_ms=16``):
    # ``frame_ms=1`` -> 9.35 s (10.7x real time), ``frame_ms=0`` -> 3.78 s
    # (26.5x). The 1 ms budget was 68% ``Thread.sleep`` wake-up overhead
    # (``sleep_per_frame(N) ~= 0.93 + 0.99N``), i.e. pure loss.
    #
    # This is safe *because* of ``physics_ms``: if that is explicitly set to 0,
    # ``process.effective_frame_ms`` clamps this back to 1 and warns, preserving
    # the old "frame_ms is also the delta" behaviour.
    frame_ms: int = 0
    vision_radius: float = 0.0  # >0 enables observation-level fog of war (enemy filtering)
    # Deterministic engine seed, forwarded as ``--seed`` when non-zero.
    #
    # ``0`` keeps the engine's own behaviour (a fresh random seed per game). Set it
    # to make "same weights, N episodes" reproducible: the engine re-randomises its
    # seed inside ``loadMap`` for every local game, so without pinning it an
    # evaluation is not repeatable and no win rate can be audited. Measured before
    # this existed: identical 20-game evaluations of one checkpoint scored 0.700
    # and 0.900, and every game log printed a different ``global Seed:``.
    seed: int = 0
    # Army size / observation contract.
    #
    # ``max_entities`` defaults to :data:`WIDE_MAX_ENTITIES` (32) rather than
    # ``DEFAULT_CONFIG.max_entities`` (8) because this class only ever describes a
    # **real game**, where 8 was measured to be too few: a 1200-frame Lake replay
    # dropped 26.1% of enemy unit-frames, kept 61.7% buildings in the slots it did
    # fill, and never showed the enemy's artillery once (0 of 500 unit-frames).
    #
    # The narrow value stays reachable by passing it explicitly, which is what an
    # A/B or a legacy-checkpoint reproduction needs. ``MicroDuelConfig``'s own
    # default is untouched: that one is the library contract the mock engine,
    # fixtures and every shipped checkpoint are expressed in, and changing it would
    # silently reinterpret all of them.
    units_per_team: int = DEFAULT_CONFIG.units_per_team
    max_entities: int = WIDE_MAX_ENTITIES
    target_slots: int = 0  # 0 = max_entities + 1 so every visible enemy is nameable
    #: Independent action slots per decision: ``1`` (historical) or ``2``.
    #:
    #: Kept next to ``units_per_team`` because both define the action *contract*:
    #: this one decides how many orders a single decision can carry. See
    #: ``MicroDuelConfig.action_slots`` for the full argument; the short version is
    #: that one slot cannot express "build AND move", and the measured consequence
    #: was a trained policy (43.75%) losing to a policy that issued nothing (93.75%).
    action_slots: int = 1
    # Terrain passability JSON (TerrainGrid ``map.json`` format). When it resolves
    # to a readable file the observation gains a static terrain channel and
    # ``obs_dim`` grows accordingly; a missing file falls back to the legacy
    # 3-channel observation with a warning.
    terrain_path: Path | None = None
    # Bridge pass-through options. Both are appended to the rwpp-server command
    # line only when set, so legacy invocations stay byte-identical.
    win_mode: str = ""  # e.g. "annihilation" -> --win-mode annihilation
    unlock_ai: bool = False  # -> --unlock-ai
    #: ``--debug-build``: have the bridge print every gate that refuses a build
    #: order (``canQueueAction=false``, ``isValidNewWaypoint=false``, placement
    #: illegal at (x,y), "no builder passed every gate").
    #:
    #: Exposed because the alternative is guessing. The measured symptom this was
    #: added for: on Lake at difficulty 1 the expansion stage dispatches
    #: ``build_extractor`` and the builder's distance to that pool *grows* (894 ->
    #: 1111 px) before the stage gives up and picks a farther pool, so the map keeps
    #: one extractor and income stays at 26 for the whole episode. Every order comes
    #: back ``accepted: true``, so the refusal -- if there is one -- happens inside
    #: the engine, and only the bridge can say which gate it was.
    #:
    #: Off by default: it prints a line per refused order, which is exactly what a
    #: training run does not need.
    debug_build: bool = False  # -> --debug-build
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
    # Engine-native fast forward: ``--fast-forward N`` sets ``engine.gameSpeed = N``
    # once the map is loaded, so every tick advances N times the game time and a
    # match needs 1/N the ticks. ``1`` = off (the flag is then not sent at all).
    #
    # This one *does* change fidelity: it is the engine's own "game speed", the
    # same knob the official UI exposes up to 64. Measured on IronZero25:
    # ``physics_ms=16`` decided a game in 418 frames / 4.06 s, while
    # ``physics_ms=50`` (the 3x that ``updateFrameInternal``'s 3.0 clamp allows)
    # needed 133 frames / 2.71 s. Above that only ``gameSpeed`` helps.
    fast_forward: int = 1
    # Lockstep: the bridge blocks after every ``obs`` until the client answers
    # with ``step``/``shutdown``, for at most ``lockstep_timeout_ms``. Purpose is
    # frame-drop control, not speed -- removing the pacing sleep (``frame_ms=0``)
    # makes the engine outrun the client, and a dropped frame is a decision the
    # policy never made (measured at ``frame_ms=1`` with 8 envs: only 14% of
    # frames consumed, i.e. one decision per ~70 ticks instead of 10).
    #
    # Off by default because it only makes sense when the client answers every
    # observation: ``scripts/record_rwpp_replay.py`` sends orders every 4th obs
    # and would stall. Training/eval turn it on from their CLI.
    lockstep: bool = False
    # Lockstep wait ceiling; on timeout the bridge falls back to free running and
    # counts the event (``lockstepTimeouts=`` in the final log).
    lockstep_timeout_ms: int = 30000
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
    kill_bonus: float = 0.069
    loss_penalty: float = 0.069
    invalid_penalty: float = 0.05
    idle_penalty: float = 0.02
    # One-off reward for an accepted economy build (extractor / factory). Kept
    # well under the +/-1.0 terminal reward: an accepted build used to be worth
    # a hard-coded +5.0, which -- once the bridge actually made building work --
    # would pay more than winning and turn the policy into an extractor farmer.
    build_bonus: float = 0.50
    # One-off reward for a *new combat unit* appearing (queued by ``produce_unit``
    # at one of our factories). ``build_bonus`` paid for the factory and the
    # income term paid for extractors, but nothing paid for actually using the
    # factory -- so "finish the factory, then idle" scored the same as "finish the
    # factory, then field an army", and the cheapest possible build dominated.
    # Kept below ``build_bonus`` * 2 so a single structure still outranks one
    # unit (the structure is the prerequisite), and far below a won game.
    produce_bonus: float = 0.03
    # One-off penalty for losing a completed building. The hp-based ``losses``
    # term cannot price this: a command center has more hp than an army but its
    # real value is the capability it unlocks (production / income). Small enough
    # that retreating to save a factory never outweighs fighting for the win.
    building_loss_penalty: float = 0.30
    # One-off reward for a building that actually gained a tier (T1 -> T2 -> T3).
    #
    # Upgrading is the core of the Rusted Warfare economy -- a T2 extractor earns
    # several times what a T1 does -- and until now *nothing* in the reward pointed
    # at it. ``income`` pays for the resulting rate growth, but only if the agent
    # ever samples ``upgrade_building``, and the measured behaviour was that it
    # never did: the production curriculum eats every credit above
    # ``economy_credit_reserve`` (900), while an upgrade costs 1200, so the bank
    # was structurally never available at the moment the mask opened the kind.
    #
    # Paid on the *tier change* reported by the engine (``tech_level``), not on an
    # accepted receipt, for the same reason ``bonus_build`` counts new buildings:
    # re-issuing an order is accepted every time and would be farmable.
    upgrade_bonus: float = 0.15
    # One-off reward for a *defensive* structure finishing (turret / laser).
    #
    # Separate from ``build_bonus`` because it is a different decision: a factory
    # compounds into an army, a turret only stops the enemy from killing what you
    # built. The agent never built one at all -- the action space could not even
    # name a turret -- so this term is what makes the option visible to PPO once it
    # becomes expressible.
    defence_bonus: float = 0.10
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
    timeout_penalty: float = 2.8
    # Terminal outcome weights, overriding ``MicroDuelConfig``.
    #
    # These are **not** free parameters and they are not the micro-duel's
    # defaults. They are derived from the real-game reward flow, and the
    # derivation is the whole point (see
    # ``audit/奖励最优解_时间上限_敌方AI_定量结论.md``
    # and ``.runtime/optimal_policy_analysis.py``, which recompute every number
    # below from source constants):
    #
    # "Produce a unit" is an unbounded FLOW -- every unit pays ``produce_bonus``,
    # and units keep coming for as long as credits keep arriving. "Win" is a
    # fixed STOCK paid once at the end. Over a long episode the flow always
    # beats the stock unless the stock exceeds the flow's total.
    #
    # Measured on Lake (20056 ticks = 335 s):
    #   income ramp 18 -> 58 cr/s, mean 38  ->  12,712 credits per episode
    #   cheapest combat unit 350 cr          ->  36.3 units affordable
    #   flow cap = produce_bonus * 36.3
    #
    # Under the historical weights (``produce_bonus = 0.10``, ``win = 1.0``) the
    # flow cap was **3.63** against a win of **1.00**: a win was worth 28% of
    # the production it had to compete with, so "produce forever, never attack"
    # was the *argmax of the reward*, not a failure of training. That is exactly
    # the behaviour measured over 231 updates (``bonus_produce`` = +288.70 vs
    # ``terminal`` = +4.51, a 64x ratio), and 96 evaluation episodes at two
    # difficulty levels produced 0 wins and 0 losses with both armies alive.
    #
    # So the win must exceed the flow, and ``produce_bonus`` must shrink so the
    # flow stays small. **The episode multiplier changes this arithmetic**, which
    # is why the two must be set together: doubling the episode doubles the
    # credits (12,712 -> 25,387) and therefore doubles the units the flow can
    # fund (36.3 -> 72.5). With ``episode_tick_multiplier = 2.0`` and
    # ``produce_bonus = 0.03`` the flow cap is ``0.03 * 72.5 = 2.18`` against a
    # win of 5.5 -- a 2.5x margin -- and a pure stall pays
    # ``2.18 - 2.8 = -0.62``, i.e. negative. Raising the horizon *without*
    # cutting this term would have made winning less attractive, not more.
    #
    # Kept on ``RwppGameConfig`` rather than edited into ``MicroDuelConfig``
    # because that one is a pinned library contract: the mock engine, the
    # fixtures and every shipped checkpoint are expressed in its +/-1.0, and
    # changing it would silently reinterpret all of them.
    win_reward: float | None = 5.5
    loss_reward: float | None = -5.5
    #: Multiplier applied to :func:`recommended_episode_ticks`.
    #:
    #: The derived cap is consumed in full (verified: on the lockstep run, 4 of 4
    #: episodes reach ``ticks=20056`` / ``decisions=2006``), so it is a real
    #: constraint rather than a safety fuse that never fires. It is nevertheless
    #: too short for the built-in AI to complete a march: at difficulty 1 it
    #: fields 25.8 units and reaches the cap with both armies alive, and at
    #: difficulty 3 it fields 78.1 and still never closes.
    #:
    #: 2.0 doubles the episode to ~40112 ticks (669 s of game time, 4011
    #: decisions on Lake) and costs roughly 2x wall clock per update, because the
    #: engine simulation -- not the GPU -- is the bottleneck (measured 162 s per
    #: update at 20056 ticks, ~323 s at 40112).
    #:
    #: Deliberately NOT higher by default: the flow cap grows with the episode,
    #: so a longer episode with unchanged weights makes winning *less*
    #: attractive. This field must be raised together with ``win_reward`` and a
    #: reduced ``produce_bonus``, never on its own.
    episode_tick_multiplier: float = 2.0
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
    # Factory curriculum: queue one land factory once the opening extractor is
    # finished.
    #
    # This is the **missing middle link** between ``opening_extractor`` and
    # ``opening_production``. Measured on ``[p2]Lake (2p)`` with a pure ``noop``
    # policy -- so no order of ours can cancel anything -- a whole 20056-tick
    # episode reached:
    #
    #   extractor finished  tick 151
    #   non-CC factory      never (the entity list still showed only
    #                       commandCenter + extractorT1 at tick 20056)
    #   bonus_produce       0.0
    #
    # The extractor curriculum builds the first structure and the production
    # curriculum needs a *finished factory*, but nothing in between ever builds
    # the factory. ``_issue_opening_production`` therefore never fires, which is
    # exactly why two 40-update PPO calibration runs reported
    # ``bonus_produce`` = 0 in **40/40** updates while ``bonus_build`` was
    # nonzero in 17/40: the agent learns to build extractors and stops.
    #
    # The gap is not position legality (a grid scan of all 64 cells of the
    # policy's position head was accepted 64/64) and not interruption (issuing
    # ``build_structure`` once and then only noops finishes the factory at tick
    # 1891; re-issuing it every decision also finishes, at tick 2611). It is
    # simply that no term in the curriculum ever orders it.
    #
    # Like the other two this is a *curriculum*, not a hidden policy: the order
    # goes through the normal bridge path, ``bonus_build``/``bonus_produce`` are
    # still paid by the normal observation-stream rules, and the agent owns every
    # decision afterwards. Set ``--no-opening-factory`` to measure the difference.
    opening_factory: bool = True
    # Production curriculum: keep queueing combat units while a finished factory
    # exists, so the "build a factory, then actually use it" chain is reachable by
    # exploration and stays reachable once learned.
    #
    # Without it ``bonus_produce`` was exactly 0.000 in 199 of 199 measured
    # updates: the agent learned the economy up to "factory standing" and stopped
    # there, because a Lake factory only finishes around tick 491 and a from-scratch
    # policy spends nearly every decision on ``build_extractor``. The gradient that
    # pays for *using* the factory therefore never fired.
    #
    # It used to fire exactly **once** per episode, which was not enough either:
    # the noop baseline ended every episode with ``peak_own = 5`` (one combat unit)
    # while holding **14182 unspent credits** and facing 15 AI units. One unit is
    # not an army, so there was no gradient towards "build many". The stage now
    # sustains ``economy_army_target`` units.
    #
    # Like ``opening_extractor`` this issues an order through the normal bridge
    # path and is paid by the normal rules only (``bonus_produce`` still requires a
    # new combat unit to actually appear), so it removes an exploration barrier
    # without inventing reward. Set ``--no-opening-production`` to measure it.
    opening_production: bool = True
    #: How many combat units the production curriculum sustains.
    #:
    #: This is a *floor*, not a cap: the policy may build as many as it likes on
    #: top, and the stage simply stops adding once this many are alive. Ten is the
    #: size at which a player first has a real army on a stock map (the AI reaches
    #: 13-17), and it is reachable on Lake's economy: 26 credits/s of income plus a
    #: ~4000 starting bank pays for it well inside the 19674-tick budget.
    #: Ten is measured, not guessed. Twelve was tried and was **worse**
    #: (58.3% vs 75.0% over 12 episodes): the extra units tie up credits, so the
    #: expansion stage stops firing and the economy stalls at 2 extractors instead
    #: of reaching more. Ten is the size at which a real army exists while the
    #: economy still compounds.
    economy_army_target: int = 10
    #: Hard ceiling on sustained production, well above ``economy_army_target``.
    #:
    #: ``economy_army_target`` is the point at which the *curriculum* stops adding
    #: units so the economy can keep compounding; this is the point at which the
    #: factory stops accepting new work. The gap matters for the remaining
    #: stalemates: every measured timeout had ``peak_own >= 17`` against an AI with
    #: 13-15 units still alive, i.e. two large armies facing each other with neither
    #: able to finish the other before the tick cap. Left at the army target, a
    #: sustained fight can never out-reinforce the AI; a higher ceiling lets the
    #: factory keep replacing losses during the endgame.
    economy_army_cap: int = 16
    #: How many combat units the attack stage waits for before it marches.
    #:
    #: This is its own knob on purpose. It used to be derived as
    #: ``0.35 * economy_army_cap``, which silently coupled "when do we attack" to
    #: "how many units may we ever own". ``economy_army_cap`` is a *ceiling* used
    #: to let a long game keep replacing losses; deriving the march gate from it
    #: meant raising the ceiling raised the bar the army had to clear before it
    #: was allowed to leave home.
    #:
    #: Measured (scripts/probe_attack_gate32.py, Lake, difficulty 1, 19674 ticks):
    #:
    #:   army_cap=56 -> derived gate 19 units, army peaked at 17
    #:                  1909/1909 decisions gated shut, **0 attack orders**
    #:   army_cap=16 -> derived gate  5 units, army peaked at 10
    #:                  503 attack_move orders issued
    #:
    #: With cap=56 the army never received a single attack order in a whole
    #: episode, which is the real reason rounds 28-31 measured "enemy buildings
    #: lose 0 HP over 19674 ticks". A separate knob makes the two questions
    #: independent, so raising the ceiling can no longer disarm the army.
    #:
    #: The measured trade-off, 4 seeds each on Lake at difficulty 1:
    #:
    #:   march_at=5  (the old cap=16 derivation): army reaches 4-7 units and dies
    #:               in detail -- 0/4 wins, every episode a timeout with a small
    #:               army that had already been traded away
    #:   march_at=12: the army assembles first and the losses are affordable
    #:
    #: Twelve is the smallest number that is a *force* rather than a probe: below
    #: it the army arrives in pieces and the AI's 30-38 units defeat it in detail.
    economy_march_at: int = 12
    #: How many units may sit in a factory's build queue at once.
    #:
    #: The production stage used to go silent whenever the queue was non-empty, on
    #: the theory that re-ordering restarts production the way re-issuing a build
    #: order restarts construction. They are not alike: a Rusted Warfare factory
    #: queue is a *pipeline* that drains as units are delivered, so a non-empty queue
    #: is the normal state of a working factory.
    #:
    #: The cost of the old rule is measured. On Lake at AI difficulty 1, with the
    #: curriculum alone (no policy interference) and both factories standing intact
    #: for the whole episode:
    #:
    #:   curriculum chose ``produce_unit``   9 of 2100 decisions
    #:   queue non-empty after the army peak  540 of 836 decisions
    #:   army                                 11 -> 5
    #:   enemy                                15 -> 31
    #:   unspent credits in the bank          ~600 rising
    #:
    #: (``.runtime/army_decline.json``). The army lost a reinforcement race it never
    #: entered: the credits existed, the factories existed, and the stage declined to
    #: use them because the previous order was still in flight.
    #:
    #: Two is enough to keep the pipeline fed without the "order every decision"
    #: failure that drained the bank to ~496 credits and cancelled an in-progress
    #: extractor (``.runtime/produce_spam.json``). The bound is per-episode *total*
    #: queued units across all factories, so two factories may each hold one.
    economy_production_queue_depth: int = 2
    #: Reward paid for growing the income *rate*, as a fraction of the base rate.
    #:
    #: The historical economy term pays ``extractor_reward * delta / 10``, which is
    #: **0.04** for a whole new extractor -- against a terminal worth +/-1.0, a 25:1
    #: ratio. An agent choosing between a 700-credit extractor and two scouts
    #: therefore sees almost no difference at the moment of the choice, and the
    #: extractor's payoff only arrives hundreds of decisions later through a terminal
    #: it cannot yet attribute. Measured on the two-slot layout at difficulty 1: after
    #: 34 updates the policy had built **zero** extractors and income sat at exactly
    #: the 18.0 base rate while it massed scouts.
    #:
    #: This term pays ``economy_growth_bonus * delta / base_income_reference``, so the
    #: first extractor (18 -> 26) is worth ``0.5 * 0.44 = 0.22`` immediately -- half a
    #: win, for a decision that really is worth about that much. It is a rate *change*,
    #: so it cannot be farmed by sitting still, and it is deliberately kept below 1.0
    #: so it shapes the economy without replacing the objective.
    #:
    #: What it is *not*: a script. It pays for the outcome ("your income rose"), never
    #: for the action, so how many extractors, which ones, whether to upgrade instead,
    #: and when to stop expanding all stay the policy's decision.
    economy_growth_bonus: float = 0.5
    #: Denominator for :attr:`economy_growth_bonus`: the map's base income rate.
    #:
    #: A fixed reference rather than the current rate, because normalising by the
    #: current rate would make the *first* extractor count for more than the sixth,
    #: which is backwards: later income is what pays for the late-game army.
    base_income_reference: float = 18.0
    #: Weight of the potential-based income term (see ``reward_shaping``).
    #:
    #: Unlike :attr:`economy_growth_bonus` -- a one-off payment for the *event* of
    #: raising income -- this pays for the *level*, so it rewards holding a productive
    #: state and charges for losing it. The distinction matters for saving: measured on
    #: the two-slot layout at difficulty 1, the policy's funds sat at 100-650 for a whole
    #: episode while an extractor costs 700, so it never accumulated; a 250-credit tank
    #: paid 0.10 immediately and was always affordable.
    #:
    #: The term is
    #: ``economy_potential_coef * (income - previous_income) / base_income_reference``,
    #: i.e. potential-based shaping, which is proven not to change the optimal policy
    #: (Ng, Harada & Russell 1999). A first extractor (18 -> 26) pays
    #: ``2.0 * 8 / 18 = 0.89``, and a sixth pays the same again -- so expansion stays
    #: attractive instead of being a one-off. It also charges ``-0.89`` if the rate
    #: collapses, which is the correct lesson.
    #:
    #: Sized below a decided game on purpose: the sum over a fully-built economy
    #: (18 -> 58) is ``2.0 * 40 / 18 = 4.4``, paid once across hundreds of decisions,
    #: while a win is 1.0 at the end. It shapes without replacing the objective.
    economy_potential_coef: float = 2.0
    #: Credit floor the production curriculum will not spend below.
    #:
    #: Strictly larger than one cheap combat unit (a ``hoverTank`` costs a few
    #: hundred) so the builder can always finish the next structure. Measured: an
    #: ungated production stage drained the bank to ~496 credits and *cancelled the
    #: in-progress extractor* (``.runtime/produce_spam.json`` -- the episode ended
    #: with ``builder x25``, income down from 26 to 18, no extractor and still zero
    #: combat units).
    economy_credit_reserve: float = 900.0
    #: How many extractors the economy-expansion stage will build.
    #:
    #: Lake has nine resource pools, but capacity is not the limit -- defence is.
    #: Six extractors plus a T2 upgrade is roughly triple the opening income and is
    #: enough to fund a continuous army; every pool beyond that is another static
    #: building the AI can pick off. Zero or negative disables the stage, which is
    #: what ``--no-opening-extractor`` already implies.
    economy_max_extractors: int = 6
    #: How many land factories the build curriculum will raise.
    #:
    #: Two, not one. The measured remaining timeouts (n=24) all reached the enemy
    #: base and left 4 buildings standing, because one factory reinforces too slowly
    #: to finish a base before the tick cap. The cap stops the stage from turning
    #: into an economy sink: a third factory costs credits the army needs.
    economy_max_factories: int = 2
    #: How many builders the production stage will raise.
    #:
    #: **One builder is a serial bottleneck on the whole economy.** Measured on
    #: Lake at difficulty 1 (probe in round 17): the enemy fields 2-3 builders by
    #: tick 9011 while we still have exactly 1, because nothing ever produced a
    #: second one -- ``_UNIT_COMBAT_RANK`` scores ``builder`` as 0, so the
    #: production stage could never pick it. Every construction (extractors,
    #: factories, turrets) therefore queues behind a single unit walking the map.
    #:
    #: A second builder is what lets an extractor expansion and a factory run
    #: concurrently instead of one after the other. Zero disables the stage.
    economy_max_builders: int = 2
    #: Let the curriculum upgrade extractors and factories up the tech ladder.
    #:
    #: The single most valuable investment in a Rusted Warfare economy is a T1
    #: extractor becoming T2, and nothing in the system ever did it. Two independent
    #: reasons, both measured:
    #:
    #: * no stage ever issued ``upgrade_building``. The kind existed, the mask opened
    #:   it whenever the engine reported ``can_upgrade``, and the policy still never
    #:   sampled it usefully -- which is unsurprising, because the *reward* had no
    #:   term for it either (``income`` only pays for the resulting rate growth, and
    #:   only after the fact);
    #: * even if it had sampled it, the bank was never available:
    #:   ``economy_credit_reserve`` (900) is what the production stage refuses to
    #:   spend below, and an upgrade costs 1200, so the production stage drained every
    #:   credit above the reserve down to the reserve and the upgrade was
    #:   unaffordable for the entire episode.
    #:
    #: So this stage runs **before** production in the priority order: it claims the
    #: builder and the credits for a tier-up first, and production only spends what
    #: is left. That is the correct economic ordering -- a permanent rate increase
    #: compounds, a tank does not.
    economy_upgrade: bool = True
    #: How many tier-ups the upgrade stage will attempt per episode.
    #:
    #: Bounded because the top of the ladder is finite and each step costs 1200: an
    #: unbounded stage would spend the whole game's credits climbing tiers on one
    #: extractor while the army never grows.
    #:
    #: **Two, not six**, and the number is measured end to end rather than reasoned
    #: about. With a generous budget the stage spent credits on T2 *and* T3, and the
    #: 64-episode evaluation fell from **96.88% to 73.44%** (47 wins / 0 losses / 17
    #: timeouts, Wilson 95% CI [0.615, 0.827]) -- the confidence interval no longer
    #: cleared 70%. Every timeout was a live army that could not finish the enemy
    #: (measured `own_alive` 8-16 against `enemy_alive` 7-18), i.e. the credits that
    #: should have become units went into tiers instead.
    #:
    #: A T1 -> T2 extractor is worth it: +4 credits/s, which compounds for the whole
    #: episode. T2 -> T3 costs 1200 for +8 credits/s but the upgrade itself takes
    #: 3480 ticks to finish, which on a 20000-tick episode is most of the game paying
    #: nothing. Two upgrades buys the compounding step and leaves the army its funding.
    economy_max_upgrades: int = 2
    #: Build defensive structures (turrets) at the base once the economy stands.
    #:
    #: A turret is the only structure that protects the income that pays for
    #: everything else, and the agent never built one -- because, like the structure
    #: *type*, the decision did not exist in the action space. Now that
    #: ``build_structure`` names a type, the curriculum places the first one at a
    #: spot between our base and the enemy, which is where the AI's pushes arrive.
    #:
    #: Deliberately small: ``economy_max_turrets`` is two. Measured preference order
    #: for credits is upgrade > production > turret -- a turret that never fires is a
    #: pure cost, while income and army compound.
    economy_defence: bool = True
    #: How many defensive structures to place.
    economy_max_turrets: int = 2
    #: Attack curriculum: march the army at the enemy base once it is big enough.    #:
    #: Without it the army is built and then never used. Measured
    #: (``.runtime/engagement_trace.json``): with a complete economy and 10 combat
    #: units, the nearest own unit to the enemy base sat at 1972 px on tick 9001 and
    #: **the same 1972 px on tick 15001** -- nothing moved for 60 s of game time, and
    #: five of six episodes ended in a ``timeout`` with both armies alive. On a stock
    #: skirmish map a battle that is never fought never resolves, so a timeout here
    #: is a lost objective, not a near-miss.
    #:
    #: The stage issues one ``attack_move`` toward the enemy base at the group the
    #: policy may override next decision; it decides no tactics.
    opening_attack: bool = True
    #: How close (world px) the army centroid must be before the attack stage stops
    #: re-issuing. Inside this radius the fight is on and the policy owns the micro
    #: decisions instead of being overridden every decision.
    economy_engage_distance: float = 700.0
    #: Who decides the economy: the scripted curriculum, the policy, or nobody.
    #:
    #: ``"orders"`` -- the historical behaviour. The curriculum stages
    #: (:meth:`RwppDuelEnv._curriculum_actions`) *issue real orders*: they build
    #: the opening extractor, the first factory, the expansion extractors, the
    #: turrets, and they march the army at the enemy base. The policy is left with
    #: the decisions the stages do not make (what to produce, where to move, whom
    #: to attack).
    #:
    #: That arrangement is why the measured win rates said so little about the
    #: policy. When a scripted stage performs a decision, a checkpoint that scores
    #: well may be doing so on the strength of the script; the cleaner statement is
    #: that the *system* wins, not that the agent learned to. It also actively hurt:
    #: the policy's own orders cancel the stage's in-flight construction (every real
    #: order cancels the previous one), and the measured result was 43.75% for the
    #: trained policy against **93.75% for a policy that issued nothing**
    #: (``.runtime/ppo_interference.json``).
    #:
    #: ``"mask"`` -- the curriculum issues **no orders at all**. Its knowledge is
    #: kept, but only as the *legality layer*: the action masks still refuse the
    #: orders the engine would reject, and ``curriculum_owns_build_kinds`` is forced
    #: off so the build kinds are genuinely available. Every economy decision
    #: (whether to build, what, where, when to attack) becomes the policy's, which
    #: is what "the agent learns the economy" has to mean.
    #:
    #: ``"off"`` -- neither orders nor mask gating; the raw engine truth only.
    #:
    #: Default ``"orders"`` so every existing checkpoint, measurement and test keeps
    #: its meaning. Switching to ``"mask"`` changes what the policy is responsible
    #: for, and a checkpoint trained under ``"orders"`` has never made those
    #: decisions, so the two are not interchangeable at fixed weights.
    curriculum_mode: str = "orders"
    log_dir: Path = REPO_ROOT / "runs" / "_rwpp_ppo"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def resolved_episode_ticks(self) -> int:
        """The tick cap actually used: explicit ``episode_ticks`` or a derived one.

        Deriving it from :func:`recommended_episode_ticks` (map diagonal + player
        count) is what makes the fuse meaningful across maps instead of pretending
        one constant fits a 1600 px duel and a 25-player epic alike.

        When derived, it is scaled by :attr:`episode_tick_multiplier`. An
        explicit ``episode_ticks`` is returned verbatim -- an operator who names
        a number gets that number, because silently multiplying an explicit
        value would make ``--episode-ticks 40112`` mean something other than
        what it says.
        """

        explicit = int(self.episode_ticks or 0)
        if explicit > 0:
            return explicit
        base = recommended_episode_ticks(
            self.map_width_px, self.map_height_px, self.num_teams
        )
        multiplier = float(getattr(self, "episode_tick_multiplier", 1.0) or 1.0)
        scaled = int(round(base * max(0.1, multiplier)))
        # Same clamp ``recommended_episode_ticks`` applies, so the multiplier
        # cannot push the cap outside the range the trainer and web UI accept.
        return max(120, min(216000, scaled))

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
        win = float(self.win_reward if self.win_reward is not None else DEFAULT_CONFIG.win_reward)
        loss = float(
            self.loss_reward if self.loss_reward is not None else DEFAULT_CONFIG.loss_reward
        )
        return replace(
            DEFAULT_CONFIG,
            # Terminal weights. ``None`` keeps the micro-duel's +/-1.0 so the mock
            # engine and every pre-existing caller are unaffected; the real-game
            # defaults on this class override them with the derived values (see
            # the field docstring for the flow-vs-stock derivation).
            win_reward=win,
            loss_reward=loss,
            map_width=float(self.map_width_px),
            map_height=float(self.map_height_px),
            episode_ticks=int(self.resolved_episode_ticks()),
            vision_radius=float(self.vision_radius or 0.0),
            units_per_team=int(self.units_per_team),
            max_entities=self.resolved_max_entities(),
            target_slots=self.resolved_target_slots(),
            map_channels=channels,
            terrain_path=str(terrain) if terrain is not None else None,
            # Action layout. ``1`` is the historical single slot; ``2`` gives the
            # economy and the tactical layer their own heads so one decision can
            # say "keep building, and move group 2". Passed through rather than
            # hard-coded because it changes what the action vector *means*: a
            # checkpoint saved under one layout cannot be read under the other.
            action_slots=int(getattr(self, "action_slots", 1) or 1),
            # Hand the two build kinds to the curriculum, but only when the
            # curriculum is actually going to issue those orders.
            #
            # Measured: with the policy free to pick ``build_extractor`` on top of
            # the curriculum's own order, a structure is unfinished on 1949 of 1949
            # decisions and the episode ends at ``occupied = 1`` / ``income = 26``
            # with three own entities -- against ``occupied = 3-4`` / ``income =
            # 42-50`` and a 93% win rate for a policy that issues nothing. Each
            # duplicate build order restarts construction, so the two sources of the
            # same order destroy each other.
            #
            # Two conditions, and the second one is load-bearing for
            # ``curriculum_mode``:
            #
            # * ``opening_extractor`` -- without any curriculum this would remove
            #   the only way to build at all;
            # * the curriculum must actually be *issuing orders*. In
            #   ``curriculum_mode="mask"`` it does not, so closing the kind would
            #   leave nobody able to build the first extractor: measured on the real
            #   Lake opening, ``build_extractor`` is closed in the economy slot and
            #   income stays at the base rate for the whole episode
            #   (``.runtime/check_two_slot_start.py``, P1). A mask layer that
            #   removes the only action capable of starting the economy is not a
            #   safety gate, it is a dead end.
            curriculum_owns_build_kinds=bool(
                self.opening_extractor and self.curriculum_issues_orders
            ),
        )

    @property
    def curriculum_applies_masks(self) -> bool:
        """Whether the curriculum's knowledge may narrow the action masks.

        Both ``"orders"`` and ``"mask"`` do; only ``"off"`` does not. The
        distinction matters for what "the agent learned it" can mean:

        * ``"orders"`` -- the stages act *and* gate. A high win rate may be the
          script's achievement;
        * ``"mask"`` -- the stages only gate. Every order is the policy's, so a
          high win rate is the policy's, and the gates are just "what the engine
          would accept anyway" (derived from reported capabilities, not from
          strategy);
        * ``"off"`` -- neither. The mask still reflects engine legality -- there is
          no way to hand the policy an order the bridge cannot execute -- but the
          curriculum's opinion about *when* a build is timely is gone.
        """

        return self.curriculum_enabled

    @property
    def curriculum_issues_orders(self) -> bool:
        """Whether the scripted stages may emit real orders this episode.

        Only ``curriculum_mode="orders"`` does. ``"mask"`` keeps the stages'
        legality knowledge (through the action masks) but never lets them act,
        which is what makes the economy a *decision* the policy has to learn
        instead of a script it merely refrains from breaking.
        """

        return str(self.curriculum_mode or "orders").strip().lower() == "orders"

    @property
    def curriculum_enabled(self) -> bool:
        """Whether the curriculum participates at all (orders or mask gating)."""

        return str(self.curriculum_mode or "orders").strip().lower() in {"orders", "mask"}


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
