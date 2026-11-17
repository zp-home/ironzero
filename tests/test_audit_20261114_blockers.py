# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the four structural blockers found in the 2026-11-14 audit.

Each test pins a defect that made the training loop unable to observe a win:

1. ``episode_ticks`` too short for a verdict (1724/1724 games timed out).
2. Observation frames silently dropped under the real training configuration
   (training ran at 70 ticks/decision while evaluation ran at 19).
3. ``--physics-ms`` never reaching the engine's physics argument, so the
   documented "16 ms = real time" claim was false.
4. HP fractions clamped to 1.0, making the dashboard KPI constant in 89/89
   original-map episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ironzero.rwpp.config import (
    RwppGameConfig,
    recommended_episode_ticks,
)
from ironzero.rwpp.reward import hp_fraction_ratios, hp_fractions
from ironzero.training.ppo import _config_from_saved
from ironzero.training.evaluate import (
    minimum_episodes_for_target,
    wilson_interval,
    wilson_lower_for_target,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Episode length must be able to hold a verdict
# ---------------------------------------------------------------------------
class TestEpisodeLengthIsReachableTests:
    """The tick cap has to be long enough for the engine to decide the match."""

    def test_default_config_derives_the_cap_from_the_map(self) -> None:
        """``episode_ticks=0`` means "derive", and that is the shipped default.

        A fixed 4800/3600 cut every original-map episode off long before a
        verdict: measured on ``[p2]Lake (2p)`` the earliest native result ever
        observed was tick 46972 and 1724 of 1724 real games reported ``timeout``.

        The derived cap is additionally scaled by
        ``RwppGameConfig.episode_tick_multiplier``. That is not a second
        derivation: the derived value is *consumed in full* (verified on the
        lockstep run -- 4 of 4 episodes reach ``ticks=20056`` /
        ``decisions=2006``), so it is a real constraint, and it is still too
        short for the built-in AI to complete a march. See the field docstring.
        """

        game = RwppGameConfig()
        assert game.episode_ticks == 0
        base = recommended_episode_ticks(
            game.map_width_px, game.map_height_px, game.num_teams
        )
        assert game.resolved_episode_ticks() == int(
            round(base * game.episode_tick_multiplier)
        )
        # With the multiplier opted out the historical number comes back exactly.
        historical = RwppGameConfig(episode_tick_multiplier=1.0)
        assert historical.resolved_episode_ticks() == base

    def test_lake_derived_cap_exceeds_the_measured_decisive_tick(self) -> None:
        """Lake must get a cap in the right order of magnitude for a real match."""

        derived = recommended_episode_ticks(2600.0, 2600.0, 2)
        # 8000 was the value that produced 1724/1724 timeouts.
        assert derived > 8000
        # 46972 ticks is the earliest native ``defeat`` ever logged; the derived
        # cap does not have to reach it (the objective/dominance modes decide much
        # earlier) but it must not be an order of magnitude short either.
        assert derived >= 20000

    def test_explicit_episode_ticks_still_wins(self) -> None:
        game = RwppGameConfig(episode_ticks=12345)
        assert game.resolved_episode_ticks() == 12345


# ---------------------------------------------------------------------------
# 2. Dropped observation frames must be visible, not silent
# ---------------------------------------------------------------------------
class TestSkippedFrameAccountingTests:
    """A drained-and-dropped frame is a decision the policy never made."""

    def test_env_exposes_a_skipped_frame_counter(self) -> None:
        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5999)
        assert env.skipped_frames == 0

    def test_counter_helper_tolerates_objects_without_the_attribute(self) -> None:
        """Test doubles and external wrappers predate the counter."""

        from ironzero.rwpp.session import _count_skipped

        class Bare:
            pass

        env = Bare()
        _count_skipped(env, 3)
        assert env.skipped_frames == 3
        _count_skipped(env, 2)
        assert env.skipped_frames == 5
        # A zero skip writes nothing, so it can never resurrect a deleted attr.
        _count_skipped(Bare(), 0)


# ---------------------------------------------------------------------------
# 3. HP fractions must not be a constant on original maps
# ---------------------------------------------------------------------------
class TestHpFractionKeepsInformationTests:
    """``enemy_hp_frac`` was exactly 1.0 in 89 of 89 original-map episodes."""

    @staticmethod
    def _state(own_hp: float, enemy_hp: float) -> dict:
        return {
            "own_entities": [{"hp": 4000.0, "max_hp": 4000.0}],
            "visible_enemy_entities": [{"hp": enemy_hp, "max_hp": 4000.0}],
        }

    def test_ratio_recovers_information_the_clamped_fraction_drops(self) -> None:
        state = self._state(4000.0, 8000.0)
        _, clamped = hp_fractions(state, 4170.0, 4170.0)
        _, honest = hp_fraction_ratios(state, 4170.0, 4170.0)
        assert clamped == 1.0, "the clamping behaviour itself is unchanged"
        assert honest > 1.5, "the unclamped ratio reports the real over-growth"

    def test_ratio_matches_the_clamped_value_below_one(self) -> None:
        state = self._state(210.0, 1000.0)
        clamped = hp_fractions(state, 4170.0, 4170.0)
        honest = hp_fraction_ratios(state, 4170.0, 4170.0)
        assert clamped == pytest.approx(honest)

    def test_ratio_is_exported(self) -> None:
        from ironzero.rwpp import reward

        assert "hp_fraction_ratios" in reward.__all__


# ---------------------------------------------------------------------------
# 4. Evaluation must be able to demonstrate the target
# ---------------------------------------------------------------------------
class TestEvaluationSizingTests:
    """"Won 70%" cannot be claimed from a sample that cannot express it."""

    def test_four_clean_wins_cannot_prove_seventy_percent(self) -> None:
        low, _ = wilson_interval(4, 4)
        assert low < 0.70, "the old eval_episodes=4 could never clear 70%"
        assert low == pytest.approx(0.510, abs=0.002)

    def test_sizing_helpers_agree_with_wilson(self) -> None:
        # 15/16 clears 0.70, 14/16 does not.
        assert wilson_interval(15, 16)[0] >= 0.70
        assert wilson_interval(14, 16)[0] < 0.70
        assert wilson_lower_for_target(0.7, 16) == 15.0

    def test_minimum_episode_count_for_target_is_reported(self) -> None:
        """The helper answers "how many games before a clean sweep proves 70%?".

        9 is the real answer: ``wilson_interval(9, 9)`` lower bound is 0.7008,
        while 8/8 only reaches 0.6756. Pinned so the number cannot drift silently
        (and so nobody re-introduces a 4-episode "proof", whose bound is 0.510).
        """

        assert minimum_episodes_for_target(0.7) == 9
        assert wilson_interval(9, 9)[0] >= 0.70
        assert wilson_interval(8, 8)[0] < 0.70


# ---------------------------------------------------------------------------
# 5. Verdict modes: the new objective mode must be wired end to end
# ---------------------------------------------------------------------------
class TestObjectiveWinModeIsWiredTests:
    """``native`` produced 0 victories over 7129 real games on stock maps."""

    def test_bridge_accepts_the_objective_mode(self) -> None:
        source = (
            REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
        ).read_text(encoding="utf-8")
        assert '"objective".equals(requested)' in source, (
            "the bridge must parse --win-mode objective"
        )
        assert "'objective'.equals" not in source
        assert 'if ("objective".equals(winMode))' in source, (
            "detectGameResult must branch on objective mode"
        )

    def test_bridge_uses_the_engine_objective_semantics(self) -> None:
        """The objective verdict must follow the engine's own definition."""

        source = (
            REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
        ).read_text(encoding="utf-8")
        # mainBuildings = factory building that is not a turret and not an extractor
        assert "mainBuildings++" in source
        assert "instanceof Turret" in source
        assert "instanceof Extractor" in source

    def test_host_service_defaults_objective_for_non_arena_maps(self) -> None:
        text = (REPO / "scripts" / "rwpp_service" / "ppo.py").read_text(encoding="utf-8")
        assert 'win_mode = "objective"' in text, (
            "non-arena runs must not silently fall back to native"
        )

    def test_web_form_exposes_win_mode_and_opponent_difficulty(self) -> None:
        html = (REPO / "web" / "frontend" / "index.html").read_text(encoding="utf-8")
        for element in (
            "fe-ppo-win-mode",
            "fe-ppo-enemy-ai-difficulty",
            "fe-ppo-unlock-ai",
        ):
            assert f'id="{element}"' in html, f"{element} must be a real control"

        state = (REPO / "web" / "frontend" / "js" / "state.js").read_text(encoding="utf-8")
        for element in ("fe-ppo-win-mode", "fe-ppo-enemy-ai-difficulty", "fe-ppo-unlock-ai"):
            assert f"'{element}'" in state, f"{element} must be cached in els"

        run = (REPO / "web" / "frontend" / "js" / "ppoRun.js").read_text(encoding="utf-8")
        for key in ("win_mode:", "enemy_ai_difficulty:", "unlock_ai:"):
            assert key in run, f"the PPO request must send {key}"


# ---------------------------------------------------------------------------
# 7. --physics-ms must reach the engine's physics argument
# ---------------------------------------------------------------------------
class TestPhysicsMsReachesTheEngine:
    """``updateFrameInternal(1.0f, physicsMs)`` put the flag in the FPS slot.

    The engine only advances on its first (float) argument -- see
    ``RWGameEngine.java:1217`` (fps accumulator uses ``var2``) vs ``:1521``
    (``gameTimeMillis += var1 * 16.667``) and ``:1524`` (``simulationTick++``).
    """

    @staticmethod
    def _bridge_source() -> str:
        return (
            REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
        ).read_text(encoding="utf-8")

    def test_delta_is_passed_in_the_float_argument(self) -> None:
        source = self._bridge_source()
        assert "updateFrameInternal(physicsDelta, pacingElapsedMs)" in source
        # The old, broken call must be gone from *executable* code. The doc comment
        # above the call quotes the old form on purpose (it explains the bug), so
        # strip comment lines before asserting.
        code_lines = [
            line for line in source.splitlines() if not line.lstrip().startswith("//")
        ]
        code = "\n".join(code_lines)
        assert "updateFrameInternal(1.0f, physicsElapsedMs)" not in code

    def test_zero_keeps_the_historical_delta(self) -> None:
        """``physics_ms=0`` must stay bit-identical to the pre-fix behaviour."""

        source = self._bridge_source()
        assert "physicsMs > 0 ? (physicsMs / 16.666666f) : 1.0f" in source

    def test_measuring_instrument_is_documented(self) -> None:
        """tick counts and ``game_time`` cannot detect the flag -- say so in code.

        Both are linear in ``simulationTick``, so a future "verification" that
        compares them would wrongly conclude the flag is inert (which is exactly
        what happened once during this audit).
        """

        source = self._bridge_source()
        assert "fpsTimeAccumulatorMs" in source
        assert "simulationTick++" in source or "simulationTick" in source


# ---------------------------------------------------------------------------
# 9. The trainer module must stay runnable as ``-m ironzero.training.train_rwpp``
# ---------------------------------------------------------------------------
class TestTrainerModuleEntryPoint:
    """A silently missing ``__main__`` guard makes every web run do nothing.

    The host service launches training as
    ``python -m ironzero.training.train_rwpp ...``. If the module-level
    ``if __name__ == "__main__": raise SystemExit(main())`` block is lost, that
    command imports the module, runs nothing, and exits 0 -- so the job is
    recorded as ``done`` with ``updates_done=0`` and a 0-byte ``train.log``,
    which looks exactly like "the run finished immediately for no reason".

    This happened for real while removing a helper during the 2026-11-14 fixes.
    """

    def test_module_has_a_main_guard(self) -> None:
        source = (
            REPO / "src" / "ironzero" / "training" / "train_rwpp.py"
        ).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source, (
            "python -m ironzero.training.train_rwpp must actually start training"
        )
        assert "raise SystemExit(main())" in source

    def test_parser_and_game_are_importable(self) -> None:
        from ironzero.training.train_rwpp import build_game, build_parser

        args = build_parser().parse_args([])
        game = build_game(args)
        assert game.resolved_episode_ticks() > 0


# ---------------------------------------------------------------------------
# 10. Engine seed: what can and cannot be pinned
# ---------------------------------------------------------------------------
class TestEngineSeedPlumbing:
    """``reset(seed)`` used to reach nothing but the episode id.

    The bridge now forwards ``--seed`` and re-seeds the engine after the map
    loads, which makes the *engine-side* RNG reproducible. It cannot make a whole
    episode bit-reproducible: the game calls ``Math.random()`` in 12 places that
    never see ``GameEngine.randomSeed`` (AI target choice, unit-group scatter,
    zone selection, waypoint jitter). Those are pinned nowhere and cannot be
    patched from outside the game.
    """

    def test_config_exposes_a_seed_field(self) -> None:
        assert RwppGameConfig().seed == 0

    def test_seed_is_forwarded_only_when_set(self) -> None:
        from ironzero.rwpp.process import _bridge_command

        plain = _bridge_command(Path("rwpp-server.bat"), 5130, RwppGameConfig())
        assert "--seed" not in plain, "an unseeded run keeps the engine's own seed"

        seeded = _bridge_command(
            Path("rwpp-server.bat"), 5130, RwppGameConfig(seed=4242)
        )
        assert "--seed" in seeded
        assert seeded[seeded.index("--seed") + 1] == "4242"

    def test_bridge_parses_and_applies_the_seed(self) -> None:
        source = (
            REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
        ).read_text(encoding="utf-8")
        assert '"--seed".equals(argument)' in source
        # The engine re-randomises the seed inside loadMap, so a pre-load write is
        # not enough; the run loop must re-apply it afterwards.
        assert "seed applied after load" in source
        assert "global Random re-seeded" in source

    def test_the_known_non_seeded_randoms_are_documented(self) -> None:
        """A future reader must not assume "seeded" means "reproducible"."""

        source = (
            REPO / "src" / "ironzero_bridge" / "java" / "ironzero" / "IronZeroBridge.java"
        ).read_text(encoding="utf-8")
        assert "Math.random" in source, (
            "the javadoc must record which randomness is out of reach"
        )


class TestWarmStartReadsTheCheckpointsOwnLayout:
    """``--init-checkpoint`` used to migrate against the wrong source layout.

    ``train_rwpp.main`` called ``_config_from_saved(payload)`` with the *whole*
    checkpoint payload, but that helper rebuilds a ``MicroDuelConfig`` from a dict
    of config fields -- and every field it looks for lives under
    ``payload["micro_config"]``. The filtered dict therefore came back empty and
    the function returned ``DEFAULT_CONFIG``, so every warm start was told the
    source layout was the 8-entity / 3-channel default regardless of what the
    checkpoint actually held.

    The consequence is the quiet one: ``_column_source`` computed a remap for a
    layout that never existed, so the transferred trunk columns were bound to
    features they were never trained on -- "loaded 99% of parameters, behaves like
    noise". No warning is emitted because every shape still fits.
    """

    def test_a_whole_payload_yields_the_default_config(self) -> None:
        """Pin the failure mode itself, so the helper's contract stays explicit."""

        from ironzero.micro_config import DEFAULT_CONFIG

        payload = {
            "model": {},
            "micro_config": {
                "max_entities": 25,
                "target_slots": 26,
                "map_channels": 4,
                "type_features": 9,
            },
            "update": 7,
        }
        assert _config_from_saved(payload) == DEFAULT_CONFIG, (
            "passing the whole payload must be understood as a bug, not a feature"
        )

    def test_passing_the_micro_config_dict_recovers_the_real_layout(self) -> None:
        payload = {
            "model": {},
            "micro_config": {
                "max_entities": 25,
                "target_slots": 26,
                "map_channels": 4,
                "type_features": 9,
            },
        }
        recovered = _config_from_saved(dict(payload.get("micro_config") or {}))
        assert recovered.max_entities == 25
        assert recovered.target_slots == 26
        assert recovered.map_channels == 4

    def test_the_trainer_hands_over_the_micro_config_sub_dict(self) -> None:
        """The call site, not just the helper, must be correct."""

        source = (REPO / "src" / "ironzero" / "training" / "train_rwpp.py").read_text(
            encoding="utf-8"
        )
        assert "source_micro=_config_from_saved(payload)" not in source, (
            "the whole-payload call is the bug; it must not come back"
        )
        assert '_config_from_saved(dict(payload.get("micro_config") or {}))' in source


# ---------------------------------------------------------------------------
# 9. The production curriculum exists (bonus_produce was 0 in 199/199 updates)
# ---------------------------------------------------------------------------
class TestProductionCurriculumTests:
    def test_config_exposes_the_switch_and_cli_can_disable_it(self) -> None:
        game = RwppGameConfig()
        assert game.opening_production is True

        from ironzero.training.train_rwpp import build_parser

        args = build_parser().parse_args(["--no-opening-production"])
        assert args.opening_production is False

    def test_cli_exposes_the_extractor_switch_promised_in_the_docs(self) -> None:
        """``config.py`` promised ``--no-opening-extractor`` but never shipped it."""

        from ironzero.training.train_rwpp import build_parser

        assert build_parser().parse_args([]).opening_extractor is True
        assert build_parser().parse_args(["--no-opening-extractor"]).opening_extractor is False

    def test_env_has_the_curriculum_entry_point(self) -> None:
        from ironzero.rwpp.env import RwppDuelEnv

        assert hasattr(RwppDuelEnv, "_issue_opening_production")

    def test_curriculum_only_fires_for_a_finished_factory(self) -> None:
        """An under-construction factory advertises an empty menu and rejects orders."""

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5998)
        env.production_curriculum_done = False
        env.masks = {}
        env.obs_state = {"own_entities": []}
        env.state = {
            "own_entities": [
                {"is_factory": True, "build_progress": 0.4, "is_builder": False}
            ],
            "factories": [
                {"entity_id": 30, "unit_type": "landFactory", "can_produce": []}
            ],
        }
        # Must not issue anything: the factory's menu is still empty.
        assert env._issue_opening_production() == []
        assert env.production_curriculum_done is False

    def test_curriculum_fires_once_the_factory_advertises_a_combat_unit(self) -> None:
        """The production stage is gated on ``can_produce``, not on entity existence.

        ``decode_action`` turns ``produce_unit`` into a real order only when the
        factory is named by ``entity_id`` (otherwise it degrades to ``hold``, which
        is the correct behaviour for a malformed order). Both halves are asserted
        here so the test pins the contract rather than the incidental payload.
        """

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5999)
        env.production_curriculum_done = False
        env.masks = {}
        state = {
            "own_entities": [
                {"entity_id": 30, "unit_type": "landFactory", "is_factory": True}
            ],
            "factories": [
                {
                    "entity_id": 30,
                    "unit_type": "landFactory",
                    "can_produce": ["builder", "hoverTank"],
                }
            ],
        }
        env.state = state
        # The order is decoded against what the policy saw, so ``obs_state`` must
        # describe the same world (that is the production contract, not a test
        # convenience: with fog enabled the two legitimately differ).
        env.obs_state = state
        # The new stage sustains an army, so it also needs affordable credits. The
        # reserve gate exists because an ungated stage drained the bank to ~496 and
        # cancelled the in-progress extractor (``.runtime/produce_spam.json``).
        state["global_state"] = {"funds": 5000.0, "income": 40.0}
        orders = env._issue_opening_production()
        assert [order["kind"] for order in orders] == ["produce_unit"]
        # A combat unit, not a builder: filling the menu with builders would pay
        # nothing and teach nothing.
        assert orders[0]["unit_type"] == "hoverTank"
        assert orders[0]["factory_entity_id"] == 30

    def test_production_stops_at_the_army_target(self) -> None:
        """Sustained production must stop, or it drains the bank forever.

        The stage used to fire exactly once per episode, which capped the army at a
        single unit while 14182 credits went unspent. Sustaining an army fixes that
        but needs an upper bound: a stage that never stops spending competes with
        the extractor/factory chain for credits and stalls the economy.
        """

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5998)
        cap = env.game.economy_army_cap
        env.masks = {}
        state = {
            "own_entities": [
                {"entity_id": 30, "unit_type": "landFactory", "is_factory": True}
            ]
            + [
                {
                    "entity_id": 100 + index,
                    "unit_type": "hoverTank",
                    "can_attack": True,
                    "is_builder": False,
                    "is_building": False,
                    "hp": 200.0,
                }
                for index in range(cap)
            ],
            "factories": [
                {
                    "entity_id": 30,
                    "unit_type": "landFactory",
                    "can_produce": ["builder", "hoverTank"],
                    "queue_length": 0,
                }
            ],
            "global_state": {"funds": 5000.0, "income": 40.0},
        }
        env.state = state
        env.obs_state = state
        assert env._issue_opening_production() == []

    def test_production_reinforces_between_target_and_cap(self) -> None:
        """Between the army target and the cap the factory keeps replacing losses.

        Measured: a stage that stopped at the target left the endgame to a 1:1
        reinforcement race the AI wins, because every remaining stalemate ended with
        our army at ``peak_own >= 17`` against 13-15 surviving AI units.
        """

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5995)
        target = env.game.economy_army_target
        cap = env.game.economy_army_cap
        assert target < cap, "the gap between target and cap is what this test covers"
        env.masks = {}
        state = {
            "own_entities": [
                {"entity_id": 30, "unit_type": "landFactory", "is_factory": True}
            ]
            + [
                {
                    "entity_id": 200 + index,
                    "unit_type": "hoverTank",
                    "can_attack": True,
                    "is_builder": False,
                    "is_building": False,
                    "hp": 200.0,
                }
                for index in range(target)
            ],
            "factories": [
                {
                    "entity_id": 30,
                    "unit_type": "landFactory",
                    "can_produce": ["builder", "hoverTank"],
                    "queue_length": 0,
                }
            ],
            "global_state": {"funds": 5000.0, "income": 40.0},
        }
        env.state = state
        env.obs_state = state
        orders = env._issue_opening_production()
        assert [order["kind"] for order in orders] == ["produce_unit"]

    def test_production_will_not_spend_below_the_reserve(self) -> None:
        """The reserve keeps the builder funded.

        Measured (``.runtime/produce_spam.json``): ordering a unit every decision
        drained the bank to ~496 credits, cancelled the in-progress extractor, and
        produced 25 builders and **zero** combat units.
        """

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5997)
        env.masks = {}
        state = {
            "own_entities": [
                {"entity_id": 30, "unit_type": "landFactory", "is_factory": True}
            ],
            "factories": [
                {
                    "entity_id": 30,
                    "unit_type": "landFactory",
                    "can_produce": ["builder", "hoverTank"],
                    "queue_length": 0,
                }
            ],
            "global_state": {"funds": 100.0, "income": 40.0},
        }
        env.state = state
        env.obs_state = state
        assert env._issue_opening_production() == []

    def test_production_waits_while_the_factory_queue_is_busy(self) -> None:
        """A busy factory needs no further order; repeating one costs credits."""

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=5996)
        env.masks = {}
        state = {
            "own_entities": [
                {"entity_id": 30, "unit_type": "landFactory", "is_factory": True}
            ],
            "factories": [
                {
                    "entity_id": 30,
                    "unit_type": "landFactory",
                    "can_produce": ["builder", "hoverTank"],
                    "queue_length": 2,
                }
            ],
            "global_state": {"funds": 5000.0, "income": 40.0},
        }
        env.state = state
        env.obs_state = state
        assert env._issue_opening_production() == []

    def test_factory_stage_exists_and_defaults_on(self) -> None:
        """The factory stage is the link between the extractor and production stages.

        Without it ``bonus_produce`` cannot fire: the extractor curriculum builds
        the first structure and the production curriculum needs a *finished*
        factory, but nothing ordered the factory. Measured with a pure ``noop``
        policy on Lake over a full 20056-tick episode, the only structures were
        ``commandCenter`` + ``extractorT1``, and ``bonus_produce`` was 0.0.
        """

        from ironzero.training.train_rwpp import build_parser
        from ironzero.rwpp.env import RwppDuelEnv

        assert RwppGameConfig().opening_factory is True
        assert build_parser().parse_args([]).opening_factory is True
        assert build_parser().parse_args(["--no-opening-factory"]).opening_factory is False
        assert hasattr(RwppDuelEnv, "_issue_opening_factory")

    def test_factory_stage_waits_for_an_operational_extractor(self) -> None:
        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=6001)
        env.masks = {}
        env.obs_state = {"own_entities": []}
        env.factory_curriculum_done = False
        env.factory_site = None
        env.state = {
            "own_entities": [
                {
                    "entity_id": 25,
                    "unit_type": "extractorT1",
                    "is_building": True,
                    "build_progress": 0.3,
                    "is_operational": False,
                    "x": 1170.0,
                    "y": 2050.0,
                },
                {
                    "entity_id": 24,
                    "unit_type": "builder",
                    "is_builder": True,
                    "hp": 120.0,
                    "x": 1100.0,
                    "y": 2100.0,
                },
            ],
            "factories": [],
        }
        assert env._issue_opening_factory() == []

    def test_factory_stage_orders_a_land_factory_and_latches_its_site(self) -> None:
        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=6002)
        env.masks = {}
        env.obs_state = {"own_entities": []}
        env.factory_curriculum_done = False
        env.factory_site = None
        env.state = {
            "own_entities": [
                {
                    "entity_id": 25,
                    "unit_type": "extractorT1",
                    "is_building": True,
                    "build_progress": 1.0,
                    "is_operational": True,
                    "x": 1170.0,
                    "y": 2050.0,
                },
                {
                    "entity_id": 24,
                    "unit_type": "builder",
                    "is_builder": True,
                    "hp": 120.0,
                    "x": 1100.0,
                    "y": 2100.0,
                },
            ],
            "factories": [],
        }
        orders = env._issue_opening_factory()
        assert [order["kind"] for order in orders] == ["build_structure"]
        assert orders[0]["structure_type"] == "land_factory"
        # The site is latched: re-issuing must target the same cell, otherwise the
        # factory restarts wherever the builder has wandered to and never finishes.
        site = env.factory_site
        assert site is not None
        env._issue_opening_factory()
        assert env.factory_site == site

    def test_curriculum_advances_past_a_satisfied_stage(self) -> None:
        """A satisfied stage returns ``[]`` and the *next* stage must still run.

        This is the defect that hid every other one. ``_curriculum_actions`` walked
        the stages and short-circuited on ``order is not None``, but each stage
        returns a *list* -- so the empty list a satisfied stage returns is not
        ``None``, the loop stopped there, and the factory/production stages were
        never called. Measured on a pure ``noop`` Lake episode with the extractor
        finished at tick 2001: the extractor stage ran 119/119 times and returned
        ``[]`` every time, the factory stage ran zero times, ``bonus_produce``
        stayed 0.0, and the symptom looked exactly like "the curriculum does not
        exist".

        The observable contract is that the extractor stage is skipped once the
        extractor is operational and the factory stage is reached in the same call.
        """

        from ironzero.rwpp.env import RwppDuelEnv

        env = RwppDuelEnv(RwppGameConfig(), port=6003)
        env.masks = {}
        state = {
            "own_entities": [
                {
                    "entity_id": 25,
                    "unit_type": "extractorT1",
                    "is_building": True,
                    "build_progress": 1.0,
                    "is_operational": True,
                    "x": 1170.0,
                    "y": 2050.0,
                },
                {
                    "entity_id": 24,
                    "unit_type": "builder",
                    "is_builder": True,
                    "hp": 120.0,
                    "x": 1100.0,
                    "y": 2100.0,
                },
            ],
            "factories": [],
            "resource_slots": [],
        }
        env.state = state
        env.obs_state = state

        # The completed extractor means stage 1 has nothing to do.
        assert env._issue_opening_extractor() == []
        # ... and the composite entry point must fall through to stage 2.
        orders = env._curriculum_actions()
        assert [order["kind"] for order in orders] == ["build_structure"], (
            "the curriculum stopped at the satisfied extractor stage; the factory "
            "stage must still be reached"
        )
        assert env.curriculum_order is not None
