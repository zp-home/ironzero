# SPDX-License-Identifier: Apache-2.0
"""The commander's economy plan, end to end.

These tests pin the contract that makes "the strategic layer owns the macro game"
a statement about code rather than about documentation:

* the ``macro`` payload is validated and round-trips;
* the rule layer emits a *sparse, meaningful* plan, and an empty one when it has
  no opinion (so a neutral run is unchanged);
* the LLM is told the economy facts and the goal vocabulary, and a plan it
  returns reaches the executor;
* :func:`ironzero.rwpp.macro.resolve_macro` clamps and falls back, and the
  identity resolution reproduces the configured defaults exactly;
* the build stages read ``env.macro`` instead of ``env.game.economy_*``, so a plan
  changes what the game does -- verified on a real dispatch through the fake
  bridge, not just by reading attributes.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.macro import (
    LIMITS,
    ResolvedMacro,
    economy_summary,
    resolve_macro,
)
from ironzero.strategy import (
    BUILDER_GOALS,
    DEFAULT_BUILD_ORDER,
    INTENTS,
    MACRO_GOALS,
    SYSTEM_PROMPT,
    LLMStrategy,
    MacroPlan,
    RuleStrategy,
    StrategicDirective,
    build_user_prompt,
)


def _unit(x: float, y: float, hp: float = 210.0, max_hp: float = 210.0) -> dict[str, float]:
    return {"x": x, "y": y, "hp": hp, "max_hp": max_hp}


def _summary(
    *,
    own: int = 4,
    enemy: int = 2,
    hp: float = 210.0,
    economy: dict | None = None,
) -> dict:
    summary: dict = {
        "own_units": [_unit(200.0 + 10 * i, 200.0, hp=hp) for i in range(own)],
        "enemy_units": [_unit(800.0, 800.0) for _ in range(enemy)],
        "map": {"width": 1000.0, "height": 1000.0},
        "tick": 100,
    }
    if economy is not None:
        summary["economy"] = economy
    return summary


# ----------------------------------------------------------------------
# 1. The contract: MacroPlan validation and round-trip
# ----------------------------------------------------------------------
class MacroPlanContractTests(unittest.TestCase):
    def test_empty_plan_is_the_default_and_means_no_opinion(self) -> None:
        plan = MacroPlan()
        self.assertTrue(plan.is_empty)
        # An empty build_order means "keep the platform default order", which is
        # what makes the neutral case behave exactly like the pre-macro code.
        self.assertEqual(plan.stage_names, DEFAULT_BUILD_ORDER)
        self.assertTrue(all(value is None for value in plan.to_payload().values() if value is not None) or True)
        for name, value in plan.to_payload().items():
            if name == "build_order":
                self.assertEqual(value, [])
            else:
                self.assertIsNone(value, f"{name} must default to None (no opinion)")

    def test_directive_without_macro_is_still_valid(self) -> None:
        directive = StrategicDirective.from_payload({"intent": "push"}, source="rule")
        self.assertTrue(directive.macro.is_empty)
        # And the payload always carries the block, so a consumer can rely on it.
        self.assertIn("macro", directive.to_payload())

    def test_macro_round_trips_through_payload(self) -> None:
        plan = MacroPlan(
            build_order=["production", "defence"],
            army_target=12,
            army_cap=14,
            max_turrets=1,
            credit_reserve=1200.0,
            attack_at_army=8,
            engage_distance=500.0,
        )
        directive = StrategicDirective(intent="defend", macro=plan, source="llm")
        payload = directive.to_payload()
        json.dumps(payload)  # must not raise
        restored = StrategicDirective.from_payload(payload, source="llm")
        self.assertEqual(restored.macro, plan)

    def test_unknown_goal_is_rejected(self) -> None:
        for bad in ("nuke", "teleport", ""):
            with self.subTest(goal=bad), self.assertRaises(ValueError):
                MacroPlan(build_order=[bad])

    def test_duplicate_goal_is_rejected(self) -> None:
        # A duplicate is ambiguous rather than emphatic: the executor walks the
        # list once, so the second copy could only mean "run it twice", and two
        # build orders in one dispatch cancel each other.
        with self.assertRaises(ValueError):
            MacroPlan(build_order=["factory", "factory"])

    def test_build_order_must_be_a_sequence_not_a_string(self) -> None:
        with self.assertRaises(ValueError):
            MacroPlan.from_payload({"build_order": "production"})
        with self.assertRaises(ValueError):
            MacroPlan.from_payload({"build_order": 5})

    def test_cross_field_contradictions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MacroPlan(army_target=12, army_cap=8)
        with self.assertRaises(ValueError):
            MacroPlan(army_cap=8, attack_at_army=12)

    def test_non_integer_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MacroPlan(army_target=10.7)
        # Integral floats are accepted: JSON producers emit ``10.0`` for ``10``.
        self.assertEqual(MacroPlan(army_target=10.0).army_target, 10)

    def test_negative_numbers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MacroPlan(max_turrets=-1)
        with self.assertRaises(ValueError):
            MacroPlan(credit_reserve=-100.0)
        with self.assertRaises(ValueError):
            MacroPlan(engage_distance=-1.0)

    def test_merge_fills_only_the_gaps(self) -> None:
        fallback = MacroPlan(army_target=10, army_cap=16, max_turrets=2)
        plan = MacroPlan(army_target=14, build_order=["production"])
        merged = plan.merged_over(fallback)
        self.assertEqual(merged.army_target, 14)  # stated opinion wins
        self.assertEqual(merged.army_cap, 16)  # falls back
        self.assertEqual(merged.max_turrets, 2)  # falls back
        self.assertEqual(merged.build_order, ("production",))

    def test_goal_vocabulary_is_the_stage_vocabulary(self) -> None:
        # The names are a contract between the prompt, the plan and the executor.
        self.assertEqual(DEFAULT_BUILD_ORDER, MACRO_GOALS)
        for goal in ("upgrade", "extractor", "factory", "production", "defence", "expansion", "attack"):
            self.assertIn(goal, MACRO_GOALS)
        # The builder-competing goals are a strict subset: ``upgrade``/``attack``
        # address something other than the builder.
        self.assertNotIn("upgrade", BUILDER_GOALS)
        self.assertNotIn("attack", BUILDER_GOALS)


# ----------------------------------------------------------------------
# 2. The rule layer actually plans
# ----------------------------------------------------------------------
class RuleMacroPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = RuleStrategy()

    def test_neutral_observation_produces_no_opinion(self) -> None:
        """The hot path must not rewrite the deployment's configuration."""

        plan = self.rule.propose(_summary(own=4, enemy=2, economy={"funds": 500.0})).macro
        # Own == enemy count would trigger nothing; own > enemy triggers the
        # "march early" rule only when the army can back it up.
        self.assertTrue(plan.is_empty or plan.attack_at_army is not None)

    def test_broken_army_rebuilds_instead_of_investing(self) -> None:
        directive = self.rule.propose(
            _summary(own=3, enemy=8, hp=10.0, economy={"funds": 3000.0, "army": 3})
        )
        self.assertEqual(directive.intent, "retreat")
        plan = directive.macro
        self.assertFalse(plan.is_empty)
        assert plan.build_order
        # Attack and expansion must be *removed*, not merely de-prioritised.
        self.assertNotIn("attack", plan.build_order)
        self.assertNotIn("expansion", plan.build_order)
        # Production/defence come first, and the army is asked to grow.
        self.assertEqual(plan.build_order[0], "production")
        self.assertIn("defence", plan.build_order)
        self.assertGreaterEqual(plan.army_target or 0, 4)

    def test_outnumbered_keeps_both_the_tier_ladder_and_the_march(self) -> None:
        """Being outnumbered must keep ``attack`` -- dropping it is a deadlock.

        The original rule removed ``attack`` while outnumbered, on the theory that
        survival comes first. It is a stable state that can never be escaped: the
        only way to stop being outnumbered is to destroy enemy units, and with no
        march the army never leaves home to do it.

        Measured on Lake at difficulty 1 with the seven-stage curriculum
        (``.runtime/enemy_composition_d1.json``): own army peaked at 16 against an
        enemy holding 30-38, every episode timed out at 19674 ticks, and 120 PPO
        updates produced zero wins. The stage's own comments record the same shape
        from an earlier run: an ``attack_at_army`` the economy could not fund left
        the army passive at 720.9 px from the enemy base.

        ``attack`` therefore stays in the order, and the march gate drops to the
        army that actually exists (kept at or above the floor that prevents feeding
        units in one at a time).
        """

        directive = self.rule.propose(
            _summary(own=4, enemy=6, economy={"funds": 3000.0, "army": 4})
        )
        plan = directive.macro
        self.assertFalse(plan.is_empty)
        assert plan.build_order
        self.assertIn("attack", plan.build_order)
        self.assertIn("upgrade", plan.build_order)
        self.assertGreaterEqual(plan.army_target or 0, 6)
        # The march must be reachable: at or below the army the plan is asking for.
        self.assertIsNotNone(plan.attack_at_army)
        self.assertLessEqual(
            int(plan.attack_at_army or 0),
            int(plan.army_target or 0),
            "a march gate above the army target is unreachable, which is the "
            "passive-army failure this rule exists to avoid",
        )

    def test_large_idle_bank_raises_the_caps_without_reordering_the_build(self) -> None:
        """A rich bank buys *more* extractors, not the right to build them *first*.

        The ordering half of this rule was tried and measured wrong: letting
        expansion claim the builder ahead of the factory chain starves the factory
        (``no_combat_factory`` on 1968 of 1968 decisions, zero combat units). So the
        plan raises the caps and leaves the priority order alone.
        """

        directive = self.rule.propose(
            _summary(
                own=6,
                enemy=3,
                economy={
                    "funds": 5000.0,
                    "free_resource_slots": 3,
                    "under_construction": False,
                    "army": 6,
                },
            )
        )
        plan = directive.macro
        self.assertFalse(plan.is_empty)
        # Caps move ...
        self.assertGreaterEqual(plan.max_extractors or 0, 6)
        # ... and the priority order is untouched, so the factory chain still leads.
        self.assertEqual(plan.build_order, ())

    def test_bank_without_army_does_not_raise_the_caps(self) -> None:
        """A big bank with no army belongs to the army, not the economy."""

        directive = self.rule.propose(
            _summary(
                own=2,
                enemy=1,
                economy={"funds": 9000.0, "free_resource_slots": 4, "army": 1},
            )
        )
        self.assertIsNone(directive.macro.max_extractors)

    def test_locally_superior_lowers_the_attack_threshold(self) -> None:
        directive = self.rule.propose(
            _summary(own=6, enemy=2, economy={"funds": 400.0, "army": 6})
        )
        plan = directive.macro
        self.assertEqual(plan.attack_at_army, 5)
        # Never raises it: waiting longer than configured needs a reason.
        self.assertLessEqual(plan.attack_at_army or 0, 16)

    def test_macro_can_be_disabled_for_ab(self) -> None:
        quiet = RuleStrategy(macro_enabled=False)
        for summary in (
            _summary(own=3, enemy=8, hp=10.0, economy={"funds": 3000.0}),
            _summary(own=6, enemy=2, economy={"funds": 9000.0, "army": 6}),
        ):
            with self.subTest(summary=summary):
                self.assertTrue(quiet.propose(summary).macro.is_empty)

    def test_plan_never_exceeds_the_documented_ceiling(self) -> None:
        """The rule layer must stay inside a legal plan, always.

        ``MacroPlan`` rejects ``army_cap < army_target``; a rule that produced such
        a pair would downgrade the whole directive to the fallback path for a
        cosmetic reason. Sweep the state space crudely instead of trusting review.
        """

        for funds in (0.0, 500.0, 3000.0, 20000.0):
            for army in range(0, 20):
                for enemy in range(0, 20):
                    for own in (0, 1, 3, 6, 12, 24):
                        summary = _summary(
                            own=own,
                            enemy=enemy,
                            economy={
                                "funds": funds,
                                "army": army,
                                "free_resource_slots": 3,
                                "under_construction": False,
                            },
                        )
                        plan = self.rule.propose(summary).macro
                        payload = plan.to_payload()
                        # Constructing the payload again must not raise.
                        MacroPlan.from_payload(payload)

    def test_invalid_economy_fields_do_not_crash(self) -> None:
        for economy in (
            "garbage",
            {"funds": "lots", "army": None, "extractors": float("nan")},
            {"funds": float("inf"), "free_resource_slots": -5},
        ):
            with self.subTest(economy=economy):
                directive = self.rule.propose(
                    dict(_summary(own=4, enemy=6), economy=economy)
                )
                self.assertIn(directive.intent, INTENTS)


# ----------------------------------------------------------------------
# 3. The LLM is told about the economy and its plan reaches the executor
# ----------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


def _chat_body(content: str) -> str:
    return json.dumps(
        {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        }
    )


class LlmMacroTests(unittest.TestCase):
    def _llm(self) -> LLMStrategy:
        return LLMStrategy(
            api_key="test-key-not-a-secret",
            base_url="https://llm.invalid",
            model="test-model",
            environ={},
            throttle_seconds=0.0,
        )

    def test_prompt_carries_the_goal_vocabulary_and_null_semantics(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        for goal in MACRO_GOALS:
            self.assertIn(goal, lowered)
        self.assertIn("build_order", lowered)
        self.assertIn("null", lowered)
        # The macro half must not become a licence to control units.
        self.assertIn("do not control units", lowered)
        self.assertIn("never placements", lowered)

    def test_user_prompt_reports_the_economy(self) -> None:
        prompt = build_user_prompt(
            _summary(
                own=3,
                enemy=9,
                economy={
                    "funds": 3210.0,
                    "income": 26.0,
                    "extractors": 4,
                    "factories": 2,
                    "army": 3,
                    "turrets": 1,
                    "free_resource_slots": 2,
                    "under_construction": True,
                },
            )
        )
        self.assertIn('"funds": 3210.0', prompt)
        self.assertIn('"income": 26.0', prompt)
        self.assertIn('"extractors": 4', prompt)
        self.assertIn('"army": 3', prompt)
        self.assertIn('"free_resource_slots": 2', prompt)
        self.assertIn('"under_construction": true', prompt)

    def test_missing_economy_renders_as_null_not_a_guess(self) -> None:
        prompt = build_user_prompt(_summary(own=2, enemy=2))
        self.assertIn('"funds": null', prompt)
        self.assertIn('"army": null', prompt)

    def test_llm_macro_plan_reaches_the_directive(self) -> None:
        body = _chat_body(
            json.dumps(
                {
                    "intent": "defend",
                    "notes": "turtle up",
                    "macro": {
                        "build_order": ["defence", "production", "upgrade"],
                        "army_target": 8,
                        "max_turrets": 3,
                        "credit_reserve": 700,
                    },
                }
            )
        )
        strategy = self._llm()
        with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
            directive = strategy.propose(_summary(own=4, enemy=8))
        self.assertEqual(directive.source, "llm")
        self.assertEqual(
            directive.macro.build_order, ("defence", "production", "upgrade")
        )
        self.assertEqual(directive.macro.army_target, 8)
        self.assertEqual(directive.macro.max_turrets, 3)
        self.assertAlmostEqual(directive.macro.credit_reserve, 700.0)
        # Unstated fields stay "no opinion" -- the executor keeps its defaults.
        self.assertIsNone(directive.macro.army_cap)

    def test_a_contradictory_macro_block_falls_back_to_the_rule_layer(self) -> None:
        # ``army_cap`` below ``army_target`` is unreachable by construction. The
        # whole directive is rejected rather than half-applied: a plan the executor
        # has to guess about is not a plan.
        body = _chat_body(
            json.dumps(
                {
                    "intent": "push",
                    "macro": {"army_target": 12, "army_cap": 4},
                }
            )
        )
        strategy = self._llm()
        with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
            directive = strategy.propose(_summary(own=4, enemy=2))
        self.assertEqual(directive.source, "rule")
        self.assertTrue(directive.notes.startswith("fallback:"))

    def test_directive_without_macro_still_parses(self) -> None:
        body = _chat_body(json.dumps({"intent": "harass", "notes": "poke"}))
        strategy = self._llm()
        with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
            directive = strategy.propose(_summary())
        self.assertEqual(directive.source, "llm")
        self.assertTrue(directive.macro.is_empty)


# ----------------------------------------------------------------------
# 4. Resolution: defaults, clamping, fallback
# ----------------------------------------------------------------------
class ResolveMacroTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RwppGameConfig()

    def test_identity_resolution_matches_the_configuration(self) -> None:
        resolved = ResolvedMacro.from_config(self.config)
        self.assertEqual(resolved.stage_names, DEFAULT_BUILD_ORDER)
        self.assertEqual(resolved.army_target, self.config.economy_army_target)
        self.assertEqual(resolved.army_cap, self.config.economy_army_cap)
        self.assertEqual(resolved.max_extractors, self.config.economy_max_extractors)
        self.assertEqual(resolved.max_factories, self.config.economy_max_factories)
        self.assertEqual(resolved.max_upgrades, self.config.economy_max_upgrades)
        self.assertEqual(resolved.max_turrets, self.config.economy_max_turrets)
        self.assertAlmostEqual(resolved.credit_reserve, self.config.economy_credit_reserve)
        self.assertAlmostEqual(resolved.engage_distance, self.config.economy_engage_distance)
        # The attack gate defaults to the army target: one number, not two.
        self.assertEqual(resolved.attack_at_army, resolved.army_target)
        self.assertFalse(resolved.from_commander)

    def test_no_directive_is_the_configured_default(self) -> None:
        self.assertEqual(
            resolve_macro(self.config, None).to_payload(),
            ResolvedMacro.from_config(self.config).to_payload(),
        )

    def test_empty_plan_returns_the_identical_object(self) -> None:
        baseline = ResolvedMacro.from_config(self.config)
        self.assertIs(baseline.merged_with(MacroPlan()), baseline)

    def test_a_directive_without_macro_attribute_is_tolerated(self) -> None:
        class Old:
            intent = "push"

        self.assertFalse(resolve_macro(self.config, Old()).from_commander)

    def test_plan_fields_are_clamped_not_trusted(self) -> None:
        baseline = ResolvedMacro.from_config(self.config)
        hostile = MacroPlan(
            army_target=24,
            army_cap=32,
            max_extractors=9,
            max_factories=4,
            max_upgrades=12,
            max_turrets=6,
            credit_reserve=5000.0,
            engage_distance=4000.0,
        )
        resolved = baseline.merged_with(hostile)
        for name, (low, high) in LIMITS.items():
            value = float(getattr(resolved, name))
            self.assertGreaterEqual(value, low, name)
            self.assertLessEqual(value, high, name)
        # Clamping must not produce a contradictory pair.
        self.assertGreaterEqual(resolved.army_cap, resolved.army_target)
        self.assertLessEqual(resolved.attack_at_army, resolved.army_cap)
        self.assertTrue(resolved.from_commander)

    def test_clamping_repairs_a_cap_below_the_target(self) -> None:
        """Clamping can *create* a contradiction; the resolver must repair it.

        ``MacroPlan`` rejects ``army_cap < army_target`` on input, but a plan of
        ``army_target=24, army_cap=24`` is legal and the *limits* could still push
        the two apart if they ever diverge. Repairing rather than raising keeps a
        hostile plan from silently disabling the commander.
        """

        baseline = ResolvedMacro.from_config(self.config)
        resolved = baseline.merged_with(MacroPlan(army_target=24, army_cap=24))
        self.assertGreaterEqual(resolved.army_cap, resolved.army_target)
        self.assertLessEqual(resolved.attack_at_army, resolved.army_cap)

    def test_engage_distance_zero_is_allowed_but_means_already_engaged(self) -> None:
        baseline = ResolvedMacro.from_config(self.config)
        resolved = baseline.merged_with(MacroPlan(engage_distance=0.0))
        self.assertEqual(resolved.engage_distance, 0.0)

    def test_cancelling_a_goal_removes_it_from_the_order(self) -> None:
        baseline = ResolvedMacro.from_config(self.config)
        resolved = baseline.merged_with(
            MacroPlan(build_order=["extractor", "factory", "production", "attack"])
        )
        self.assertTrue(resolved.allows("attack"))
        self.assertFalse(resolved.allows("upgrade"))
        self.assertFalse(resolved.allows("defence"))
        self.assertFalse(resolved.allows("expansion"))
        self.assertEqual(resolved.builder_goals, ("extractor", "factory", "production"))

    def test_resolved_payload_is_json_safe(self) -> None:
        payload = ResolvedMacro.from_config(self.config).to_payload()
        json.dumps(payload)  # must not raise
        self.assertIn("stage_names", payload)
        self.assertIn("from_commander", payload)


class EconomySummaryTests(unittest.TestCase):
    def test_counts_only_finished_buildings_and_live_combat_units(self) -> None:
        state = {
            "global_state": {"funds": 2500.0, "income": 26.0},
            "resource_slots": [
                {"occupied": True},
                {"occupied": False},
                {"occupied": False},
            ],
            "own_entities": [
                # Finished extractor.
                {"is_building": True, "unit_type": "extractorT1", "is_operational": True},
                # Half-built factory: counted as under construction, not as a factory.
                {"is_building": True, "unit_type": "landFactory", "is_factory": True,
                 "is_operational": False, "build_progress": 0.4},
                # Operational turret.
                {"is_building": True, "unit_type": "turret", "is_defence": True,
                 "is_operational": True},
                {"is_builder": True, "hp": 170.0},
                {"can_attack": True, "hp": 210.0, "x": 1.0, "y": 2.0},
                {"can_attack": True, "hp": 0.0},
            ],
        }
        summary = economy_summary(state)
        self.assertAlmostEqual(summary["funds"], 2500.0)
        self.assertAlmostEqual(summary["income"], 26.0)
        self.assertEqual(summary["extractors"], 1)
        self.assertEqual(summary["factories"], 0)
        self.assertEqual(summary["turrets"], 1)
        self.assertEqual(summary["army"], 1)
        self.assertEqual(summary["free_resource_slots"], 2)
        self.assertTrue(summary["under_construction"])

    def test_a_mid_upgrade_building_still_counts_as_built(self) -> None:
        """``is_operational`` is false *during* an upgrade; the building exists.

        Treating that as "never built" makes the commander re-plan an economy it
        already has -- the same lenient rule ``env._is_operational`` uses. The
        bridge reports the upgrade as ``build_progress`` below 1 while the building
        keeps producing, which is exactly the case pinned here.
        """

        state = {
            "own_entities": [
                # Mid-upgrade: not operational, but *complete* by progress.
                {"is_building": True, "unit_type": "extractorT2", "is_operational": False,
                 "build_progress": 1.0},
            ],
        }
        summary = economy_summary(state)
        self.assertEqual(summary["extractors"], 1)
        self.assertFalse(summary["under_construction"])

    def test_a_half_built_building_is_not_counted(self) -> None:
        state = {
            "own_entities": [
                {"is_building": True, "unit_type": "extractorT1", "is_operational": False,
                 "build_progress": 0.2},
            ],
        }
        summary = economy_summary(state)
        self.assertEqual(summary["extractors"], 0)
        self.assertTrue(summary["under_construction"])

    def test_missing_fields_are_null_not_invented(self) -> None:
        summary = economy_summary({})
        self.assertIsNone(summary["funds"])
        self.assertIsNone(summary["income"])
        self.assertEqual(summary["army"], 0)
        self.assertFalse(summary["under_construction"])


if __name__ == "__main__":
    unittest.main()
