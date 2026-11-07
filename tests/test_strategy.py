# SPDX-License-Identifier: Apache-2.0
"""Tests for the IronZero LLM strategic layer.

Every test that configures an API key patches ``urllib.request.urlopen``; no
test opens a real network connection. Run with::

    $env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_strategy.py"
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from typing import Any
from unittest.mock import patch

from ironzero.strategy import (
    DEFAULT_RETREAT_HP_THRESHOLD,
    DIRECTIVE_SCHEMA_VERSION,
    FALLBACK_NOTE_PREFIX,
    SYSTEM_PROMPT,
    LLMStrategy,
    RuleStrategy,
    StrategicDirective,
    build_strategy,
    build_user_prompt,
)

TEST_KEY = "unit-test-key-not-a-real-secret"


def _unit(x: float, y: float, hp: float = 210.0, max_hp: float = 210.0) -> dict[str, float]:
    return {"x": x, "y": y, "hp": hp, "max_hp": max_hp}


def _summary(
    *,
    own: list[dict[str, float]] | None = None,
    enemy: list[dict[str, float]] | None = None,
    width: float = 1000.0,
    height: float = 1000.0,
    tick: int = 0,
) -> dict[str, Any]:
    return {
        "own_units": own or [],
        "enemy_units": enemy or [],
        "map": {"width": width, "height": height},
        "tick": tick,
    }


class FakeResponse:
    """Minimal stand-in for the object returned by ``urllib.request.urlopen``."""

    def __init__(self, payload: str | bytes) -> None:
        self._payload = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


def _chat_body(content: str) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        }
    )


VALID_PAYLOAD: dict[str, Any] = {
    "intent": "push",
    "attack_region": [500.0, 400.0, 400.0],
    "rally_point": [100.0, 200.0],
    "retreat_hp_threshold": 0.4,
    "priority_target": "artillery",
    "notes": "take the middle",
    "created_at": 1234.5,
}


def _make_llm(**overrides: Any) -> LLMStrategy:
    options: dict[str, Any] = {
        "api_key": TEST_KEY,
        "base_url": "https://llm.invalid",
        "model": "test-model",
        "environ": {},
        "throttle_seconds": 0.0,
    }
    options.update(overrides)
    return LLMStrategy(**options)


class DirectivePayloadTests(unittest.TestCase):
    def test_from_payload_accepts_valid_payload(self) -> None:
        directive = StrategicDirective.from_payload(dict(VALID_PAYLOAD), source="llm")
        self.assertEqual(directive.intent, "push")
        self.assertEqual(directive.attack_region, (500.0, 400.0, 400.0))
        self.assertEqual(directive.rally_point, (100.0, 200.0))
        self.assertAlmostEqual(directive.retreat_hp_threshold, 0.4)
        self.assertEqual(directive.priority_target, "artillery")
        self.assertEqual(directive.notes, "take the middle")
        self.assertEqual(directive.source, "llm")
        self.assertAlmostEqual(directive.created_at, 1234.5)

    def test_from_payload_applies_defaults(self) -> None:
        directive = StrategicDirective.from_payload({"intent": "defend"}, source="rule")
        self.assertIsNone(directive.attack_region)
        self.assertIsNone(directive.rally_point)
        self.assertAlmostEqual(directive.retreat_hp_threshold, DEFAULT_RETREAT_HP_THRESHOLD)
        self.assertEqual(directive.priority_target, "nearest")
        self.assertEqual(directive.notes, "")
        self.assertEqual(directive.source, "rule")
        self.assertGreater(directive.created_at, 0.0)

    def test_payload_round_trip_is_json_serializable(self) -> None:
        directive = StrategicDirective.from_payload(dict(VALID_PAYLOAD), source="llm")
        payload = directive.to_payload()
        self.assertEqual(payload["schema_version"], DIRECTIVE_SCHEMA_VERSION)
        self.assertEqual(payload["attack_region"], [500.0, 400.0, 400.0])
        self.assertEqual(payload["rally_point"], [100.0, 200.0])
        self.assertEqual(payload["source"], "llm")
        json.dumps(payload)  # must not raise

        restored = StrategicDirective.from_payload(payload, source="llm")
        self.assertEqual(restored.intent, directive.intent)
        self.assertEqual(restored.attack_region, directive.attack_region)
        self.assertEqual(restored.rally_point, directive.rally_point)
        self.assertAlmostEqual(restored.created_at, directive.created_at)

    def test_missing_or_unknown_intent_is_rejected(self) -> None:
        for payload in ({}, {"intent": None}, {"intent": "dance"}, {"intent": 7}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                StrategicDirective.from_payload(payload, source="llm")

    def test_attack_region_validation(self) -> None:
        bad_regions = (
            [100.0, 100.0],
            [100.0, 100.0, 10.0, 20.0],
            [100.0, 100.0, 0.0],
            [100.0, 100.0, -5.0],
            [100.0, 100.0, "400"],
            [float("nan"), 10.0, 100.0],
            [10.0, float("inf"), 100.0],
            "not-a-region",
        )
        for region in bad_regions:
            with self.subTest(region=region), self.assertRaises(ValueError):
                StrategicDirective.from_payload(
                    {"intent": "push", "attack_region": region}, source="llm"
                )

    def test_rally_point_validation(self) -> None:
        bad_points = ([1.0], [1.0, 2.0, 3.0], ["a", 2.0], [float("nan"), 1.0], 42)
        for point in bad_points:
            with self.subTest(point=point), self.assertRaises(ValueError):
                StrategicDirective.from_payload(
                    {"intent": "defend", "rally_point": point}, source="llm"
                )

    def test_threshold_validation(self) -> None:
        for threshold in (-0.1, 1.1, "0.5", None, float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                StrategicDirective.from_payload(
                    {"intent": "push", "retreat_hp_threshold": threshold}, source="llm"
                )

    def test_other_field_validation(self) -> None:
        cases = (
            {"intent": "push", "priority_target": ""},
            {"intent": "push", "priority_target": 3},
            {"intent": "push", "notes": 3},
            {"intent": "push", "created_at": float("nan")},
            {"intent": "push", "schema_version": "ironzero.strategy/999"},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                StrategicDirective.from_payload(payload, source="llm")
        with self.assertRaises(ValueError):
            StrategicDirective.from_payload({"intent": "push"}, source="human")

    def test_direct_construction_validates_fields(self) -> None:
        with self.assertRaises(ValueError):
            StrategicDirective(intent="dance")
        with self.assertRaises(ValueError):
            StrategicDirective(intent="push", retreat_hp_threshold=2.0)
        with self.assertRaises(ValueError):
            StrategicDirective(intent="push", source="human")
        with self.assertRaises(ValueError):
            StrategicDirective(intent="push", attack_region=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            StrategicDirective(intent="push", rally_point=(float("nan"), 0.0))


class RuleStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = RuleStrategy()

    def test_low_hp_retreats_towards_own_half(self) -> None:
        directive = self.rule.propose(
            _summary(
                own=[_unit(400.0, 300.0, hp=20.0), _unit(600.0, 300.0, hp=20.0)],
                enemy=[_unit(800.0, 800.0) for _ in range(6)],
            )
        )
        self.assertEqual(directive.intent, "retreat")
        self.assertEqual(directive.source, "rule")
        self.assertIsNotNone(directive.rally_point)
        assert directive.rally_point is not None
        x, y = directive.rally_point
        self.assertTrue(0.0 <= x <= 1000.0)
        self.assertTrue(0.0 <= y <= 1000.0)
        # Own centroid is in the upper half, so the rally line is deep there.
        self.assertLess(y, 500.0)
        self.assertIsNone(directive.attack_region)

    def test_equal_or_superior_numbers_push(self) -> None:
        equal = self.rule.propose(
            _summary(
                own=[_unit(100.0, 100.0) for _ in range(4)],
                enemy=[_unit(800.0, 800.0) for _ in range(4)],
            )
        )
        superior = self.rule.propose(
            _summary(
                own=[_unit(100.0, 100.0) for _ in range(6)],
                enemy=[_unit(800.0, 800.0) for _ in range(2)],
            )
        )
        self.assertEqual(equal.intent, "push")
        self.assertEqual(superior.intent, "push")

    def test_push_region_targets_enemy_centroid(self) -> None:
        directive = self.rule.propose(
            _summary(
                own=[_unit(200.0, 200.0) for _ in range(4)],
                enemy=[_unit(700.0, 700.0), _unit(900.0, 900.0)],
            )
        )
        self.assertEqual(directive.intent, "push")
        assert directive.attack_region is not None
        x, y, radius = directive.attack_region
        self.assertAlmostEqual(x, 800.0)
        self.assertAlmostEqual(y, 800.0)
        self.assertAlmostEqual(radius, 400.0)

    def test_outnumbered_defends_at_own_centroid(self) -> None:
        directive = self.rule.propose(
            _summary(
                own=[_unit(300.0, 400.0), _unit(500.0, 400.0)],
                enemy=[_unit(800.0, 800.0) for _ in range(5)],
            )
        )
        self.assertEqual(directive.intent, "defend")
        self.assertEqual(directive.rally_point, (400.0, 400.0))
        self.assertIsNone(directive.attack_region)

    def test_empty_or_missing_summary_defends(self) -> None:
        for summary in (_summary(), None, {}, {"own_units": "garbage"}):
            with self.subTest(summary=summary):
                directive = self.rule.propose(summary)  # type: ignore[arg-type]
                self.assertEqual(directive.intent, "defend")
                self.assertIsNone(directive.attack_region)
                self.assertIsNone(directive.rally_point)

    def test_custom_threshold(self) -> None:
        cautious = RuleStrategy(retreat_hp_threshold=0.9)
        directive = cautious.propose(
            _summary(
                own=[_unit(100.0, 100.0, hp=180.0)],
                enemy=[_unit(900.0, 900.0)],
            )
        )
        self.assertEqual(directive.intent, "retreat")
        self.assertAlmostEqual(directive.retreat_hp_threshold, 0.9)

    def test_priority_target_is_configurable(self) -> None:
        directive = RuleStrategy(priority_target="artillery").propose(
            _summary(own=[_unit(100.0, 100.0)], enemy=[_unit(900.0, 900.0)])
        )
        self.assertEqual(directive.priority_target, "artillery")

    def test_all_outputs_are_bounded_valid_directives(self) -> None:
        scenarios = (
            _summary(own=[_unit(-50.0, -50.0)], enemy=[]),
            _summary(own=[_unit(1200.0, 1500.0)], enemy=[_unit(2000.0, 2000.0)]),
            _summary(
                own=[_unit(10.0, 990.0, hp=5.0)],
                enemy=[_unit(500.0, 500.0)],
                width=500.0,
                height=300.0,
            ),
            _summary(own=[_unit("bad", 100.0)], enemy=[_unit(100.0, 100.0)]),  # type: ignore[list-item]
        )
        for summary in scenarios:
            with self.subTest(summary=summary):
                directive = self.rule.propose(summary)
                self.assertIn(directive.intent, ("push", "defend", "retreat"))
                self.assertEqual(directive.source, "rule")
                self.assertGreater(directive.created_at, 0.0)
                width = float(summary["map"]["width"])
                height = float(summary["map"]["height"])
                if directive.attack_region is not None:
                    x, y, radius = directive.attack_region
                    self.assertTrue(0.0 <= x <= width)
                    self.assertTrue(0.0 <= y <= height)
                    self.assertGreater(radius, 0.0)
                if directive.rally_point is not None:
                    x, y = directive.rally_point
                    self.assertTrue(0.0 <= x <= width)
                    self.assertTrue(0.0 <= y <= height)

    def test_invalid_units_do_not_crash(self) -> None:
        directive = self.rule.propose(
            {
                "own_units": [None, {"x": float("nan"), "y": 1.0}, {"x": "x", "y": 2.0}],
                "enemy_units": "not-a-list",
                "map": {"width": -5.0, "height": "tall"},
                "tick": "soon",
            }
        )
        self.assertEqual(directive.intent, "defend")
        self.assertEqual(directive.source, "rule")


class LLMStrategyTests(unittest.TestCase):
    def test_enabled_requires_api_key(self) -> None:
        self.assertFalse(LLMStrategy(environ={}).enabled)
        self.assertFalse(LLMStrategy(environ={"IRONZERO_LLM_API_KEY": ""}).enabled)
        self.assertTrue(LLMStrategy(environ={"IRONZERO_LLM_API_KEY": "env-key"}).enabled)
        self.assertTrue(_make_llm().enabled)

    def test_configuration_comes_from_environment(self) -> None:
        strategy = LLMStrategy(
            environ={
                "IRONZERO_LLM_API_KEY": "env-key",
                "IRONZERO_LLM_BASE_URL": "https://env.example/v1/",
                "IRONZERO_LLM_MODEL": "env-model",
            }
        )
        self.assertEqual(strategy.api_key, "env-key")
        self.assertEqual(strategy.base_url, "https://env.example/v1")
        self.assertEqual(strategy.model, "env-model")

    def test_propose_parses_valid_llm_json(self) -> None:
        strategy = _make_llm()
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(_chat_body(json.dumps(VALID_PAYLOAD))),
        ) as mocked:
            directive = strategy.propose(_summary(own=[_unit(100.0, 100.0)]))
        mocked.assert_called_once()
        self.assertEqual(directive.source, "llm")
        self.assertEqual(directive.intent, "push")
        self.assertEqual(directive.attack_region, (500.0, 400.0, 400.0))
        self.assertEqual(directive.priority_target, "artillery")

    def test_request_body_and_prompt_follow_contract(self) -> None:
        strategy = _make_llm()
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(_chat_body(json.dumps(VALID_PAYLOAD))),
        ) as mocked:
            strategy.propose(_summary(own=[_unit(100.0, 100.0)]))
        request = mocked.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
        self.assertIn("Bearer " + TEST_KEY, str(request.headers))
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])
        system_prompt = body["messages"][0]["content"]
        user_prompt = body["messages"][1]["content"]
        self.assertEqual(system_prompt, SYSTEM_PROMPT)
        self.assertIn("JSON", system_prompt)
        self.assertIn("unit-level", system_prompt)
        self.assertIn("attack_region", system_prompt)
        self.assertIn("Battlefield summary", user_prompt)
        self.assertIn('"count": 1', user_prompt)
        self.assertNotIn(TEST_KEY, user_prompt)

    def test_accepts_markdown_fenced_json(self) -> None:
        fenced = "```json\n" + json.dumps({"intent": "regroup", "notes": "regroup"}) + "\n```"
        strategy = _make_llm()
        with patch("urllib.request.urlopen", return_value=FakeResponse(_chat_body(fenced))):
            directive = strategy.propose(_summary(own=[_unit(100.0, 100.0)]))
        self.assertEqual(directive.source, "llm")
        self.assertEqual(directive.intent, "regroup")

    def test_throttle_reuses_previous_directive_without_network(self) -> None:
        strategy = _make_llm(throttle_seconds=60.0)
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(_chat_body(json.dumps(VALID_PAYLOAD))),
        ) as mocked:
            first = strategy.propose(_summary())
            second = strategy.propose(_summary())
        mocked.assert_called_once()
        self.assertIs(first, second)

    def test_zero_throttle_calls_network_every_time(self) -> None:
        strategy = _make_llm(throttle_seconds=0.0)
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(_chat_body(json.dumps(VALID_PAYLOAD))),
        ) as mocked:
            strategy.propose(_summary())
            strategy.propose(_summary())
        self.assertEqual(mocked.call_count, 2)

    def test_invalid_json_falls_back_to_rule(self) -> None:
        strategy = _make_llm()
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(_chat_body("I will attack with everything!")),
        ):
            directive = strategy.propose(_summary(own=[_unit(100.0, 100.0) for _ in range(2)]))
        self.assertEqual(directive.source, "rule")
        self.assertTrue(directive.notes.startswith(FALLBACK_NOTE_PREFIX))
        self.assertIn("JSON", directive.notes)

    def test_network_error_falls_back_to_rule(self) -> None:
        strategy = _make_llm()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            directive = strategy.propose(_summary())
        self.assertEqual(directive.source, "rule")
        self.assertTrue(directive.notes.startswith(FALLBACK_NOTE_PREFIX))
        self.assertIn("URLError", directive.notes)

    def test_timeout_falls_back_to_rule(self) -> None:
        strategy = _make_llm()
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            directive = strategy.propose(_summary())
        self.assertEqual(directive.source, "rule")
        self.assertIn("TimeoutError", directive.notes)

    def test_invalid_directive_falls_back_to_rule(self) -> None:
        strategy = _make_llm()
        bad = json.dumps({"intent": "teleport", "attack_region": [1.0, 2.0, -1.0]})
        with patch("urllib.request.urlopen", return_value=FakeResponse(_chat_body(bad))):
            directive = strategy.propose(_summary())
        self.assertEqual(directive.source, "rule")
        self.assertTrue(directive.notes.startswith(FALLBACK_NOTE_PREFIX))

    def test_malformed_completion_falls_back_to_rule(self) -> None:
        strategy = _make_llm()
        malformed = json.dumps({"error": {"message": "rate limited"}})
        with patch("urllib.request.urlopen", return_value=FakeResponse(malformed)):
            directive = strategy.propose(_summary())
        self.assertEqual(directive.source, "rule")
        self.assertIn("fallback", directive.notes)
        self.assertIn("rate limited", directive.notes)

    def test_disabled_strategy_never_touches_network(self) -> None:
        strategy = LLMStrategy(environ={}, throttle_seconds=0.0)
        with patch("urllib.request.urlopen") as mocked:
            directive = strategy.propose(_summary())
        mocked.assert_not_called()
        self.assertEqual(directive.source, "rule")
        self.assertTrue(directive.notes.startswith("fallback:no_api_key"))

    def test_api_key_never_leaks_into_notes_or_repr(self) -> None:
        key = "sk-unit-test-secret-123456"
        strategy = _make_llm(api_key=key)
        self.assertNotIn(key, repr(strategy))
        with patch("urllib.request.urlopen", side_effect=RuntimeError(f"boom {key}")):
            directive = strategy.propose(_summary())
        self.assertNotIn(key, directive.notes)
        self.assertIn("***", directive.notes)
        self.assertEqual(directive.source, "rule")

    def test_build_strategy_selects_implementation(self) -> None:
        rule = build_strategy(False, environ={})
        self.assertIsInstance(rule, RuleStrategy)

        without_key = build_strategy(True, environ={})
        self.assertIsInstance(without_key, RuleStrategy)

        with_key = build_strategy(
            True, api_key=TEST_KEY, base_url="https://llm.invalid", environ={}
        )
        self.assertIsInstance(with_key, LLMStrategy)
        assert isinstance(with_key, LLMStrategy)
        self.assertTrue(with_key.enabled)
        self.assertIsInstance(with_key.fallback, RuleStrategy)

    def test_build_strategy_honours_rule_threshold(self) -> None:
        rule = build_strategy(False, retreat_hp_threshold=0.8, environ={})
        self.assertIsInstance(rule, RuleStrategy)
        assert isinstance(rule, RuleStrategy)
        self.assertAlmostEqual(rule.retreat_hp_threshold, 0.8)


class PromptContractTests(unittest.TestCase):
    def test_system_prompt_forbids_low_level_control(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        for phrase in ("do not control units", "unit-level orders", "json object"):
            self.assertIn(phrase, lowered)

    def test_user_prompt_is_compact_and_deterministic(self) -> None:
        summary = _summary(
            own=[_unit(100.0, 100.0), _unit(300.0, 100.0)],
            enemy=[_unit(900.0, 900.0)],
            tick=42,
        )
        prompt = build_user_prompt(summary)
        self.assertEqual(prompt, build_user_prompt(summary))
        self.assertIn('"tick": 42', prompt)
        self.assertIn('"count": 2', prompt)
        self.assertIn("JSON object only", prompt)


if __name__ == "__main__":
    unittest.main()
