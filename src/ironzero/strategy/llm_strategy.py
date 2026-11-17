# SPDX-License-Identifier: Apache-2.0
"""LLM strategic policy speaking the OpenAI-compatible Chat Completions API.

The LLM is a *commander*, not a controller: it receives a compressed
battlefield summary and returns one validated ``StrategicDirective``
(``ironzero.strategy/1``). Unit-level actions stay with the PPO tactical layer
at 2-10 Hz and the per-tick rule safety layer.

Design constraints from the architecture document (12.4 / phase 5):

* No third-party SDK -- plain ``urllib.request`` keeps the dependency surface
  small and the HTTP call mockable in tests.
* Every network/parse/validation failure degrades to :class:`RuleStrategy` with
  a ``fallback:<reason>`` note, so turning the LLM off cannot break training.
* The API key is only sent in the ``Authorization`` header. It never appears in
  prompts, logs, notes or exception messages.
* No network I/O at import or construction time; ``enabled`` only means "an
  API key is configured".
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Mapping

from .directive import DEFAULT_RETREAT_HP_THRESHOLD, StrategicDirective
from .rule_strategy import RuleStrategy

ENV_BASE_URL = "IRONZERO_LLM_BASE_URL"
ENV_API_KEY = "IRONZERO_LLM_API_KEY"
ENV_MODEL = "IRONZERO_LLM_MODEL"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_THROTTLE_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_TEMPERATURE = 0.2
#: Token budget for one directive.
#:
#: Raised from 400 when the ``macro`` block was added: the payload now carries up
#: to ten extra keys, and a truncated response is not a cosmetic failure -- it
#: fails JSON parsing and degrades the whole directive to the rule fallback, which
#: silently removes the commander from the loop.
DEFAULT_MAX_TOKENS = 700

#: Prefix used in ``StrategicDirective.notes`` when a rule fallback was used.
FALLBACK_NOTE_PREFIX = "fallback:"

MAX_FALLBACK_REASON_CHARS = 200
MAX_SUMMARY_UNITS = 32

DIRECTIVE_USER_AGENT = "ironzero-strategy/1"

_AUTH_HEADER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{4,}")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9._\-]{4,}")

SYSTEM_PROMPT = (
    "You are the strategic commander of IronZero, a Rusted Warfare 1.15 "
    "reinforcement-learning platform. You choose the high-level goal and the "
    "economy plan. A PPO tactical policy executes the fighting at 2-10 Hz, a "
    "build executor places every structure, and a rule safety layer runs every "
    "tick.\n"
    "STRICT RULES:\n"
    "1. Do NOT control units. Never output unit-level orders, per-unit lists, "
    "entity ids, build-queue entries, actions, scripts or mouse/screen "
    "coordinates for units. You name *goals* and *budgets*, never placements.\n"
    "2. Output exactly ONE JSON object and nothing else: no markdown, no code "
    "fences, no commentary.\n"
    "3. attack_region and rally_point use map pixel coordinates (x right, y "
    "down). Use null when a field is not needed.\n"
    "4. intent must be one of: push, defend, retreat, harass, regroup. Prefer "
    "retreat when the own HP ratio is below retreat_hp_threshold.\n"
    "5. attack_region is [x, y, radius] with radius > 0; rally_point is [x, y].\n"
    "6. macro.build_order sets WHICH economy goals run and in what priority. "
    "Allowed goal names: upgrade (tier up a standing building), extractor "
    "(claim one resource pool), factory (one production building), production "
    "(queue combat units), defence (one turret between you and the enemy), "
    "expansion (claim an extra pool once the factory chain stands), attack "
    "(march the army). Omit a goal to cancel it. An empty list [] means "
    "'keep the platform default order'. Never repeat a goal.\n"
    "7. Every macro field is optional. Use null (or omit it) when you have no "
    "opinion -- the executor then keeps its configured default. That is "
    "usually the correct choice; only set a field when the battlefield summary "
    "gives you a reason.\n"
    "8. macro constraints: army_cap >= army_target and attack_at_army <= "
    "army_cap. Credit figures are game credits, distances are map pixels, "
    "counts are whole numbers.\n"
    "JSON schema:\n"
    '{"intent": "push|defend|retreat|harass|regroup", '
    '"attack_region": [x, y, radius] | null, '
    '"rally_point": [x, y] | null, '
    '"retreat_hp_threshold": 0.0-1.0, '
    '"priority_target": "nearest|artillery|none", '
    '"notes": "one short sentence", '
    '"macro": {"build_order": ["upgrade","extractor","factory","production",'
    '"defence","expansion","attack"], "army_target": int|null, '
    '"army_cap": int|null, "max_extractors": int|null, '
    '"max_factories": int|null, "max_upgrades": int|null, '
    '"max_turrets": int|null, "credit_reserve": number|null, '
    '"attack_at_army": int|null, "engage_distance": number|null}}\n'
    "Do not add keys that are not listed."
)


# ----------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------
def _finite(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _map_dimensions(summary: Mapping[str, Any]) -> tuple[float, float]:
    raw_map = summary.get("map")
    if isinstance(raw_map, Mapping):
        width = _finite(raw_map.get("width"), 1000.0)
        height = _finite(raw_map.get("height"), 1000.0)
    else:
        width, height = 1000.0, 1000.0
    if width <= 0.0:
        width = 1000.0
    if height <= 0.0:
        height = 1000.0
    return width, height


def _parse_units(units: Any) -> list[tuple[float, float, float, float]]:
    parsed: list[tuple[float, float, float, float]] = []
    if not isinstance(units, (list, tuple)):
        return parsed
    for unit in units[:MAX_SUMMARY_UNITS]:
        if not isinstance(unit, Mapping):
            continue
        x = _finite(unit.get("x"), math.nan)
        y = _finite(unit.get("y"), math.nan)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        hp = max(0.0, _finite(unit.get("hp"), 0.0))
        max_hp = _finite(unit.get("max_hp"), 0.0)
        if max_hp <= 0.0:
            max_hp = hp if hp > 0.0 else 1.0
        parsed.append((x, y, min(hp, max_hp), max_hp))
    return parsed


def _side_stats(units: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    if not units:
        return {"count": 0, "hp_ratio": 0.0, "centroid": None}
    total_hp = sum(unit[2] for unit in units)
    total_max = sum(unit[3] for unit in units)
    ratio = math.nan if total_max <= 0.0 else max(0.0, min(1.0, total_hp / total_max))
    if not math.isfinite(ratio):
        ratio = 1.0
    cx = sum(unit[0] for unit in units) / len(units)
    cy = sum(unit[1] for unit in units) / len(units)
    return {
        "count": len(units),
        "hp_ratio": round(ratio, 3),
        "centroid": [round(cx, 1), round(cy, 1)],
    }


def _economy_stats(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compress the economy block the commander needs to plan with.

    Without this the model could only see unit positions, so it had no way to
    know whether the bank was full, how many extractors stood, or whether a
    build was already in progress -- and "plan the economy" from that information
    is guessing, not commanding. Every field is optional on input: a bridge that
    does not report an item simply yields ``None``, which the prompt renders as
    ``null`` (the model is told ``null`` means "unknown, keep the default").
    """

    raw = summary.get("economy")
    economy: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    def number(name: str) -> float | None:
        value = economy.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def count(name: str) -> int | None:
        value = number(name)
        return None if value is None else int(max(0.0, value))

    under_construction = economy.get("under_construction")
    return {
        "funds": number("funds"),
        "income": number("income"),
        "extractors": count("extractors"),
        "factories": count("factories"),
        "army": count("army"),
        "turrets": count("turrets"),
        "free_resource_slots": count("free_resource_slots"),
        "under_construction": (
            bool(under_construction) if under_construction is not None else None
        ),
    }


def compact_summary(observation_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compress a battlefield summary into the aggregate stats sent to the LLM."""

    summary: Mapping[str, Any] = (
        observation_summary if isinstance(observation_summary, Mapping) else {}
    )
    width, height = _map_dimensions(summary)
    return {
        "tick": _finite(summary.get("tick"), 0.0),
        "map": {"width": width, "height": height},
        "own": _side_stats(_parse_units(summary.get("own_units"))),
        "enemy": _side_stats(_parse_units(summary.get("enemy_units"))),
        "economy": _economy_stats(summary),
    }


def build_user_prompt(observation_summary: Mapping[str, Any] | None) -> str:
    """Build the user message; kept public so tests can assert the contract."""

    summary = compact_summary(observation_summary)
    return (
        "Battlefield summary (compressed JSON):\n"
        f"{json.dumps(summary, ensure_ascii=False, sort_keys=True)}\n\n"
        "Choose the next strategic directive, including the economy plan. One "
        "JSON object only, no unit-level orders, no markdown. Use null for any "
        "macro field you have no reason to change."
    )


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------
def _strip_code_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from model text (optionally fenced)."""

    cleaned = _strip_code_fence(str(text))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM output does not contain a JSON object") from None
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object")
    return payload


def parse_chat_completion(text: str) -> dict[str, Any]:
    """Pull the assistant message out of an OpenAI-compatible response body."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"chat completion response is not JSON ({exc.msg})") from None
    if not isinstance(data, Mapping):
        raise ValueError("chat completion response must be a JSON object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        error = data.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            raise ValueError(f"LLM API error: {_short(str(error.get('message')))}")
        raise ValueError("chat completion response has no choices")

    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat completion message has no text content")
    return extract_json_object(content)


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ----------------------------------------------------------------------
# Strategy implementation
# ----------------------------------------------------------------------
def _coerce_float(value: Any, name: str, *, minimum: float, allow_equal: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if allow_equal and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if not allow_equal and number <= minimum:
        raise ValueError(f"{name} must be > {minimum}")
    return number


class LLMStrategy:
    """OpenAI-compatible strategic policy with a :class:`RuleStrategy` fallback."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        fallback: RuleStrategy | None = None,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        environ: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ

        raw_key = api_key if api_key is not None else env.get(ENV_API_KEY)
        self.api_key = str(raw_key).strip() if raw_key else ""

        raw_base = base_url if base_url is not None else env.get(ENV_BASE_URL)
        self.base_url = str(raw_base).strip().rstrip("/") if raw_base else DEFAULT_BASE_URL

        raw_model = model if model is not None else env.get(ENV_MODEL)
        self.model = str(raw_model).strip() if raw_model else DEFAULT_MODEL

        self.fallback = fallback if fallback is not None else RuleStrategy()
        self.throttle_seconds = _coerce_float(throttle_seconds, "throttle_seconds", minimum=0.0)
        self.timeout = _coerce_float(timeout, "timeout", minimum=0.0, allow_equal=False)
        self.temperature = _coerce_float(temperature, "temperature", minimum=0.0)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens

        self._opener = opener
        self._last_directive: StrategicDirective | None = None
        self._last_call = float("-inf")

    # -- introspection ---------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when an API key is configured (LLM calls are possible)."""

        return bool(self.api_key)

    def __repr__(self) -> str:
        masked = "***" if self.api_key else None
        return (
            f"LLMStrategy(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key={masked!r}, throttle_seconds={self.throttle_seconds!r}, "
            f"timeout={self.timeout!r})"
        )

    # -- main entry point ------------------------------------------------
    def propose(self, observation_summary: Mapping[str, Any] | None) -> StrategicDirective:
        """Return a strategic directive, throttled and with rule fallback.

        Throttling: within ``throttle_seconds`` of the previous attempt the
        cached directive is returned without touching the network. Any failure
        returns a rule directive whose ``notes`` start with ``fallback:<reason>``.
        """

        now = time.monotonic()
        if (
            self._last_directive is not None
            and (now - self._last_call) < self.throttle_seconds
        ):
            return self._last_directive

        self._last_call = now
        if not self.enabled:
            directive = self._fallback_directive(observation_summary, "no_api_key")
        else:
            try:
                payload = self._request_payload(observation_summary)
                directive = StrategicDirective.from_payload(payload, source="llm")
            except Exception as exc:  # noqa: BLE001 - fall back instead of crashing
                directive = self._fallback_directive(
                    observation_summary, _describe_error(exc, self.api_key)
                )

        self._last_directive = directive
        return directive

    # -- HTTP plumbing ---------------------------------------------------
    def _endpoint_url(self) -> str:
        base = self.base_url.strip().rstrip("/")
        if not base.lower().startswith(("http://", "https://")):
            raise ValueError("LLM base URL must start with http:// or https://")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _build_request(self, observation_summary: Mapping[str, Any] | None) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation_summary)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self._endpoint_url(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": DIRECTIVE_USER_AGENT,
            },
            method="POST",
        )

    def _request_payload(self, observation_summary: Mapping[str, Any] | None) -> dict[str, Any]:
        request = self._build_request(observation_summary)
        opener = self._opener if self._opener is not None else urllib.request.urlopen
        response = opener(request, timeout=self.timeout)
        try:
            raw = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass

        if isinstance(raw, (bytes, bytearray)):
            text = bytes(raw).decode("utf-8", errors="replace")
        elif isinstance(raw, str):
            text = raw
        else:
            raise ValueError(f"unexpected response body type: {type(raw).__name__}")
        return parse_chat_completion(text)

    # -- fallback --------------------------------------------------------
    def _fallback_directive(
        self, observation_summary: Mapping[str, Any] | None, reason: str
    ) -> StrategicDirective:
        base = self.fallback.propose(observation_summary)
        note = f"{FALLBACK_NOTE_PREFIX}{_sanitize_reason(reason, self.api_key)}"
        if base.notes:
            note = f"{note}; {base.notes}"
        return replace(base, notes=note, source="rule", created_at=time.time())


def _describe_error(exc: BaseException, api_key: str) -> str:
    name = type(exc).__name__
    message = str(exc).strip()
    text = f"{name}: {message}" if message else name
    return _sanitize_reason(text, api_key)


def _sanitize_reason(reason: str, api_key: str) -> str:
    text = " ".join(str(reason).split())
    if api_key:
        text = text.replace(api_key, "***")
    text = _AUTH_HEADER_RE.sub(r"\1***", text)
    text = _BEARER_RE.sub("Bearer ***", text)
    text = _API_KEY_RE.sub("sk-***", text)
    if len(text) > MAX_FALLBACK_REASON_CHARS:
        text = text[: MAX_FALLBACK_REASON_CHARS - 3] + "..."
    return text or "unknown_error"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_strategy(
    use_llm: bool,
    *,
    retreat_hp_threshold: float = DEFAULT_RETREAT_HP_THRESHOLD,
    fallback: RuleStrategy | None = None,
    **llm_options: Any,
) -> RuleStrategy | LLMStrategy:
    """Return the best available strategy for the current configuration.

    ``use_llm=False`` always returns a :class:`RuleStrategy` so RL training keeps
    running with the LLM disabled (phase 5 acceptance criterion). When the LLM is
    requested but no API key is configured, the rule strategy is returned too.
    """

    rule = (
        fallback
        if fallback is not None
        else RuleStrategy(retreat_hp_threshold=retreat_hp_threshold)
    )
    if not use_llm:
        return rule
    llm = LLMStrategy(fallback=rule, **llm_options)
    return llm if llm.enabled else rule
