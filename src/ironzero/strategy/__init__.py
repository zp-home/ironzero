# SPDX-License-Identifier: Apache-2.0
"""LLM strategic layer for IronZero (architecture doc 12.4 / phase 5).

The layer is deliberately small and decoupled:

* :class:`~ironzero.strategy.directive.StrategicDirective` -- validated
  ``ironzero.strategy/1`` goal contract. The LLM emits only this; it never emits
  unit-level actions or screen coordinates.
* :class:`~ironzero.strategy.directive.MacroPlan` -- the economy/tempo half of a
  directive: which economy goals run, in what priority, and within what budgets.
* :class:`~ironzero.strategy.rule_strategy.RuleStrategy` -- deterministic,
  network-free fallback so RL training works with the LLM disabled. It emits both
  the attack intent and a *sparse* macro plan.
* :class:`~ironzero.strategy.llm_strategy.LLMStrategy` -- OpenAI-compatible
  Chat Completions client (``urllib.request``) with throttling and automatic
  rule fallback.
* :func:`~ironzero.strategy.llm_strategy.build_strategy` -- factory that picks
  the LLM only when requested *and* configured with an API key.

Typical adapter usage::

    from ironzero.strategy import build_strategy

    strategy = build_strategy(use_llm=True)
    directive = strategy.propose(observation_summary)
    event = directive.to_payload()  # append to live.jsonl
"""

from .directive import (
    BUILDER_GOALS,
    DEFAULT_BUILD_ORDER,
    DEFAULT_PRIORITY_TARGET,
    DEFAULT_RETREAT_HP_THRESHOLD,
    DIRECTIVE_SCHEMA_VERSION,
    INTENTS,
    MACRO_GOALS,
    SOURCES,
    MacroPlan,
    StrategicDirective,
)
from .llm_strategy import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_THROTTLE_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    FALLBACK_NOTE_PREFIX,
    SYSTEM_PROMPT,
    LLMStrategy,
    build_strategy,
    build_user_prompt,
    compact_summary,
    extract_json_object,
    parse_chat_completion,
)
from .rule_strategy import (
    DEFAULT_ATTACK_RADIUS,
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    RuleStrategy,
)
from .macro_controller import MacroController

__all__ = [
    "BUILDER_GOALS",
    "MacroController",
    "DEFAULT_ATTACK_RADIUS",
    "DEFAULT_BASE_URL",
    "DEFAULT_BUILD_ORDER",
    "DEFAULT_MAP_HEIGHT",
    "DEFAULT_MAP_WIDTH",
    "DEFAULT_MODEL",
    "DEFAULT_PRIORITY_TARGET",
    "DEFAULT_RETREAT_HP_THRESHOLD",
    "DEFAULT_THROTTLE_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DIRECTIVE_SCHEMA_VERSION",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "FALLBACK_NOTE_PREFIX",
    "INTENTS",
    "LLMStrategy",
    "MACRO_GOALS",
    "MacroPlan",
    "SOURCES",
    "SYSTEM_PROMPT",
    "RuleStrategy",
    "StrategicDirective",
    "build_strategy",
    "build_user_prompt",
    "compact_summary",
    "extract_json_object",
    "parse_chat_completion",
]
