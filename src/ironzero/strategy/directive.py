# SPDX-License-Identifier: Apache-2.0
"""Strategic directive contract for the IronZero LLM strategy layer.

The architecture document (section 12.4 / phase 5) splits control into three
time scales:

* LLM commander (1-5 s): emits a :class:`StrategicDirective` only -- never
  unit-level actions, mouse positions or screen coordinates.
* PPO tactical policy (2-10 Hz): turns the directive into unit-group orders.
* Rule safety layer (every tick): filters illegal actions and forces retreats.

This module owns the ``ironzero.strategy/1`` payload schema so the LLM layer,
the rule fallback and the PPO adapter all speak the same validated format.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

DIRECTIVE_SCHEMA_VERSION = "ironzero.strategy/1"

#: Allowed high-level intents. The LLM may only pick one of these; anything
#: else is rejected at the validation boundary and degrades to the rule layer.
INTENTS: tuple[str, ...] = ("push", "defend", "retreat", "harass", "regroup")

#: Origin of a directive. ``rule`` means the deterministic fallback produced it.
SOURCES: tuple[str, ...] = ("llm", "rule")

DEFAULT_RETREAT_HP_THRESHOLD = 0.35
DEFAULT_PRIORITY_TARGET = "nearest"


def _as_number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a [x, y] sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 numbers")
    return _as_number(value[0], f"{name}[0]"), _as_number(value[1], f"{name}[1]")


def _as_region(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a [x, y, radius] sequence")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    x = _as_number(value[0], f"{name}[0]")
    y = _as_number(value[1], f"{name}[1]")
    radius = _as_number(value[2], f"{name}[2]")
    if radius <= 0.0:
        raise ValueError(f"{name} radius must be positive")
    return x, y, radius


@dataclass(frozen=True, slots=True)
class StrategicDirective:
    """Validated high-level goal handed from the LLM/rule layer to PPO.

    Coordinates are map pixels (``x`` right, ``y`` down). ``attack_region`` is
    ``(x, y, radius)``; ``rally_point`` is ``(x, y)``. Both are ``None`` when the
    intent does not need them. ``source`` records whether the LLM produced the
    directive or the deterministic rule fallback did.
    """

    intent: str
    attack_region: tuple[float, float, float] | None = None
    rally_point: tuple[float, float] | None = None
    retreat_hp_threshold: float = DEFAULT_RETREAT_HP_THRESHOLD
    priority_target: str = DEFAULT_PRIORITY_TARGET
    notes: str = ""
    source: str = "rule"
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Normalise containers to immutable tuples and validate every field so
        # a directive can never leave this module in an illegal state.
        if self.attack_region is not None:
            object.__setattr__(self, "attack_region", _as_region(self.attack_region, "attack_region"))
        if self.rally_point is not None:
            object.__setattr__(self, "rally_point", _as_pair(self.rally_point, "rally_point"))

        if not isinstance(self.intent, str) or self.intent not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}, got {self.intent!r}")
        if not isinstance(self.source, str) or self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if not isinstance(self.priority_target, str) or not self.priority_target.strip():
            raise ValueError("priority_target must be a non-empty string")
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")

        threshold = _as_number(self.retreat_hp_threshold, "retreat_hp_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("retreat_hp_threshold must be within [0, 1]")
        object.__setattr__(self, "retreat_hp_threshold", threshold)

        created_at = _as_number(self.created_at, "created_at")
        if created_at < 0.0:
            raise ValueError("created_at must be an epoch timestamp >= 0")
        object.__setattr__(self, "created_at", created_at)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, source: str) -> StrategicDirective:
        """Build a directive from an untrusted payload (LLM JSON or JSONL log).

        Raises ``ValueError`` for anything that does not satisfy the contract;
        callers are expected to fall back to the rule strategy in that case.
        """

        if not isinstance(payload, Mapping):
            raise ValueError("directive payload must be a mapping")
        if not isinstance(source, str) or source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")

        version = payload.get("schema_version")
        if version is not None and version != DIRECTIVE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version!r}")

        raw_intent = payload.get("intent")
        if not isinstance(raw_intent, str) or raw_intent.strip().lower() not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}, got {raw_intent!r}")
        intent = raw_intent.strip().lower()

        raw_region = payload.get("attack_region")
        attack_region = None if raw_region is None else _as_region(raw_region, "attack_region")

        raw_rally = payload.get("rally_point")
        rally_point = None if raw_rally is None else _as_pair(raw_rally, "rally_point")

        threshold = _as_number(
            payload.get("retreat_hp_threshold", DEFAULT_RETREAT_HP_THRESHOLD),
            "retreat_hp_threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("retreat_hp_threshold must be within [0, 1]")

        raw_priority = payload.get("priority_target", DEFAULT_PRIORITY_TARGET)
        if raw_priority is None:
            raw_priority = DEFAULT_PRIORITY_TARGET
        if not isinstance(raw_priority, str) or not raw_priority.strip():
            raise ValueError("priority_target must be a non-empty string")

        raw_notes = payload.get("notes", "")
        if raw_notes is None:
            raw_notes = ""
        if not isinstance(raw_notes, str):
            raise ValueError("notes must be a string")

        raw_created_at = payload.get("created_at")
        created_at = time.time() if raw_created_at is None else _as_number(raw_created_at, "created_at")

        return cls(
            intent=intent,
            attack_region=attack_region,
            rally_point=rally_point,
            retreat_hp_threshold=threshold,
            priority_target=raw_priority.strip(),
            notes=raw_notes,
            source=source,
            created_at=created_at,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload (for ``live.jsonl`` and replay)."""

        return {
            "schema_version": DIRECTIVE_SCHEMA_VERSION,
            "intent": self.intent,
            "attack_region": list(self.attack_region) if self.attack_region is not None else None,
            "rally_point": list(self.rally_point) if self.rally_point is not None else None,
            "retreat_hp_threshold": self.retreat_hp_threshold,
            "priority_target": self.priority_target,
            "notes": self.notes,
            "source": self.source,
            "created_at": self.created_at,
        }
