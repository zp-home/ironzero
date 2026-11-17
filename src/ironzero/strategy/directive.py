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

Two dimensions, and why the second one exists
---------------------------------------------
A directive used to carry *only* an attack intent (``push``/``defend``/...). That
made "the commander" a misnomer for the macro game: the economy (extractors,
factories, tier upgrades, turrets, reinforcement rate) and the decision *when* to
march were not expressible at all, so they were hard-coded in
``rwpp.env.RwppDuelEnv._curriculum_actions`` and no directive could change them.

:class:`MacroPlan` closes that gap. It is a *plan*, not an action list: it names
the economy goals in priority order and the budgets/limits the executor must
respect. It never names a unit, a coordinate, a screen point, a build queue entry
or a mouse button -- the executor (the environment's build stages) still decides
*which* pool, *where* and *when in the tick stream*.

Backward compatibility
----------------------
``macro`` is optional in the payload and defaults to an **empty** plan, whose
every field is ``None``. ``None`` means "the commander has no opinion", and the
executor then falls back to its own configured default. So an old directive (or
an LLM that ignores the macro block) behaves exactly as before, and a deployment
keeps its measured behaviour until a plan actually says otherwise.
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

#: Economy goals a macro plan may name, and simultaneously the **default**
#: execution order of the executor's build stages.
#:
#: The order is not decoration: it is the priority the executor already uses, and
#: it is load-bearing. Two placements are measured rather than stylistic:
#:
#: * ``upgrade`` precedes ``production`` because a tier-up compounds (a permanent
#:   rate increase) while a tank does not;
#: * ``expansion`` precedes ``production`` for the same reason, and that one was
#:   measured the hard way. With ``production`` first, income read exactly **26.0
#:   credits/s (one T1 extractor) for all 20056 ticks** of an episode whose
#:   ``economy_max_extractors`` was 6: both stages sit behind the same
#:   ``economy_credit_reserve`` gate, production drains every credit above it into
#:   one unit, the gate closes, and expansion never gets a turn (measured
#:   ``.runtime/stage_trace.json``: ``funds_below_reserve`` was the expansion
#:   blocker on 747/1081 decisions, or 69%). Swapping the two took income 30 -> 46
#:   and extractors 1 -> 3 on identical seeds (``.runtime/order_ab.json``). Raising
#:   the caps instead changed *nothing*, which is what proved the caps were never
#:   the binding constraint -- the stage that spends them never ran.
#:
#: ``attack`` stays last: an economy starved by an early march collapses (see
#: ``RwppGameConfig.opening_attack``).
MACRO_GOALS: tuple[str, ...] = (
    "upgrade",
    "extractor",
    "factory",
    "expansion",
    "production",
    "defence",
    "attack",
)

#: The order the executor runs when the plan does not name one.
DEFAULT_BUILD_ORDER: tuple[str, ...] = MACRO_GOALS

#: Goals that spend the single builder (or the single factory queue) and are
#: therefore mutually exclusive within one dispatch. The executor uses this to
#: keep two construction orders out of the same step.
BUILDER_GOALS: tuple[str, ...] = ("extractor", "factory", "production", "expansion")


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


def _optional_int(value: Any, name: str, *, minimum: int = 0) -> int | None:
    """Coerce to ``int | None``.

    Integral floats are accepted because JSON producers (and language models)
    routinely emit ``10.0`` where ``10`` was meant; refusing those would send a
    well-formed plan down the fallback path for a cosmetic reason. Anything that
    is not an exact integer is rejected -- silently truncating ``10.7`` units
    would be a lie about what the commander asked for.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer")
    if number < float(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return int(number)


def _optional_float(value: Any, name: str, *, minimum: float = 0.0) -> float | None:
    if value is None:
        return None
    number = _as_number(value, name)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


@dataclass(frozen=True, slots=True)
class MacroPlan:
    """Economy / tempo plan the executor must respect.

    Every field is optional and ``None`` means "no opinion, use the configured
    default". That is what keeps an old directive -- or an LLM that only fills in
    the attack intent -- behaving exactly as before.

    Two kinds of field, deliberately separated:

    * :attr:`build_order` decides **which** economy goals run and in **what
      priority**. It is the only field that can remove a goal entirely (e.g. omit
      ``"defence"`` for a map where turrets are a pure cost).
    * the numeric fields decide **how much**: how big the army should get, how
      many extractors/factories/turrets, the credit floor that protects the
      builder, and the army size and distance at which the march begins.

    What a plan may never contain: unit ids, entity ids, coordinates, screen
    positions, build-queue entries or mouse actions. Those stay with the executor
    and the tactical policy.
    """

    build_order: tuple[str, ...] = ()
    #: Combat units the economy should sustain.
    army_target: int | None = None
    #: Hard ceiling the production goal stops at.
    army_cap: int | None = None
    #: How many units may sit in a factory queue at once.
    #:
    #: A *tempo* knob rather than a budget: depth 1 means "order only when the
    #: queue is empty", which on a real factory (whose queue drains as units are
    #: delivered) is almost never true, so the pipeline runs dry and an army loses a
    #: reinforcement race it could have afforded. Deeper keeps it fed at the cost of
    #: tying credits up in units that are still being built.
    production_queue_depth: int | None = None
    max_extractors: int | None = None
    max_factories: int | None = None
    max_upgrades: int | None = None
    max_turrets: int | None = None
    #: Credits the executor must not spend below (protects the builder).
    credit_reserve: float | None = None
    #: Army size at which the attack goal starts marching.
    attack_at_army: int | None = None
    #: World px to the target at which the march stops being re-issued.
    engage_distance: float | None = None

    def __post_init__(self) -> None:
        raw_order = self.build_order
        if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, (list, tuple)):
            raise ValueError("build_order must be a sequence of goal names")
        normalised: list[str] = []
        for entry in raw_order:
            if not isinstance(entry, str):
                raise ValueError(f"build_order entries must be strings, got {type(entry).__name__}")
            goal = entry.strip().lower()
            if goal not in MACRO_GOALS:
                raise ValueError(f"build_order entries must be one of {MACRO_GOALS}, got {entry!r}")
            if goal in normalised:
                # A duplicate is not a stronger instruction, it is an ambiguous
                # one: the executor walks the list once, so the second copy could
                # only mean "run it twice", which would cancel its own build.
                raise ValueError(f"build_order must not repeat {goal!r}")
            normalised.append(goal)
        object.__setattr__(self, "build_order", tuple(normalised))

        for name, minimum in (
            ("army_target", 0),
            ("army_cap", 0),
            ("production_queue_depth", 0),
            ("max_extractors", 0),
            ("max_factories", 0),
            ("max_upgrades", 0),
            ("max_turrets", 0),
            ("attack_at_army", 0),
        ):
            object.__setattr__(
                self, name, _optional_int(getattr(self, name), name, minimum=minimum)
            )
        reserve = _optional_float(self.credit_reserve, "credit_reserve", minimum=0.0)
        object.__setattr__(self, "credit_reserve", reserve)
        # ``0`` is a meaningful value, not a missing one: it means "the army is
        # always considered engaged", so the executor never re-issues the march
        # order on the way in. Only a negative distance is nonsense.
        engage = _optional_float(self.engage_distance, "engage_distance", minimum=0.0)
        object.__setattr__(self, "engage_distance", engage)

        if (
            self.army_cap is not None
            and self.army_target is not None
            and self.army_cap < self.army_target
        ):
            # The cap is the point production stops at; a cap below the target
            # makes the target unreachable and is always a mistake, never a plan.
            raise ValueError("army_cap must be >= army_target")
        if (
            self.attack_at_army is not None
            and self.army_cap is not None
            and self.attack_at_army > self.army_cap
        ):
            raise ValueError("attack_at_army must be <= army_cap (it can never be reached)")

    # -- introspection ---------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """True when the plan expresses no opinion at all."""

        return not self.build_order and all(
            getattr(self, name) is None
            for name in (
                "army_target",
                "army_cap",
                "production_queue_depth",
                "max_extractors",
                "max_factories",
                "max_upgrades",
                "max_turrets",
                "credit_reserve",
                "attack_at_army",
                "engage_distance",
            )
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Goals in execution order (the default order when none is named)."""

        return self.build_order or DEFAULT_BUILD_ORDER

    # -- serialisation ---------------------------------------------------
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "MacroPlan":
        """Build a plan from an untrusted mapping; ``None``/``{}`` mean empty."""

        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("macro must be a mapping")
        raw_order = payload.get("build_order")
        if raw_order is None:
            order: tuple[str, ...] = ()
        elif isinstance(raw_order, (str, bytes)):
            raise ValueError("build_order must be a list of goal names, not a string")
        elif isinstance(raw_order, (list, tuple)):
            order = tuple(raw_order)
        else:
            raise ValueError("build_order must be a list of goal names")
        return cls(
            build_order=order,
            army_target=payload.get("army_target"),
            army_cap=payload.get("army_cap"),
            production_queue_depth=payload.get("production_queue_depth"),
            max_extractors=payload.get("max_extractors"),
            max_factories=payload.get("max_factories"),
            max_upgrades=payload.get("max_upgrades"),
            max_turrets=payload.get("max_turrets"),
            credit_reserve=payload.get("credit_reserve"),
            attack_at_army=payload.get("attack_at_army"),
            engage_distance=payload.get("engage_distance"),
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON-serialisable form (``live.jsonl``, replay, dashboard)."""

        return {
            "build_order": list(self.build_order),
            "army_target": self.army_target,
            "army_cap": self.army_cap,
            "production_queue_depth": self.production_queue_depth,
            "max_extractors": self.max_extractors,
            "max_factories": self.max_factories,
            "max_upgrades": self.max_upgrades,
            "max_turrets": self.max_turrets,
            "credit_reserve": self.credit_reserve,
            "attack_at_army": self.attack_at_army,
            "engage_distance": self.engage_distance,
        }

    def merged_over(self, fallback: "MacroPlan") -> "MacroPlan":
        """This plan with every ``None`` field taken from ``fallback``.

        The single place where "no opinion" is resolved, so the executor never
        has to repeat the fallback logic per field -- and so the resolved plan is
        itself loggable (that is what the trace records).
        """

        def pick(name: str) -> Any:
            value = getattr(self, name)
            return getattr(fallback, name) if value is None else value

        return MacroPlan(
            build_order=self.build_order or fallback.build_order,
            army_target=pick("army_target"),
            army_cap=pick("army_cap"),
            production_queue_depth=pick("production_queue_depth"),
            max_extractors=pick("max_extractors"),
            max_factories=pick("max_factories"),
            max_upgrades=pick("max_upgrades"),
            max_turrets=pick("max_turrets"),
            credit_reserve=pick("credit_reserve"),
            attack_at_army=pick("attack_at_army"),
            engage_distance=pick("engage_distance"),
        )


@dataclass(frozen=True, slots=True)
class StrategicDirective:
    """Validated high-level goal handed from the LLM/rule layer to PPO.

    Coordinates are map pixels (``x`` right, ``y`` down). ``attack_region`` is
    ``(x, y, radius)``; ``rally_point`` is ``(x, y)``. Both are ``None`` when the
    intent does not need them. ``source`` records whether the LLM produced the
    directive or the deterministic rule fallback did.

    ``macro`` carries the economy dimension (see :class:`MacroPlan`). It is an
    empty plan unless the commander actually states an opinion.
    """

    intent: str
    attack_region: tuple[float, float, float] | None = None
    rally_point: tuple[float, float] | None = None
    retreat_hp_threshold: float = DEFAULT_RETREAT_HP_THRESHOLD
    priority_target: str = DEFAULT_PRIORITY_TARGET
    notes: str = ""
    source: str = "rule"
    created_at: float = field(default_factory=time.time)
    macro: MacroPlan = field(default_factory=MacroPlan)

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
        if not isinstance(self.macro, MacroPlan):
            raise ValueError("macro must be a MacroPlan")

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
        An invalid ``macro`` block rejects the whole directive on purpose: a
        commander that asks for an unreachable plan (``army_cap`` below
        ``army_target``) is wrong in a way the executor must not guess about.
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

        macro = MacroPlan.from_payload(payload.get("macro"))

        return cls(
            intent=intent,
            attack_region=attack_region,
            rally_point=rally_point,
            retreat_hp_threshold=threshold,
            priority_target=raw_priority.strip(),
            notes=raw_notes,
            source=source,
            created_at=created_at,
            macro=macro,
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
            "macro": self.macro.to_payload(),
        }
