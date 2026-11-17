# SPDX-License-Identifier: Apache-2.0
"""Executable form of a commander's :class:`~ironzero.strategy.directive.MacroPlan`.

The strategy layer states *intent*; this module turns it into the concrete
numbers and the concrete stage order that :class:`~ironzero.rwpp.env.RwppDuelEnv`
executes. The split exists so the two questions stay separable:

* "what does the commander want" -- :class:`MacroPlan`, produced by the LLM or
  the rule layer, validated, and logged verbatim in ``live.jsonl``;
* "what may this deployment actually do" -- :class:`ResolvedMacro`, produced
  here, clamped against the configured defaults.

Why clamping is not optional
----------------------------
A plan arrives from an untrusted source (a language model, a JSONL replay, a
hand-written test). Every field can therefore be wrong, missing, or adversarial.
The executor must keep running, so the resolution is total: an unknown goal name
is dropped, an absurd count is clamped to a documented range, and a missing field
inherits the deployment's configured default. Nothing here raises on a legal plan
and nothing here can produce a value the stages cannot use.

The clamping bounds are deliberately wide. They exist to stop a plan from
*breaking* the game loop (a 10000-unit army target would pin production for the
whole episode; a 50-credit reserve would let production starve the builder), not
to express a preference about good play. Good play is the commander's job.

Fallback semantics
------------------
``stage_names`` is the priority order the stages run in. An **empty**
``build_order`` means "keep the platform default", which is what makes the rule
layer's neutral case (an empty plan) behave exactly like the pre-macro code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..strategy.directive import (
    BUILDER_GOALS,
    DEFAULT_BUILD_ORDER,
    MACRO_GOALS,
    MacroPlan,
)

#: Hard bounds for every numeric field, keyed by the resolved attribute name.
#:
#: Lower bounds are zero for counts and the *engine's own price list* for the
#: credit reserve, so a plan cannot order the builder to work without being able
#: to finish anything. Upper bounds are set where a value stops being a strategy
#: and starts being a way to freeze the episode.
LIMITS: dict[str, tuple[float, float]] = {
    "army_target": (1.0, 24.0),
    "army_cap": (1.0, 32.0),
    # How many units may sit in the factory queue at once.
    #
    # A queue depth of 1 means "only order when the queue is empty", which is what
    # silently disabled production: measured at difficulty 1, the curriculum ordered
    # a unit on 9 of 2100 decisions while the factories stood intact and the army
    # fell from 11 to 5. A shallow queue keeps the pipeline fed without draining the
    # bank (``.runtime/army_decline.json``).
    "production_queue_depth": (1.0, 8.0),
    "max_extractors": (0.0, 9.0),
    "max_factories": (0.0, 4.0),
    "max_upgrades": (0.0, 12.0),
    "max_turrets": (0.0, 6.0),
    "credit_reserve": (0.0, 5000.0),
    "attack_at_army": (1.0, 24.0),
    "engage_distance": (0.0, 4000.0),
}


def _clamp(name: str, value: float) -> float:
    low, high = LIMITS[name]
    if not math.isfinite(value):
        return low
    return min(max(float(value), low), high)


@dataclass(frozen=True, slots=True)
class ResolvedMacro:
    """Concrete, clamped budgets and stage order for one decision.

    Attribute names match the ``RwppGameConfig.economy_*`` fields they replace,
    so a stage reads ``self._macro.army_target`` where it used to read
    ``self.game.economy_army_target`` and the two are interchangeable by design.
    """

    stage_names: tuple[str, ...]
    army_target: int
    army_cap: int
    #: Units allowed to sit in the factory queue at once. See :data:`LIMITS`.
    production_queue_depth: int
    max_extractors: int
    max_factories: int
    max_upgrades: int
    max_turrets: int
    credit_reserve: float
    #: Army size at which the attack stage starts marching. Defaults to
    #: ``army_target``: the same threshold the production stage sustains, which is
    #: what makes "the army is big enough" one number instead of two.
    attack_at_army: int
    engage_distance: float
    #: True when the plan actually changed something (for logging / A-B).
    from_commander: bool = False

    # -- queries used by the stage loop ----------------------------------
    @property
    def builder_goals(self) -> tuple[str, ...]:
        """Goals in this plan that compete for the single builder."""

        return tuple(name for name in self.stage_names if name in BUILDER_GOALS)

    def allows(self, goal: str) -> bool:
        return goal in self.stage_names

    @classmethod
    def from_config(cls, config: Any) -> "ResolvedMacro":
        """The deployment's own defaults, as a resolved plan.

        This is the identity resolution: running with no directive at all (or an
        empty one) must reproduce the configured behaviour exactly.
        """

        army_target = int(getattr(config, "economy_army_target", 10))
        return cls(
            stage_names=DEFAULT_BUILD_ORDER,
            army_target=army_target,
            army_cap=int(getattr(config, "economy_army_cap", 16)),
            production_queue_depth=int(
                getattr(config, "economy_production_queue_depth", 2)
            ),
            max_extractors=int(getattr(config, "economy_max_extractors", 6)),
            max_factories=int(getattr(config, "economy_max_factories", 2)),
            max_upgrades=int(getattr(config, "economy_max_upgrades", 6)),
            max_turrets=int(getattr(config, "economy_max_turrets", 2)),
            credit_reserve=float(getattr(config, "economy_credit_reserve", 900.0)),
            attack_at_army=army_target,
            engage_distance=float(getattr(config, "economy_engage_distance", 700.0)),
        )

    def merged_with(self, plan: MacroPlan | None) -> "ResolvedMacro":
        """This baseline with ``plan``'s stated opinions applied.

        ``None`` and an empty plan both return ``self`` unchanged, which is the
        hot path: the rule layer's neutral case and every LLM response that only
        fills in the attack intent take it.
        """

        if plan is None or plan.is_empty:
            return self
        numbers = {
            name: _clamp(name, float(getattr(plan, name)))
            for name in LIMITS
            if getattr(plan, name) is not None
        }
        # Cross-field repair, in the order the semantics require.
        #
        # ``MacroPlan`` already rejects a cap below the target and an attack
        # threshold above the cap, but clamping can *create* those contradictions
        # (e.g. a plan asks for army_target=24 and army_cap=24; the target clamps
        # to 24 and so does the cap -- fine -- but a plan asking for
        # army_target=30/army_cap=31 clamps both to 24/31). Repairing here rather
        # than raising keeps a hostile plan from silently disabling the commander.
        target = int(numbers.get("army_target", self.army_target))
        cap = int(numbers.get("army_cap", self.army_cap))
        if cap < target:
            cap = target
        attack_at = int(numbers.get("attack_at_army", self.attack_at_army))
        if attack_at > cap:
            attack_at = cap
        stage_names = plan.stage_names
        if not stage_names:
            stage_names = self.stage_names
        return ResolvedMacro(
            stage_names=tuple(stage_names),
            army_target=target,
            army_cap=cap,
            production_queue_depth=int(
                numbers.get("production_queue_depth", self.production_queue_depth)
            ),
            max_extractors=int(numbers.get("max_extractors", self.max_extractors)),
            max_factories=int(numbers.get("max_factories", self.max_factories)),
            max_upgrades=int(numbers.get("max_upgrades", self.max_upgrades)),
            max_turrets=int(numbers.get("max_turrets", self.max_turrets)),
            credit_reserve=float(numbers.get("credit_reserve", self.credit_reserve)),
            attack_at_army=attack_at,
            engage_distance=float(numbers.get("engage_distance", self.engage_distance)),
            from_commander=True,
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe view for the ``strategy`` live event and for traces."""

        return {
            "stage_names": list(self.stage_names),
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
            "from_commander": self.from_commander,
        }


def resolve_macro(config: Any, directive: Any | None) -> ResolvedMacro:
    """Resolve ``directive``'s macro plan against ``config``'s defaults.

    Total by construction: a missing directive, a directive without ``macro``
    (older object), or a plan that fails to merge all yield the configured
    default rather than an exception. The executor may then assume it always has
    a usable plan.
    """

    baseline = ResolvedMacro.from_config(config)
    if directive is None:
        return baseline
    plan = getattr(directive, "macro", None)
    if not isinstance(plan, MacroPlan):
        return baseline
    try:
        return baseline.merged_with(plan)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return baseline


def economy_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """The economy facts a commander needs, read from a bridge state.

    Kept next to the resolver (rather than in ``rwpp.strategy``) because the two
    halves of the macro loop -- "what did the commander see" and "what did it ask
    for" -- must be described in one place or the trace cannot be read.

    Every value is derived from the observation stream the executor itself uses,
    so the commander plans against the same numbers the stages act on. A missing
    field yields ``None``/``0`` rather than a guess.
    """

    own = [unit for unit in (state.get("own_entities") or []) if isinstance(unit, dict)]
    global_state = state.get("global_state") or {}
    if not isinstance(global_state, Mapping):
        global_state = {}

    def _float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    slots = [slot for slot in (state.get("resource_slots") or []) if isinstance(slot, dict)]
    extractors = [
        unit
        for unit in own
        if unit.get("is_building") and "xtractor" in str(unit.get("unit_type") or "").lower()
    ]
    factories = [
        unit
        for unit in own
        if unit.get("is_building")
        and (unit.get("is_factory") or unit.get("is_factory_building"))
        and "commandcenter" not in str(unit.get("unit_type") or "").lower()
    ]
    turrets = [
        unit
        for unit in own
        if unit.get("is_building")
        and unit.get("is_defence")
        and not unit.get("is_factory")
        and not unit.get("is_factory_building")
    ]
    army = [
        unit
        for unit in own
        if unit.get("can_attack")
        and not unit.get("is_builder")
        and not unit.get("is_building")
        and float(unit.get("hp") or 0.0) > 0.0
    ]
    return {
        "funds": _float(global_state.get("funds")),
        "income": _float(global_state.get("income")),
        "extractors": sum(1 for unit in extractors if _is_finished(unit)),
        "factories": sum(1 for unit in factories if _is_finished(unit)),
        "turrets": sum(1 for unit in turrets if _is_finished(unit)),
        "army": len(army),
        "free_resource_slots": sum(1 for slot in slots if not slot.get("occupied")),
        "under_construction": any(
            unit.get("is_building") and not _is_finished(unit) for unit in own
        ),
    }


def _is_finished(unit: Mapping[str, Any]) -> bool:
    """Whether a building is operational (mirrors ``env._is_operational``).

    Duplicated as a tiny local helper instead of importing from ``env`` so this
    module stays free of an import cycle (``env`` imports this one). The rule is
    the same and deliberately lenient in the same direction: a building that was
    ever operational counts, because a mid-upgrade building reports
    ``is_operational: false`` for the duration of the upgrade and treating that
    as "never built" makes the commander re-plan an economy it already has.
    """

    if "is_operational" in unit:
        return bool(unit.get("is_operational")) or float(
            unit.get("build_progress", 0.0) or 0.0
        ) >= 1.0
    return float(unit.get("build_progress", 1.0) or 0.0) >= 1.0


__all__ = [
    "LIMITS",
    "MACRO_GOALS",
    "ResolvedMacro",
    "economy_summary",
    "resolve_macro",
]
