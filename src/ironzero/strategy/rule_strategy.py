# SPDX-License-Identifier: Apache-2.0
"""Deterministic rule-based strategic policy (the network-free fallback).

The rule layer is the safe floor of the hybrid stack: with the LLM disabled the
environment still receives a valid :class:`StrategicDirective` on every call, so
PPO keeps training and the per-tick safety layer keeps acting. No I/O, no
network, no randomness -- identical summaries produce identical directives.

Two dimensions, and why the macro half exists
---------------------------------------------
The attack intent (``retreat`` / ``push`` / ``defend``) is the *tempo* decision.
It used to be the whole policy, which made "commander" a misnomer: the economy --
extractor count, factory count, tier upgrades, turrets, the reinforcement rate --
and the moment the march begins were not expressible, so they lived as hard-coded
constants in ``rwpp.env``.

:meth:`RuleStrategy.propose` now also returns a :class:`MacroPlan`. The plan is
**deliberately sparse**: a field is set only when a rule fires, and every unset
field means "no opinion, the executor keeps its configured default". That is what
keeps this layer honest in both directions:

* it cannot silently rewrite a configuration it was never told about (the rule
  layer does not import ``rwpp.config`` and must not guess its numbers);
* an episode in the neutral case produces an *empty* plan, so the measured
  behaviour of a deployment is unchanged unless a rule has an actual reason.

Each rule below states the observation that triggers it. None of them is a
tuning knob that was swept: they encode an ordering argument (a permanent rate
increase compounds, a tank does not) or a measured failure mode.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .directive import (
    DEFAULT_BUILD_ORDER,
    DEFAULT_RETREAT_HP_THRESHOLD,
    MacroPlan,
    StrategicDirective,
)

DEFAULT_MAP_WIDTH = 1000.0
DEFAULT_MAP_HEIGHT = 1000.0
DEFAULT_ATTACK_RADIUS = 400.0
DEFAULT_PRIORITY_TARGET = "nearest"
RETREAT_BACK_LINE_FRACTION = 0.15

#: Largest army the rule layer will ask for.
#:
#: The rule layer does not know the deployment's ``economy_army_cap``, and a plan
#: that asks for more units than the cap allows would be rejected by
#: :class:`MacroPlan`'s cross-field validation (cap >= target) -- which would
#: downgrade a legal directive to the fallback path for a cosmetic reason. 16 is
#: the documented ceiling of the sustained-production design
#: (``RwppGameConfig.economy_army_cap``), so the rule stays inside it.
RULE_MAX_ARMY = 16
#: Smallest army a "rebuild" order asks for; below this the game is usually lost
#: already and asking for a big army only spends the reserve.
RULE_MIN_ARMY = 4
#: Combat units below which the rule layer never orders a march.
#:
#: Marching with fewer units than this feeds them in one at a time -- the exact
#: measured failure of the pre-curriculum attack stage. This is a *floor*; the
#: resolved ``attack_at_army`` normally comes from the deployment's army target.
RULE_MIN_ATTACK_ARMY = 5

#: ``funds`` at which the economy is considered rich enough to invest instead of
#: hoarding. Deliberately expressed as a multiple of one upgrade (1200) rather
#: than as an absolute: the trigger is "I can afford a tier-up and still have the
#: bank left", which is the same argument the executor's upgrade gate makes.
RULE_RICH_FUNDS = 2400.0

# Parsed unit tuple: (x, y, hp, max_hp)
_Unit = tuple[float, float, float, float]


def _finite(value: Any, default: float) -> float:
    """Coerce to a finite float, returning ``default`` for anything unsafe."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _parse_side(units: Any) -> list[_Unit]:
    parsed: list[_Unit] = []
    if not isinstance(units, (list, tuple)):
        return parsed
    for unit in units:
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


def _hp_ratio(units: list[_Unit]) -> float:
    total_hp = sum(unit[2] for unit in units)
    total_max = sum(unit[3] for unit in units)
    if total_max <= 0.0:
        return 1.0
    return max(0.0, min(1.0, total_hp / total_max))


def _centroid(units: list[_Unit]) -> tuple[float, float]:
    count = float(len(units))
    return (
        sum(unit[0] for unit in units) / count,
        sum(unit[1] for unit in units) / count,
    )


def _clamp_point(point: tuple[float, float], width: float, height: float) -> tuple[float, float]:
    return (
        min(max(point[0], 0.0), width),
        min(max(point[1], 0.0), height),
    )


def _retreat_point(units: list[_Unit], width: float, height: float) -> tuple[float, float]:
    """Rally deep in the map half that currently contains the own centroid."""

    cx, cy = _centroid(units)
    if cy < height / 2.0:
        back_y = height * RETREAT_BACK_LINE_FRACTION
    else:
        back_y = height * (1.0 - RETREAT_BACK_LINE_FRACTION)
    return _clamp_point((cx, back_y), width, height)


def _attack_region(
    enemy_units: list[_Unit], width: float, height: float, radius: float
) -> tuple[float, float, float]:
    if enemy_units:
        center = _centroid(enemy_units)
    else:
        # No visible enemy: push towards the middle of the map.
        center = (width / 2.0, height / 2.0)
    x, y = _clamp_point(center, width, height)
    return x, y, radius


def _map_dimensions(summary: Mapping[str, Any]) -> tuple[float, float]:
    raw_map = summary.get("map")
    if isinstance(raw_map, Mapping):
        width = _finite(raw_map.get("width"), DEFAULT_MAP_WIDTH)
        height = _finite(raw_map.get("height"), DEFAULT_MAP_HEIGHT)
    else:
        width, height = DEFAULT_MAP_WIDTH, DEFAULT_MAP_HEIGHT
    if width <= 0.0:
        width = DEFAULT_MAP_WIDTH
    if height <= 0.0:
        height = DEFAULT_MAP_HEIGHT
    return width, height


def _order_without(*dropped: str) -> tuple[str, ...]:
    """Default execution order minus ``dropped`` goals (order preserved)."""

    return tuple(goal for goal in DEFAULT_BUILD_ORDER if goal not in dropped)


def _order_with_first(*promoted: str) -> tuple[str, ...]:
    """Default execution order with ``promoted`` moved to the front."""

    head = tuple(goal for goal in promoted if goal in DEFAULT_BUILD_ORDER)
    tail = tuple(goal for goal in DEFAULT_BUILD_ORDER if goal not in head)
    return head + tail


@dataclass(frozen=True, slots=True)
class RuleStrategy:
    """Pure rule policy mapping a compressed battlefield summary to a directive.

    The decision order follows the architecture document:

    1. own total HP ratio below ``retreat_hp_threshold`` -> ``retreat`` towards
       the own half;
    2. own unit count >= enemy unit count -> ``push`` to the enemy centroid;
    3. otherwise -> ``defend`` at the own centroid.

    The macro plan is decided independently (see :meth:`_macro_plan`) because the
    two questions are genuinely separate: "can I win a fight right now" and "what
    should the economy look like in thirty seconds".
    """

    retreat_hp_threshold: float = DEFAULT_RETREAT_HP_THRESHOLD
    attack_radius: float = DEFAULT_ATTACK_RADIUS
    priority_target: str = DEFAULT_PRIORITY_TARGET
    #: Emit a macro plan at all. ``False`` reproduces the pre-macro layer exactly
    #: (an always-empty plan), which is what an A/B against the old behaviour
    #: needs.
    macro_enabled: bool = True

    def __post_init__(self) -> None:
        threshold = _finite(self.retreat_hp_threshold, math.nan)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("retreat_hp_threshold must be within [0, 1]")
        radius = _finite(self.attack_radius, math.nan)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("attack_radius must be a positive number")
        if not isinstance(self.priority_target, str) or not self.priority_target.strip():
            raise ValueError("priority_target must be a non-empty string")
        object.__setattr__(self, "retreat_hp_threshold", threshold)
        object.__setattr__(self, "attack_radius", radius)
        object.__setattr__(self, "priority_target", self.priority_target.strip())

    def propose(self, observation_summary: Mapping[str, Any] | None) -> StrategicDirective:
        """Return a valid directive for ``observation_summary`` (never raises)."""

        summary: Mapping[str, Any] = (
            observation_summary if isinstance(observation_summary, Mapping) else {}
        )
        width, height = _map_dimensions(summary)
        own_units = _parse_side(summary.get("own_units"))
        enemy_units = _parse_side(summary.get("enemy_units"))
        tick = _finite(summary.get("tick"), 0.0)
        macro = self._macro_plan(summary, own_units, enemy_units)

        if not own_units and not enemy_units:
            return self._directive(
                "defend", notes="rule:empty observation", tick=tick, macro=macro
            )
        if not own_units:
            return self._directive(
                "defend",
                notes=f"rule:no own units; enemy={len(enemy_units)}",
                tick=tick,
                macro=macro,
            )

        hp_ratio = _hp_ratio(own_units)
        if hp_ratio < self.retreat_hp_threshold:
            return self._directive(
                "retreat",
                rally_point=_retreat_point(own_units, width, height),
                notes=(
                    f"rule:hp_ratio={hp_ratio:.2f} < {self.retreat_hp_threshold:.2f};"
                    " retreat to own half"
                ),
                tick=tick,
                macro=macro,
            )

        if len(own_units) >= len(enemy_units):
            region = _attack_region(enemy_units, width, height, self.attack_radius)
            return self._directive(
                "push",
                attack_region=region,
                notes=(
                    f"rule:units own={len(own_units)} >= enemy={len(enemy_units)};"
                    " push to enemy centroid"
                ),
                tick=tick,
                macro=macro,
            )

        rally = _clamp_point(_centroid(own_units), width, height)
        return self._directive(
            "defend",
            rally_point=rally,
            notes=(
                f"rule:units own={len(own_units)} < enemy={len(enemy_units)};"
                " defend own centroid"
            ),
            tick=tick,
            macro=macro,
        )

    # ------------------------------------------------------------------
    # Macro plan
    # ------------------------------------------------------------------
    def _macro_plan(
        self,
        summary: Mapping[str, Any],
        own: list[_Unit],
        enemy: list[_Unit],
    ) -> MacroPlan:
        """Decide the economy/tempo plan for this observation.

        The rules are evaluated in the order below and **the first one that fires
        wins**. They are ordered by urgency, not by topic: a base about to fall is
        more urgent than a full bank, and an army that cannot win a fight makes
        every investment decision secondary.

        Every branch returns a *sparse* plan (only the fields it has an opinion
        about); see the module docstring for why that matters.
        """

        if not self.macro_enabled:
            return MacroPlan()

        economy = summary.get("economy")
        economy = economy if isinstance(economy, Mapping) else {}
        funds = max(0.0, _finite(economy.get("funds"), 0.0))
        army = int(max(0.0, _finite(economy.get("army"), 0.0)))
        extractors = int(max(0.0, _finite(economy.get("extractors"), 0.0)))
        factories = int(max(0.0, _finite(economy.get("factories"), 0.0)))
        free_slots = int(max(0.0, _finite(economy.get("free_resource_slots"), 0.0)))
        under_construction = bool(economy.get("under_construction"))
        own_count = len(own)
        enemy_count = len(enemy)
        hp_ratio = _hp_ratio(own) if own else 1.0

        # -- Rule 1: the army is broken. Rebuild, do not invest. -------------
        #
        # A retreat is the one state where "more income" is the wrong answer: the
        # buildings that produce it are what is being lost. The order therefore
        # drops ``expansion`` (a new extractor is a static building out in the
        # open) and drops ``attack`` (there is nothing to march with), promotes
        # production and defence, and raises the credit reserve so the builder can
        # always finish whatever is half-built instead of dying mid-structure.
        if own and hp_ratio < self.retreat_hp_threshold:
            target = min(RULE_MAX_ARMY, max(RULE_MIN_ARMY, enemy_count + 2))
            return MacroPlan(
                # ``attack``/``expansion`` must be *removed*, not merely
                # de-prioritised: a goal still in the list re-issues its order
                # every decision, and an extractor is a static building placed in
                # the open exactly when nothing can defend it.
                build_order=tuple(
                    goal
                    for goal in _order_with_first("production", "defence")
                    if goal not in ("attack", "expansion")
                ),
                army_target=target,
                army_cap=min(RULE_MAX_ARMY, target + 4),
            )

        # -- Rule 2: outnumbered. Survive first, spend on units. -------------
        #
        # The ordering argument: an economy is only worth what survives the next
        # two minutes. Production and defence move ahead of the tier-up ladder,
        # which is a multi-hundred-tick investment that pays back too late here.
        if own and own_count < enemy_count:
            target = min(RULE_MAX_ARMY, max(army + 2, enemy_count + 2))
            return MacroPlan(
                build_order=tuple(
                    goal
                    for goal in _order_with_first(
                        "production", "defence", "upgrade", "extractor"
                    )
                    if goal != "attack"
                ),
                army_target=target,
                attack_at_army=None,
            )

        # -- Rule 3: rich with an army already at target. Invest the bank. ---
        #
        # Both conditions are required, because either alone is a bad signal:
        #
        # * a big bank with no army means the credits are the army's, not the
        #   economy's (measured: an ungated expansion stage ended an episode with
        #   five extractors, 19748 unspent credits and *zero* combat units);
        # * a free pool with no bank just re-issues an unaffordable order.
        #
        # The plan raises the *extractor cap*; it does **not** move ``expansion``
        # to the front. That distinction was measured in this repository and it is
        # the whole subtlety of this rule: with expansion allowed to grab the
        # builder ahead of the factory chain, the factory stage never ran --
        # ``no_combat_factory`` was the blocking reason on 1968 of 1968 decisions,
        # the episode ended with five extractors and zero combat units. A rich
        # bank is a licence to build *more* extractors, not a licence to build them
        # *first*.
        if (
            funds >= RULE_RICH_FUNDS
            and free_slots >= 1
            and not under_construction
            and army >= RULE_MIN_ATTACK_ARMY
        ):
            return MacroPlan(
                max_extractors=min(9, max(6, extractors + free_slots)),
                max_upgrades=min(12, 6 + free_slots),
            )

        # -- Rule 4: locally superior. March earlier than the default gate. --
        #
        # ``attack_at_army`` is what the attack stage compares the combat-unit
        # count against, so lowering it to the *current* army (never below the
        # floor) turns "wait until the configured target is reached" into "we
        # already outnumber them, go". Raising it is never done here: waiting
        # longer than the deployment chose needs a reason this layer does not have.
        if (
            enemy_count > 0
            and own_count > enemy_count
            and army >= max(RULE_MIN_ATTACK_ARMY, enemy_count + 1)
        ):
            return MacroPlan(attack_at_army=max(RULE_MIN_ATTACK_ARMY, enemy_count + 1))

        # -- Rule 5: neutral. No opinion at all. ----------------------------
        #
        # An empty plan is not an omission: it is the statement that the rule layer
        # has nothing better than the deployment's own configuration, and it keeps
        # the measured behaviour of a default run bit-identical.
        return MacroPlan()

    def _directive(
        self,
        intent: str,
        *,
        attack_region: tuple[float, float, float] | None = None,
        rally_point: tuple[float, float] | None = None,
        notes: str,
        tick: float,
        macro: MacroPlan | None = None,
    ) -> StrategicDirective:
        return StrategicDirective(
            intent=intent,
            attack_region=attack_region,
            rally_point=rally_point,
            retreat_hp_threshold=self.retreat_hp_threshold,
            priority_target=self.priority_target,
            notes=f"{notes} tick={tick:g}",
            source="rule",
            created_at=time.time(),
            macro=macro if macro is not None else MacroPlan(),
        )
