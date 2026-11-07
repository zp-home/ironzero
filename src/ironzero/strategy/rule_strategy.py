# SPDX-License-Identifier: Apache-2.0
"""Deterministic rule-based strategic policy (the network-free fallback).

The rule layer is the safe floor of the hybrid stack: with the LLM disabled the
environment still receives a valid :class:`StrategicDirective` on every call, so
PPO keeps training and the per-tick safety layer keeps acting. No I/O, no
network, no randomness -- identical summaries produce identical directives.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .directive import DEFAULT_RETREAT_HP_THRESHOLD, StrategicDirective

DEFAULT_MAP_WIDTH = 1000.0
DEFAULT_MAP_HEIGHT = 1000.0
DEFAULT_ATTACK_RADIUS = 400.0
DEFAULT_PRIORITY_TARGET = "nearest"
RETREAT_BACK_LINE_FRACTION = 0.15

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


@dataclass(frozen=True, slots=True)
class RuleStrategy:
    """Pure rule policy mapping a compressed battlefield summary to a directive.

    The decision order follows the architecture document:

    1. own total HP ratio below ``retreat_hp_threshold`` -> ``retreat`` towards
       the own half;
    2. own unit count >= enemy unit count -> ``push`` to the enemy centroid;
    3. otherwise -> ``defend`` at the own centroid.
    """

    retreat_hp_threshold: float = DEFAULT_RETREAT_HP_THRESHOLD
    attack_radius: float = DEFAULT_ATTACK_RADIUS
    priority_target: str = DEFAULT_PRIORITY_TARGET

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

        if not own_units and not enemy_units:
            return self._directive("defend", notes="rule:empty observation", tick=tick)
        if not own_units:
            return self._directive(
                "defend", notes=f"rule:no own units; enemy={len(enemy_units)}", tick=tick
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
        )

    def _directive(
        self,
        intent: str,
        *,
        attack_region: tuple[float, float, float] | None = None,
        rally_point: tuple[float, float] | None = None,
        notes: str,
        tick: float,
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
        )
