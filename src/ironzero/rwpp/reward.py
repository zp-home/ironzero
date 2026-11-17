# SPDX-License-Identifier: Apache-2.0
"""Hp / distance potential terms used by the real-game reward shaping.

These are pure functions over a bridge state dict. They are the *input* to the
reward decomposition in :mod:`ironzero.rwpp.env`; the accumulation order of the
final reward lives there and must not change (floating-point sums are order
sensitive and the reward assertions in the tests depend on it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig


def hp_potential(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> float:
    """Normalised hp advantage in ``[-1, 1]`` (own minus enemy).

    Dead units disappear from the bridge stream, so the reference totals from
    the first observation of the episode are used when available; otherwise the
    alive-unit maximum is the denominator.
    """

    own, enemy = hp_fractions(state, own_max_ref, enemy_max_ref)
    return own - enemy


def hp_fractions(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> tuple[float, float]:
    """Return ``(own_hp_frac, enemy_hp_frac)`` from a raw state."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    own_hp = sum(float(unit.get("hp") or 0.0) for unit in own)
    enemy_hp = sum(float(unit.get("hp") or 0.0) for unit in enemies)
    own_max = sum(float(unit.get("max_hp") or 0.0) for unit in own) or 1.0
    enemy_max = sum(float(unit.get("max_hp") or 0.0) for unit in enemies) or 1.0
    own_ref = float(own_max_ref) if own_max_ref else own_max
    enemy_ref = float(enemy_max_ref) if enemy_max_ref else enemy_max
    # The AI reinforces far past its opening army, and on an original economy map it
    # keeps its command centre alive the whole game. With the opening total as the
    # denominator the ratio therefore *exceeds* 1.0 almost immediately and a naive
    # clamp pins it there.
    #
    # Measured before this fix, over every ``eval_report.json`` on stock skirmish
    # maps: ``enemy_hp_frac`` was exactly 1.0 in **89 of 89 episodes**, and
    # ``own_hp_frac`` was 1.0 in 65 of 89 (73%). The dashboard KPI read "100% vs
    # 100%" while the agent was actually down to 5 units against 15-17. Training
    # and inference share this code, so the two "agreed" and the number looked
    # trustworthy.
    #
    # The clamp stays (a fraction above 1.0 is still not meaningful to a caller
    # that treats it as a percentage), but the value now saturates only when the
    # side really is at full strength, and the *unclamped* ratio is recoverable
    # from ``hp_fraction_ratios`` for anything that wants the honest number.
    return (
        max(0.0, min(1.0, own_hp / max(1.0, own_ref))),
        max(0.0, min(1.0, enemy_hp / max(1.0, enemy_ref))),
    )


def hp_fraction_ratios(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> tuple[float, float]:
    """Unclamped ``(own_hp / ref, enemy_hp / ref)`` -- the honest army-size ratio.

    Companion of :func:`hp_fractions`, which clamps to ``[0, 1]`` for reporting.
    Above 1.0 means "this side grew past its opening army" (on an economy map the
    AI always does), which is information the KPI used to throw away.
    """

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    own_hp = sum(float(unit.get("hp") or 0.0) for unit in own)
    enemy_hp = sum(float(unit.get("hp") or 0.0) for unit in enemies)
    own_max = sum(float(unit.get("max_hp") or 0.0) for unit in own) or 1.0
    enemy_max = sum(float(unit.get("max_hp") or 0.0) for unit in enemies) or 1.0
    own_ref = float(own_max_ref) if own_max_ref else own_max
    enemy_ref = float(enemy_max_ref) if enemy_max_ref else enemy_max
    return (
        max(0.0, own_hp / max(1.0, own_ref)),
        max(0.0, enemy_hp / max(1.0, enemy_ref)),
    )


# Fraction of a unit's combat value that comes from simply being alive.
#
# Why not plain hp: Rusted Warfare units fight at *full* strength until they are
# destroyed -- there is no "damaged units shoot weaker" rule. A tank left at
# 10/210 HP still deals its whole 20 DPS. So the linear ``hp / max_hp`` model of
# a potential function mis-teaches the policy:
#
#   25-vs-25 (10500 total HP), measured:
#     focus-fire one tank dead        -> 210 HP removed  -> +0.0100 potential
#     spread 200 damage over 5 tanks  -> 1000 HP removed -> +0.0476 potential
#   i.e. spreading damage scored 4.8x better even though killing a tank
#   permanently deletes a whole gun while damaging five deletes none.
#
# This weight makes "alive" dominate (a unit that dies loses its whole value)
# and leaves hp as the tie-breaker between survivors: 0.85/0.15 puts a dead
# unit at -210 and five near-dead units at -150, so focusing fire wins while a
# genuinely weakened army still reads as weaker.
COMBAT_ALIVE_WEIGHT = 0.85


#: How much of a unit's threat value is "it is a threat at all" versus "it hits
#: hard". A unit that cannot shoot is worth its support role instead.
#:
#: Why this term exists at all: ``combat_value`` prices a unit by ``max_hp``, and
#: in Rusted Warfare hull size and firepower are only loosely correlated. Measured
#: on ``[p2]Lake (2p)`` the artillery carries **140 hp** while a builder carries
#: **170** and a scout **350**, so the old pricing made the enemy's longest-range
#: gun (290 px, versus 130 for a tank) *cheaper to leave alive than a builder*.
#: ``.runtime/policy_attrition.json`` and the 1200-frame replay
#: ``runs/platform/real-rwpp-20261116-133006-d97c5a`` both show the consequence:
#: the policy never prioritised artillery, and artillery was 100% invisible to it
#: at the same time (8 enemy slots, spawn order), so the two defects reinforced
#: each other.
#:
#: The blend keeps ``max_hp`` as the *scale* (so the potential's units stay
#: comparable with the opening-army reference) and adds a bounded multiplier for
#: what the unit can actually do. Both components are dimensionless, so the
#: potential-function shape -- and therefore policy invariance -- is preserved.
COMBAT_THREAT_WEIGHT = 0.35

#: Reference DPS proxy used to normalise the firepower component. Long-range
#: artillery is the upper end of the stock ladder on the maps this project trains
#: on; anything above it saturates rather than exploding the potential's range.
COMBAT_THREAT_RANGE_SCALE = 300.0


def _unit_threat(unit: dict[str, Any]) -> float:
    """Dimensionless "how dangerous is this unit", in ``[0, 2]``.

    Three inputs, all already reported per entity by the bridge:

    * **firepower** -- ``attack_range`` normalised. This is a *proxy*, not damage:
      stock Rusted Warfare hides damage in ``Projectile.reloadTime`` (the same
      field is the reload timer when cooling down), so a real per-unit damage
      number is not observable. Range is the honest correlate available without
      fabricating a field, and it is the one that decides who wins an exchange:
      artillery out-ranges a tank 290 to 130 and therefore kills it without being
      hit.
    * **speed of fire** -- a shorter ``shoot_delay`` means more shots.
    * **air reach** -- a unit that can shoot aircraft is a counter to a whole unit
      class, so it is worth more than its hull.

    A unit that cannot attack at all scores 0 on firepower and is priced purely by
    hull, which is correct: an extractor is not a threat.
    """

    if not unit.get("can_attack"):
        return 0.0
    try:
        reach = float(unit.get("attack_range") or 0.0)
    except (TypeError, ValueError):
        reach = 0.0
    reach_frac = max(0.0, min(1.0, reach / COMBAT_THREAT_RANGE_SCALE))
    # ``shoot_delay`` is ticks between shots, so it is the *inverse* of a rate.
    # Missing/unparseable reads as the tank's measured 70-75 ticks, i.e. a plain
    # combat unit rather than a mystery.
    try:
        delay = float(unit.get("shoot_delay") or 0.0)
    except (TypeError, ValueError):
        delay = 0.0
    rate = (75.0 / delay) if delay > 0.0 else 1.0
    rate_frac = max(0.0, min(1.0, rate))
    air = 1.0 if unit.get("can_attack_air") else 0.0
    # Weighted so reach dominates: it is what decides an engagement.
    return 1.0 + reach_frac + 0.25 * rate_frac + 0.35 * air


def combat_value(
    hp: float,
    max_hp: float,
    unit: dict[str, Any] | None = None,
) -> float:
    """Combat value of one unit: 0 when destroyed, ~``max_hp`` while alive.

    ``unit`` is optional so every existing positional caller keeps working. When
    it is supplied, the hull value is scaled by :func:`_unit_threat`, which is what
    makes "this artillery is the thing killing my army" score above "this builder
    is walking past" -- see :data:`COMBAT_THREAT_WEIGHT` for the measured reason
    that matters.
    """

    if hp <= 0.0:
        return 0.0
    ceiling = max(1e-6, float(max_hp))
    frac = max(0.0, min(1.0, float(hp) / ceiling))
    value = ceiling * (COMBAT_ALIVE_WEIGHT + (1.0 - COMBAT_ALIVE_WEIGHT) * frac)
    if unit is not None:
        value *= 1.0 + COMBAT_THREAT_WEIGHT * _unit_threat(unit)
    return value


def threat_value(unit: dict[str, Any]) -> float:
    """A unit's full threat value, i.e. what killing it removes from the enemy.

    Exported so the per-kill reward term can price kills by *what died* rather
    than by how many things died. ``kill_bonus`` used to be a flat count
    (``kill_bonus * (enemy_alive_before - enemy_alive_now)``), which made trading a
    tank for the enemy's artillery score exactly the same as trading it for a
    builder.
    """

    try:
        max_hp = float(unit.get("max_hp") or 0.0)
    except (TypeError, ValueError):
        max_hp = 0.0
    if max_hp <= 0.0:
        max_hp = float(DEFAULT_CONFIG.unit_hp)
    return combat_value(max_hp, max_hp, unit)


def combat_value_fractions(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> tuple[float, float]:
    """``(own_combat_frac, enemy_combat_frac)``, each in ``[0, ...]``.

    Normalised by the episode-start max-hp totals (same reference as
    :func:`hp_fractions`), so the two potentials are directly comparable.

    **Not clamped to 1.0, and that is deliberate.** The AI reinforces far past
    the opening army -- measured on Lake: both sides start at 4170 HP (a 4000 HP
    command center plus a 170 HP builder) and the enemy keeps producing until it
    has 24+ units. With a clamp, *every* value above the opening total read as
    exactly 1.0, so ``potential`` was pinned at ``1.0 - 1.0 = 0.0`` for the whole
    episode no matter how large the enemy army grew. Measured over a 40-update
    run: ``potential`` accumulated exactly ``0.000000`` while the enemy went from
    2 to 20 units and the policy lost 7 units. The policy was structurally blind
    to being out-produced, which is precisely what it needed to learn to avoid.
    """

    def total(units: list[dict[str, Any]]) -> float:
        return sum(
            combat_value(
                float(unit.get("hp") or 0.0),
                float(unit.get("max_hp") or 0.0) or float(DEFAULT_CONFIG.unit_hp),
                unit,
            )
            for unit in units
        )

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    own_max = sum(float(unit.get("max_hp") or 0.0) for unit in own) or 1.0
    enemy_max = sum(float(unit.get("max_hp") or 0.0) for unit in enemies) or 1.0
    own_ref = float(own_max_ref) if own_max_ref else own_max
    enemy_ref = float(enemy_max_ref) if enemy_max_ref else enemy_max
    return (
        max(0.0, total(own) / max(1.0, own_ref)),
        max(0.0, total(enemies) / max(1.0, enemy_ref)),
    )


def combat_value_potential(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> float:
    """Combat-value advantage ``own - enemy``.

    This is the term the real-game ``potential`` reward is built from: it pays
    for *destroying* units rather than for spreading chip damage.

    The result is **not** clamped to ``[-1, 1]``: with the opening army as the
    denominator, an enemy that reinforces to four times its starting size reads
    as ``-3.0``, which is the honest signal. Clamping it was what made the term
    a constant zero on every economy map (see
    :func:`combat_value_fractions`). The consumer (``potential_coef``) applies
    the scale, and the shape of the potential function -- not its absolute range
    -- is what keeps the shaping policy-invariant.
    """

    own, enemy = combat_value_fractions(state, own_max_ref, enemy_max_ref)
    return own - enemy


def army_hp_reference(
    state: dict[str, Any],
    config: MicroDuelConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """Episode-start army max-hp totals used to normalise hp fractions.

    This must run on the **ground-truth** state, never on the fogged policy
    view: the bridge streams every unit at reset, so the first frame carries the
    full army value even when ``vision_radius`` hides the enemies from the
    policy. Summing the visible enemy list instead would collapse the reference
    to a single unit (or to the configured fallback) at the start of a fogged
    episode and make every later hp fraction read as near-zero damage.

    ``max_hp`` falls back to ``config.unit_hp`` per entity; if the stream is
    empty the configured ``units_per_team * unit_hp`` is used.
    """

    expected = float(max(1, int(config.units_per_team)) * float(config.unit_hp))
    own_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("own_entities", [])
    )
    enemy_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("visible_enemy_entities", [])
    )
    own_ref = own_total if own_total > 0.0 else expected
    enemy_ref = enemy_total if enemy_total > 0.0 else expected
    return float(own_ref), float(enemy_ref)


def army_distance_norm(state: dict[str, Any], map_diagonal: float) -> float:
    """Normalised distance between the two army centroids (0 = same spot)."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    if not own or not enemies:
        return 0.0
    own_cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
    own_cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
    enemy_cx = sum(float(unit.get("x") or 0.0) for unit in enemies) / len(enemies)
    enemy_cy = sum(float(unit.get("y") or 0.0) for unit in enemies) / len(enemies)
    diagonal = max(1.0, float(map_diagonal))
    return min(1.0, float(np.hypot(enemy_cx - own_cx, enemy_cy - own_cy)) / diagonal)


__all__ = [
    "COMBAT_ALIVE_WEIGHT",
    "COMBAT_THREAT_RANGE_SCALE",
    "COMBAT_THREAT_WEIGHT",
    "army_distance_norm",
    "army_hp_reference",
    "combat_value",
    "combat_value_fractions",
    "combat_value_potential",
    "hp_fraction_ratios",
    "hp_fractions",
    "hp_potential",
    "threat_value",
]
