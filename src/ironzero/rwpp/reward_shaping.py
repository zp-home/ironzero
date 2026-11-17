# SPDX-License-Identifier: Apache-2.0
"""Reward decomposition for one ``RwppDuelEnv.step`` decision.

Extracted verbatim from the former single-file ``step()`` so the reward math is
readable on its own. Two invariants are load-bearing and must not change:

* the ``components`` dict is built in exactly the historical key order, because
  ``reward`` is ``sum(components.values())`` and floating-point addition is
  order sensitive -- the reward assertions in ``tests/test_rwpp_env.py`` depend
  on the last bits;
* ``episode_components`` is accumulated *before* the truncation adjudication
  re-reads/rewrites ``components["terminal"]``, exactly as before.

Combat terms only count units that can actually fight
-----------------------------------------------------

``engagement`` and ``idle`` are now measured on the centroid of the
**attack-capable** units on each side (see :func:`unit_can_attack`), not on the
whole-army centroid. The whole-army centroid was mixing non-combatants into a
combat signal, and the builder is the worst offender because it spends the game
travelling on its own:

* Lake is 2600x2600 px, so the map diagonal is ~3677 px. Our opening army is a
  stationary command center plus one builder, so every pixel the builder walks
  moves the *whole-army* centroid by half a pixel. Across the builder's
  1786 px -> 100 px approach that is 843 px of centroid motion, i.e. a phantom

      ``0.3 * 843 / 3677 = +0.0688``

  of ``engagement`` paid to the whole policy for a unit that cannot shoot.
  With no combat unit on our side the honest value of ``engagement`` is
  ``0.0``, so that ``+0.0688`` was pure noise, and it pointed the policy's
  only dense combat gradient at "walk the builder at the enemy".

  To be precise about what this does and does not buy: it does **not** make
  builder suicide unprofitable through this term alone. The *whole* round trip
  is worth ``0.3 * (1786 - 100) / 3677 = +0.1376`` while dying costs
  ``loss_penalty = 0.25``, so a one-way suicide run nets about ``-0.11`` and was
  never a winning move in this term. The real reason the builder wanders is
  upstream of the reward -- a uniform-random policy picks its group and its
  kind independently (builder owns a whole group of its own), so walking it into
  the enemy is the default behaviour, and every ``move``/``attack_move`` order
  also cancels an in-flight build. Fixing *that* needs the observation to expose
  attack capability and the action space to express "keep building + hold"; this
  module's job is only to stop the combat signal from being polluted.

* ``idle`` had the same defect plus a magnitude problem. It charges
  ``idle_penalty = 0.02`` on *every* decision, and an episode is ~266 decisions,
  so a policy that holds still in range can be charged ``-5.32`` in total --
  more than five times ``win_reward = 1.0`` and twenty-six times
  ``timeout_penalty = 0.20``. Keyed off the whole-army centroid, a builder that
  reached the enemy and correctly stopped to stay alive was billed the whole
  time. Now the band test uses the combat centroid, and with no attack-capable
  unit on our side ``idle`` is exactly ``0.0`` -- which is both the correct
  reading of "there is nothing to keep in the fight" and the fix for the
  magnitude blow-up.

``previous_combat_distance`` / ``combat_distance`` are optional and carry a
sentinel default, so callers that predate the combat centroid (tests, offline
harnesses) keep the historical whole-army-centroid behaviour rather than
silently reading ``engagement == 0``. The two modes are never mixed: both
endpoints of the difference always come from the *same* centroid definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .config import RwppGameConfig
from .reward import hp_fractions, threat_value

# ``idle_penalty`` only applies inside this multiple of the unit attack range.
# 3x is the "already in the fight" band: our units and the enemy centroid are
# close enough that standing still means eating damage for free, while anything
# farther away is manoeuvring / holding ground, which must not be punished.
#
# The band is measured against the *combat* centroid (see the module docstring),
# so the "our units" in the sentence above means "our units that can shoot".
ENGAGED_RANGE_MULTIPLIER = 3.0

# A battle decided by the tick cap is worth slightly less than one won outright.
#
# DEPRECATED: no longer used. The tick cap no longer awards a win or a loss at
# all -- it pays ``timeout_penalty`` -- because the engine's own verdict is
# authoritative. Kept exported only so existing imports keep resolving.
TIMEOUT_TERMINAL_SCALE = 0.8

# How much larger a surviving army had to be to be declared the winner by the
# old heuristic.
#
# DEPRECATED along with ``_decide_by_army``. See the module docstring of
# ``tests/test_timeout_adjudication.py`` and ``RwppGameConfig.timeout_penalty``
# for why the heuristic was removed: the engine already reports the result, and
# every case the heuristic actually fired on was a stalemate.
TIMEOUT_DECISIVE_RATIO = 1.25


# Sentinel for "the caller did not supply a combat-centroid distance".
#
# ``None`` cannot be the sentinel: ``combat_army_distance_norm`` returns ``None``
# for the perfectly ordinary situation "one side has no attack-capable unit",
# and that case must fall through to ``engagement = 0.0`` -- not to the legacy
# whole-army path, which is exactly the pollution being removed. A private
# object distinguishes "not provided" from "provided, but undefined".
_UNSET_DISTANCE: Any = object()


def unit_can_attack(unit: dict[str, Any]) -> bool:
    """Whether a bridge entity can deal damage, for the combat reward terms.

    Prefers the bridge's own ``can_attack`` field, which the extended
    ``entityJson()`` populates from the engine's real weapon data.

    **Fallback when the field is absent.** Legacy bridges (and the offline mock
    engine) stream neither ``can_attack`` nor ``attack_range``, so an
    unconditional read would silently count every entity as non-combatant and
    collapse ``engagement`` to a constant ``0.0`` on those paths. The
    conservative reading is "anything that is not a building and not a builder
    can fight", i.e. treat the entity stream the way the rest of this module
    already does when it counts combat units (``bonus_produce``). Buildings have
    no weapon of their own here, and the builder is precisely the unit this
    change exists to exclude, so neither may be assumed combat-capable.

    ``can_attack`` is checked with an explicit ``is None`` test rather than
    truthiness so that a bridge which *does* report ``false`` is taken at its
    word: an explicit ``false`` is data, only a missing field is uncertainty.
    """

    explicit = unit.get("can_attack")
    if explicit is not None:
        return bool(explicit)
    return not unit.get("is_builder") and not unit.get("is_building")


def targetable_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of ``units`` that can participate in a fight."""

    return [unit for unit in units if unit_can_attack(unit)]


def combat_centroid(
    units: list[dict[str, Any]],
) -> tuple[float, float] | None:
    """Mean position of the attack-capable units, or ``None`` if there are none.

    ``None`` is meaningful and callers must not substitute the whole-army
    centroid: "we have no weapon in play" has to read as "no combat signal",
    not as "whatever the builder is doing".
    """

    combat = targetable_units(units)
    if not combat:
        return None
    return (
        sum(float(unit.get("x") or 0.0) for unit in combat) / len(combat),
        sum(float(unit.get("y") or 0.0) for unit in combat) / len(combat),
    )


def combat_army_distance_norm(
    state: dict[str, Any], map_diagonal: float
) -> float | None:
    """Normalised gap between the two **combat** centroids.

    Counterpart of :func:`ironzero.rwpp.reward.army_distance_norm`, which keeps
    its whole-army-centroid meaning for every existing caller. Returns ``None``
    -- deliberately *not* ``0.0`` and not a whole-army fallback -- when either
    side has no attack-capable unit, because a distance is undefined there and
    ``0.0`` would read downstream as "the armies are touching".

    The normalisation (and the ``1.0`` clamp) matches ``army_distance_norm``
    exactly, so the two are interchangeable as time series and only ever differ
    by *which* centroid they follow.
    """

    own_c = combat_centroid(state.get("own_entities", []))
    if own_c is None:
        return None
    enemy_c = combat_centroid(state.get("visible_enemy_entities", []))
    if enemy_c is None:
        return None
    diagonal = max(1.0, float(map_diagonal))
    return min(
        1.0,
        float(math.hypot(enemy_c[0] - own_c[0], enemy_c[1] - own_c[1])) / diagonal,
    )


def _decide_by_army(
    *,
    own_alive: int,
    enemy_alive: int,
    own_frac: float,
    enemy_frac: float,
) -> str:
    """DEPRECATED: surviving-army heuristic for a tick-capped battle.

    No longer used by the reward path. The engine decides the match (see
    ``RwppGameConfig.timeout_penalty``); this function remains only because
    external tooling and older tests import it, and because its *documented*
    behaviour is still a useful description of what a stalemate looks like.

    Do not reintroduce it as the source of truth: it was patched three times (an
    absolute hp gap, then a unit-count ratio, then a disconnected carve-out) and
    still recorded a 2-vs-8 loss as a draw.
    """

    if enemy_alive == 0 and own_alive > 0:
        return "victory"
    if own_alive == 0 and enemy_alive > 0:
        return "defeat"
    if own_alive == 0 and enemy_alive == 0:
        return "timeout"
    ratio = TIMEOUT_DECISIVE_RATIO
    if own_alive >= enemy_alive * ratio:
        return "victory"
    if enemy_alive >= own_alive * ratio:
        return "defeat"
    return "timeout"


@dataclass
class StepShaping:
    """Everything ``step()`` needs after the reward math has run."""

    components: dict[str, float]
    reward: float
    terminated: bool
    truncated: bool
    last_result: str | None


def _building_tier_total(state: dict) -> int:
    """Tiers our *buildings* have gained above the starting T1.

    The quantity ``bonus_upgrade`` pays for, so what it must **not** include decides
    its correctness:

    * **not units** -- every unit reports ``tech_level`` 1, so a per-entity sum makes
      "make a scout" identical to "upgrade an extractor";
    * **not the starting tier** -- a building begins at T1, so counting
      ``tech_level`` outright pays the bonus for the command centre existing.

    A payload that reports no ``tech_level`` yields 0, which keeps the term
    identically zero on the mock engine and on older bridges -- the same
    "missing field reads as zero" contract the other counters use.
    """

    total = 0
    for unit in state.get("own_entities") or []:
        if not isinstance(unit, dict) or not unit.get("is_building"):
            continue
        try:
            level = int(float(unit.get("tech_level") or 0))
        except (TypeError, ValueError):
            continue
        if level > 1:
            total += level - 1
    return total


def _weighted_attrition(
    before: list[dict[str, Any]] | None,
    after: list[dict[str, Any]] | None,
) -> float:
    """How much threat disappeared between two observations, in baseline units.

    Priced with :func:`ironzero.rwpp.reward.threat_value`, normalised so one plain
    combat unit scores ``1.0``. That keeps the historical scale ("one kill is worth
    ``kill_bonus``") while making the *identity* of the casualty matter.

    Matching is by ``entity_id`` when both sides report it, because that is the only
    stable handle: position changes, the stream order changes, and slot index is
    not an identity. A payload without ids (mock engine, hand-written fixtures)
    falls back to the bare count difference so those environments keep their
    previous behaviour exactly.

    Returns ``0.0`` when nothing disappeared, and never a negative value -- a unit
    appearing is not a kill.
    """

    if not before or not after:
        return 0.0
    ids_before = {
        int(unit.get("entity_id")): unit
        for unit in before
        if isinstance(unit, dict) and unit.get("entity_id") is not None
    }
    ids_after = {
        int(unit.get("entity_id"))
        for unit in after
        if isinstance(unit, dict) and unit.get("entity_id") is not None
    }
    if not ids_before or not ids_after:
        # No usable ids: keep the historical count semantics.
        return float(max(0, len(before) - len(after)))
    gone = [unit for key, unit in ids_before.items() if key not in ids_after]
    if not gone:
        return 0.0
    # ``threat_value`` returns ``max_hp * (1 + threat)``; dividing by a baseline
    # hull turns it back into "how many plain units was this worth".
    baseline = max(1.0, float(DEFAULT_CONFIG.unit_hp))
    return float(sum(threat_value(unit) for unit in gone) / baseline)


def compute_step_shaping(
    *,
    micro: MicroDuelConfig,
    game: RwppGameConfig,
    state: dict[str, Any],
    actions: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    message: dict[str, Any],
    bridge_result: Any,
    potential: float,
    previous_potential: float,
    distance: float,
    previous_distance: float,
    own_alive: int,
    enemy_alive: int,
    previous_own_alive: int,
    previous_enemy_alive: int,
    previous_own_units: list[dict[str, Any]] | None = None,
    previous_enemy_units: list[dict[str, Any]] | None = None,
    strategy_component: float,
    disconnected: bool,
    done: bool,
    own_max_ref: float,
    enemy_max_ref: float,
    episode_components: dict[str, float],
    previous_income_rate: float = 0.0,
    previous_own_buildings: int = 0,
    previous_own_combat: int = 0,
    previous_own_tier_total: int = 0,
    previous_own_defences: int = 0,
    combat_distance: float | None = _UNSET_DISTANCE,
    previous_combat_distance: float | None = _UNSET_DISTANCE,
) -> StepShaping:
    """Build the per-step reward terms and adjudicate termination.

    ``episode_components`` is updated in place (running episode totals).

    ``combat_distance`` / ``previous_combat_distance`` are the normalised gap
    between the two *attack-capable* centroids, for the current and previous
    decision. They must be supplied together (or updated together by the caller)
    and come from the same function, so that ``previous - current`` telescopes
    across the episode. Leaving both at their defaults keeps the historical
    whole-army-centroid behaviour for every pre-existing caller; supplying only
    one is a caller bug and raises.
    """

    supplied = tuple(
        value is not _UNSET_DISTANCE
        for value in (combat_distance, previous_combat_distance)
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "combat_distance and previous_combat_distance must be supplied "
            "together: measuring the two endpoints of engagement against "
            "different centroid definitions leaves a constant offset in the "
            "reward instead of telescoping it away"
        )
    combat_mode = all(supplied)

    economy_stats = message.get("global_state", {})
    # ``idle`` only punishes "the enemy is on top of us and we are standing
    # still". Holding a chokepoint or waiting for the opponent to commit is a
    # standard strong play in Rusted Warfare, but the old test was "we issued
    # stop/hold while any enemy is alive" -- which penalised exactly that play
    # and pushed the policy into always charging. Measured on a 105-update run:
    # the idle term accumulated -3.344 while every combat term stayed at 0.000,
    # i.e. the only thing the policy could learn was "do not hold ground".
    # ``distance`` is the army-centroid gap normalised by the map diagonal, so
    # the "already engaged" radius converts back through the same diagonal.
    map_diagonal = float(
        math.hypot(float(micro.map_width), float(micro.map_height))
    ) or 1.0
    engaged_radius = float(micro.attack_range) * ENGAGED_RANGE_MULTIPLIER
    # The band test uses the *combat* centroid when the caller provides one, so a
    # builder alone out in the field can never trigger the penalty. ``idle``
    # charges every decision it fires, so with ~266 decisions per episode a
    # builder parked next to the enemy was billing -0.02 each step (~-5.3 total)
    # for correctly refusing to die -- 26x ``timeout_penalty`` and 5x
    # ``win_reward``. When our side has no weapon in play the honest value is
    # "nothing is idling that should be fighting", i.e. no penalty at all.
    if combat_mode:
        in_range_reference = combat_distance
        # ``None`` here means one side has no attack-capable unit: no band.
        near_enemy_in_range = (
            in_range_reference is not None
            and (float(in_range_reference) * map_diagonal) <= engaged_radius
        )
    else:
        near_enemy_in_range = (float(distance) * map_diagonal) <= engaged_radius
    # ``bonus_build`` must be paid for a *completed building*, not for an
    # accepted order. Paying per accepted receipt let a policy farm it outright:
    # re-issuing ``build_extractor`` every decision returns accepted=true each
    # time (the builder simply queues again), measured at +0.2/step over a whole
    # episode. Counting the own-building increase instead is one-off per
    # structure and cannot be farmed by repeating an order.
    own_buildings = sum(
        1 for unit in state.get("own_entities", []) if unit.get("is_building")
    )
    new_buildings = max(0, int(own_buildings) - int(previous_own_buildings))
    # ``bonus_produce`` closes the last gap in the economy loop: building was
    # rewarded and promotion was rewarded, but *using* the factory to add a
    # combat unit paid nothing. A policy that finished a factory and then never
    # queued anything earned exactly the same return as one that did, so the
    # only gradient left was "build the cheapest thing, then stop".
    #
    # Counted from the own-entity stream (buildings excluded), which is the same
    # ground truth ``bonus_build`` uses. Producers are excluded so that making a
    # builder -- which every factory's menu offers and is the cheapest entry --
    # cannot farm this term.
    own_combat = sum(
        1
        for unit in state.get("own_entities", [])
        if not unit.get("is_building") and not unit.get("is_builder")
    )
    new_combat = max(0, int(own_combat) - int(previous_own_combat))
    # Tier upgrades actually achieved, and defensive structures actually finished.
    #
    # Both are measured from the *observation stream*, never from an accepted
    # receipt, for the same reason ``bonus_build`` counts new buildings: repeating an
    # order is accepted every single time (measured: ``build_progress`` pinned at
    # 0.0 while every receipt said ``accepted: true``), so paying per receipt is a
    # farmable bug rather than a signal.
    #
    # ``tech_level`` is the engine's own tier, reported per entity by the bridge.
    # Summing it over our **buildings** and taking the positive difference is
    # exactly "some building gained a tier", which is what an upgrade *is*.
    # Summing rather than counting buildings above T1 matters because a destroyed
    # T2 building must not look like an upgrade elsewhere.
    #
    # Two filters are load-bearing, and both were measured rather than reasoned:
    #
    # * **buildings only.** Every *unit* carries ``tech_level`` 1, so a per-entity
    #   sum cannot tell "produced a scout" from "upgraded an extractor". With the
    #   unfiltered sum this term fired on **609 of 609 decisions** of a real
    #   difficulty-1 episode while **no building ever left T1** -- the policy was
    #   paid an upgrade bonus for every unit it made, on top of ``bonus_produce``,
    #   and duly spent 15 straight updates massing scouts with no economy at all
    #   (``.runtime/tier_reward_audit.json``, ``.runtime/tier_reward_verify.json``);
    # * **tiers above the starting T1.** Otherwise the command centre pays the bonus
    #   once for existing at episode start.
    #
    # This lives here rather than being imported from ``rwpp.env`` to keep
    # ``reward_shaping`` free of an env import (``env`` imports this module), so the
    # two implementations are kept identical by
    # ``tests/test_gameplay_gaps.py::test_tier_total_counts_buildings_and_matches_reward_shaping``
    # rather than by hope.
    own_tier_total = _building_tier_total(state)
    tier_gain = max(0, own_tier_total - int(previous_own_tier_total))
    own_defences = sum(
        1
        for unit in state.get("own_entities", []) or []
        if isinstance(unit, dict)
        and unit.get("is_building")
        # ``is_defence`` only, deliberately -- **not** ``or is_turret``.
        #
        # Measured on ``[p2]Lake (2p)`` (``.runtime/q2_turret.json``): a finished
        # stock gun turret reports ``unit_type = c_turret_t1`` with
        # ``is_turret: FALSE`` but ``is_defence: true`` and ``can_attack: true``.
        # The bridge computes ``is_turret`` as ``unit instanceof Turret``, and the
        # stock turret is an ``.ini``-defined CustomUnit (``turret_t1.ini`` declares
        # ``name: c_turret_t1`` / ``overrideAndReplace: turret``), so the instanceof
        # test misses it. ``is_defence`` is a derived capability and is correct.
        #
        # The main building and production facilities are excluded because
        # ``is_defence`` means "can shoot", not "is a turret": a ``commandCenter``
        # reports ``is_defence: true`` with ``attack_range: 280``
        # (``.runtime/q7d_sea_produce.json``). Counting it would pay this bonus once
        # at episode start for a building the agent never chose to build, and would
        # hide the first real turret behind a counter that already read 1.
        and unit.get("is_defence")
        and not unit.get("is_factory")
        and not unit.get("is_factory_building")
    )
    new_defences = max(0, own_defences - int(previous_own_defences))
    # Losing a building is a strategic loss, not a unit loss: the player forfeits
    # production / income capacity, which the hp-based ``losses`` term cannot
    # express (a command center's hp is huge, but its *value* is the capability).
    # Charged once per building that disappears, and deliberately smaller than a
    # won game so it shapes behaviour without dominating the terminal signal.
    own_buildings_now = int(own_buildings)
    lost_buildings = max(0, int(previous_own_buildings) - own_buildings_now)
    # Income is a *rate*, and a level rate integrated over a 5000-decision
    # episode would swamp the +/-1.0 terminal signal: at 18 credits/s the old
    # ``extractor_reward * income/10`` paid ~0.9 every decision, i.e. ~4500 per
    # episode. Reward the *growth* of income instead (a new extractor or an
    # upgrade is a one-off step change in the rate), which is dense but does not
    # accumulate, exactly like ``potential``.
    # A terminal ``done`` message carries no ``global_state``, so a naive
    # ``income or 0.0`` reads the rate as 0 and books a full -rate collapse on
    # the last step (measured: -0.9 for an active extractor). Only treat the
    # reading as real when the bridge actually reported it; otherwise carry the
    # previous rate forward so the terminal step pays nothing for economy.
    if "income" in economy_stats:
        income_rate = float(economy_stats.get("income") or 0.0)
    else:
        income_rate = float(previous_income_rate)
    income_delta = income_rate - float(previous_income_rate)
    # ``engagement`` is the closing speed of the two armies. Measured on the
    # **combat** centroids when the caller supplies them, because the whole-army
    # centroid let a non-combatant move a combat signal: on Lake (2600x2600,
    # diagonal ~3677 px) our opening army is a stationary command center plus one
    # builder, so the builder's 1786 px -> 100 px walk drags the whole-army
    # centroid 1686/2 = 843 px and pays ``0.3 * 843 / 3677 = +0.0688`` to the
    # entire policy. With no attack-capable unit on our side the correct value is
    # ``0.0``, so that ``+0.0688`` was noise pointing the only dense combat
    # gradient at "walk the builder into the enemy".
    #
    # ``None`` on either endpoint means there is no combat distance to measure
    # (one side has no weapon in play), which is a genuine ``0.0`` -- never a
    # fallback to the whole-army centroid, which is the pollution itself.
    if combat_mode:
        engagement_value = (
            0.0
            if combat_distance is None or previous_combat_distance is None
            else float(game.engagement_coef)
            * (float(previous_combat_distance) - float(combat_distance))
        )
    else:
        engagement_value = float(game.engagement_coef) * (previous_distance - distance)
    # ``kills`` / ``losses`` are weighted by **what** was destroyed, not by how
    # many things were.
    #
    # A flat count made every trade identical: killing the enemy's artillery --
    # 290 px range against a tank's 130, i.e. the unit that wins exchanges without
    # being hit -- scored exactly the same as killing a builder. Measured on
    # ``[p2]Lake (2p)`` the artillery carries **140 hp** against a builder's 170,
    # so the old ``max_hp``-priced ``potential`` made it the *cheaper* of the two to
    # leave alive. Together with the fact that artillery never entered the policy's
    # 8 enemy observation slots (0 of 500 unit-frames visible in
    # ``runs/platform/real-rwpp-20261116-133006-d97c5a``), the agent had both no
    # sight of its highest-output threat and no gradient toward it.
    #
    # ``threat_value`` prices a unit by hull *and* by what it can do, normalised so
    # one baseline combat unit is worth 1.0. The count difference is still the
    # trigger -- only the units that actually disappeared are priced -- so this
    # cannot be farmed by creating units.
    kill_weight = _weighted_attrition(
        previous_enemy_units, state.get("visible_enemy_entities") or []
    )
    loss_weight = _weighted_attrition(
        previous_own_units, state.get("own_entities") or []
    )
    components: dict[str, float] = {
        "potential": float(micro.potential_coef) * (potential - previous_potential),
        "engagement": engagement_value,
        "strategy": strategy_component,
        "kills": float(game.kill_bonus) * kill_weight,
        "losses": -float(game.loss_penalty) * loss_weight,
        "income": float(micro.extractor_reward) * income_delta / 10.0,
        # A fixed +5.0 dwarfed the +/-1.0 terminal reward, so once building
        # actually worked the cheapest build would dominate the return and the
        # policy would farm extractors instead of learning to win. Tie it to the
        # game's own economy knob and keep it below a win.
        "bonus_build": float(game.build_bonus) * float(new_buildings),
        "invalid": -float(game.invalid_penalty) * sum(1 for r in receipts if not r.get("accepted", True)),
        "idle": (
            -float(game.idle_penalty)
            if (
                actions
                and actions[0].get("kind") in {"stop", "hold"}
                and enemy_alive > 0
                and near_enemy_in_range
            )
            else 0.0
        ),
        "terminal": 0.0,
        "timeout": 0.0,
        # Appended *after* the historical keys on purpose: ``reward`` is
        # ``sum(components.values())`` and floating-point addition is order
        # sensitive, so adding the new terms at the end keeps every pre-existing
        # prefix bit-identical (the reward assertions depend on it).
        "bonus_produce": float(game.produce_bonus) * float(new_combat),
        "building_lost": -float(game.building_loss_penalty) * float(lost_buildings),
        # Appended after every historical key, for the reason given above.
        "bonus_upgrade": float(game.upgrade_bonus) * float(tier_gain),
        "bonus_defence": float(game.defence_bonus) * float(new_defences),
        # The economy's growth signal, one term per *permanent* rate increase.
        #
        # The historical ``income`` term above pays ``extractor_reward * delta / 10``,
        # i.e. **0.04** for a whole new extractor (delta 8 credits/s / 10 * 0.5). Against
        # a terminal that is worth +/-1.0 that is a 25:1 ratio, so an agent deciding
        # whether to spend 700 credits on an extractor or on two scouts sees almost no
        # reward difference at the moment of the choice: the extractor's value only
        # arrives hundreds of decisions later, through a terminal it cannot yet
        # correlate. Measured on the two-slot layout at difficulty 1: after 34 updates
        # the policy had built **zero extractors** and an income of exactly the 18.0
        # base rate, while massing scouts (``best extractors ever: 0``).
        #
        # ``income_growth`` pays the *fraction of the starting rate* that was added,
        # so a first extractor (18 -> 26, +44%) is worth ``economy_growth_bonus * 0.44``
        # and a full six-extractor economy is worth a few of these in total. It is a
        # rate *change*, not a rate, so it cannot be farmed by sitting still, and it is
        # bounded well below a win, so it shapes the economy without replacing the
        # objective.
        #
        # This is teaching, not scripting: it says "raising your income is good", and
        # leaves *how* (extractors, upgrades, how many, where) entirely to the policy.
        "income_growth": (
            float(game.economy_growth_bonus)
            * (income_delta / max(1.0, float(game.base_income_reference)))
            if income_delta > 0.0
            else 0.0
        ),
        # Potential-based shaping on the income *level*, which is what makes saving
        # for an expensive building rational.
        #
        # Measured crowding-out (``.runtime/affordability_trap.py``): the funds the
        # policy holds sit at **100-650** for a whole episode, while an extractor
        # costs **700**. So `build_extractor` is honestly closed by the mask (the
        # engine would refuse it) and the policy simply never accumulates. Meanwhile a
        # 250-credit tank pays `bonus_produce` 0.10 immediately and is always
        # affordable.
        #
        # The economics are not close: an extractor is worth **+8 credits/s for the
        # remaining ~251 s = ~2000 credits**, about eight tanks, repeating. The reward
        # paid 0.22 for it against 0.10 per tank -- a signal off by an order of
        # magnitude, which is exactly why the policy never saved.
        #
        # Raising ``economy_growth_bonus`` would be the wrong fix: it is a one-off
        # payment for an event, and making it big enough to beat tanks would teach
        # "build one extractor then stop". What the policy actually needs is a reason
        # to *hold a productive state*, so this is potential-based:
        #
        #     r' = r + gamma * Phi(s') - Phi(s),   Phi(s) = w * income(s) / scale
        #
        # Ng, Harada & Russell (1999) prove this leaves the optimal policy unchanged,
        # so it cannot teach a wrong lesson -- it only makes the correct one visible:
        # income is an asset, letting it fall costs reward, and the payment scales with
        # the rate rather than with the single act of building. A sixth extractor is
        # worth six times the potential of the first, which is what makes continued
        # expansion attractive rather than a one-off.
        "income_potential": (
            float(game.economy_potential_coef)
            * (income_rate - float(previous_income_rate))
            / max(1.0, float(game.base_income_reference))
        ),
    }

    terminated = False
    last_result: str | None = None
    if bridge_result in {"victory", "defeat"}:
        # Native game-over signal (main buildings destroyed / team defeated):
        # trust the engine instead of waiting for total annihilation.
        terminated = True
        last_result = str(bridge_result)
        components["terminal"] = float(
            micro.win_reward if bridge_result == "victory" else micro.loss_reward
        )
    elif bridge_result is None:
        # Legacy/offline bridge without a result field: fall back to the
        # annihilation heuristic.
        if enemy_alive == 0 and own_alive > 0:
            terminated = True
            last_result = "victory"
            components["terminal"] = float(micro.win_reward)
        elif own_alive == 0:
            terminated = True
            last_result = "defeat"
            components["terminal"] = float(micro.loss_reward)
    reward = float(sum(components.values()))
    for key, value in components.items():
        episode_components[key] = episode_components.get(key, 0.0) + float(value)
    # The engine decides the match; the tick cap is only a safety fuse.
    #
    # ``bridge_result`` is the game's own verdict (``victoryScreenShown`` /
    # ``defeatScreenShown`` / ``wipedOut`` forwarded by the bridge). When it is
    # present, it is used verbatim and nothing else may override it.
    #
    # When it is absent and the episode ends, the honest outcome is ``timeout``:
    # we did not win inside the allowed time. This deliberately replaces the old
    # ``_decide_by_army`` heuristic, which inferred a winner from surviving-unit
    # ratios. Two measurements showed the heuristic was the wrong tool:
    #
    # * the engine already knows the result, so the rule was second-guessing it;
    # * when it fired it was describing a stalemate, not a decision -- a
    #   39000-tick Lake episode ended with the enemy holding 22 units against our
    #   1 and the engine still reported *nothing*, because the built-in AI never
    #   left its base. Calling that "defeat" was a guess dressed as a verdict.
    truncated = (not terminated) and (
        done
        or disconnected
        or int(state.get("tick", 0)) >= int(game.resolved_episode_ticks())
    )
    if truncated:
        # A truncation without a native verdict is a timeout, and a timeout costs
        # the policy something: "ran out of time" must not be free, or the agent
        # learns that stalling is as good as winning.
        last_result = "timeout"
        components["timeout"] = -float(game.timeout_penalty)
        reward = float(sum(components.values()))
        episode_components["timeout"] = float(
            episode_components.get("timeout", 0.0) + components["timeout"]
        )
    return StepShaping(
        components=components,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        last_result=last_result,
    )


__all__ = [
    "ENGAGED_RANGE_MULTIPLIER",
    "StepShaping",
    "TIMEOUT_DECISIVE_RATIO",
    "TIMEOUT_TERMINAL_SCALE",
    "combat_army_distance_norm",
    "combat_centroid",
    "compute_step_shaping",
    "targetable_units",
    "unit_can_attack",
]
