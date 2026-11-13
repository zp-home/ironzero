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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..micro_config import MicroDuelConfig
from .config import RwppGameConfig
from .reward import hp_fractions

# ``idle_penalty`` only applies inside this multiple of the unit attack range.
# 3x is the "already in the fight" band: our units and the enemy centroid are
# close enough that standing still means eating damage for free, while anything
# farther away is manoeuvring / holding ground, which must not be punished.
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
    strategy_component: float,
    disconnected: bool,
    done: bool,
    own_max_ref: float,
    enemy_max_ref: float,
    episode_components: dict[str, float],
    previous_income_rate: float = 0.0,
    previous_own_buildings: int = 0,
    previous_own_combat: int = 0,
) -> StepShaping:
    """Build the per-step reward terms and adjudicate termination.

    ``episode_components`` is updated in place (running episode totals).
    """

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
    components: dict[str, float] = {
        "potential": float(micro.potential_coef) * (potential - previous_potential),
        "engagement": float(game.engagement_coef) * (previous_distance - distance),
        "strategy": strategy_component,
        "kills": float(game.kill_bonus) * max(0, previous_enemy_alive - enemy_alive),
        "losses": -float(game.loss_penalty) * max(0, previous_own_alive - own_alive),
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
    "compute_step_shaping",
]
