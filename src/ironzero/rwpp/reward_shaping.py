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

# A battle decided by the tick cap is worth slightly less than one won outright
# (the win is real, but it was not a clean sweep).
TIMEOUT_TERMINAL_SCALE = 0.8

# How much larger a surviving army must be to count as the winner when the tick
# cap ends the battle.
#
# The old rule compared hp *fractions* against an absolute 0.08 gap. Those
# fractions are normalised by the episode-start totals, so late in a long battle
# both sides shrink towards zero and the gap became smaller than the threshold
# exactly when the battle was decided: a measured 4-vs-1 finish (0.046 vs 0.040)
# was recorded as a `timeout` and paid terminal 0.0.
#
# Unit count is the honest primary signal (destroying units is the objective),
# with a ratio so that "21 vs 23" is still a draw rather than a defeat.
TIMEOUT_DECISIVE_RATIO = 1.25


def _decide_by_army(
    *,
    own_alive: int,
    enemy_alive: int,
    own_frac: float,
    enemy_frac: float,
) -> str:
    """Adjudicate a tick-capped battle: ``victory`` / ``defeat`` / ``timeout``.

    ``own_frac`` / ``enemy_frac`` are hp fractions against the episode-start
    totals; they are only used as a tie-breaker, because their absolute scale
    shrinks over a long battle. The primary signal is the surviving unit count:
    a battle that ends with a materially larger army left is not a draw.
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
    # Neither side holds a decisive unit advantage. Only an army that is both
    # the same size and clearly healthier breaks the tie; anything close stays a
    # timeout, because nothing was actually decided.
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
    # A bridge-declared timeout (tick cap) or a dead connection truncates the
    # episode; a decided battle terminates it.
    truncated = (not terminated) and (
        done
        or disconnected
        or int(state.get("tick", 0)) >= int(game.episode_ticks)
    )
    if truncated:
        own_frac, enemy_frac = hp_fractions(state, own_max_ref, enemy_max_ref)
        # In annihilation battles or micro duels that reach the tick limit, decide by
        # army survival / HP advantage rather than writing off clear-cut wins as
        # 0-win timeouts (matches standard RTS tournament rules).
        is_micro_battle = (
            game.win_mode == "annihilation"
            or "ironzero" in str(game.map_relative).lower()
            or "ironzero" in str(game.map_name).lower()
            or micro.units_per_team > 0
        )
        if is_micro_battle and not disconnected:
            winner = _decide_by_army(
                own_alive=own_alive,
                enemy_alive=enemy_alive,
                own_frac=own_frac,
                enemy_frac=enemy_frac,
            )
            if winner == "victory":
                last_result = "victory"
                components["terminal"] = float(micro.win_reward) * TIMEOUT_TERMINAL_SCALE
            elif winner == "defeat":
                last_result = "defeat"
                components["terminal"] = float(micro.loss_reward) * TIMEOUT_TERMINAL_SCALE
            else:
                last_result = "timeout"
        else:
            last_result = "timeout"
        reward = float(sum(components.values()))
        episode_components["terminal"] = float(
            episode_components.get("terminal", 0.0) + components["terminal"]
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
