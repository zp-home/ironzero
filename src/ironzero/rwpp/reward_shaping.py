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
            if (own_alive > enemy_alive and own_frac >= enemy_frac + 0.08) or (enemy_alive == 0 and own_alive > 0):
                last_result = "victory"
                components["terminal"] = float(micro.win_reward) * 0.8
            elif (enemy_alive > own_alive and enemy_frac >= own_frac + 0.08) or (own_alive == 0 and enemy_alive > 0):
                last_result = "defeat"
                components["terminal"] = float(micro.loss_reward) * 0.8
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


__all__ = ["StepShaping", "compute_step_shaping"]
