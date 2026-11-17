# SPDX-License-Identifier: Apache-2.0
"""Per-decision plumbing for :class:`ironzero.rwpp.env.RwppDuelEnv`.

``RwppDuelEnv.step`` is a long orchestration; the mechanical parts -- building
and sending the ``step`` message, folding the reply into the environment state,
packing ``info`` -- live here as functions taking the environment as ``env``.
Nothing here changes values or ordering: it is the same code, given a name.

The environment keeps the historical method names it exposes to the outside
world; these are internal only.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ..observations import build_masks_for_config as build_masks
from .decoding import decode_action, decode_action_multi, merge_done_message, observation_to_state

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RwppDuelEnv


def send_step(
    env: "RwppDuelEnv",
    action: np.ndarray,
    extra_actions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Decode ``action``, send it to the bridge; return ``(actions, disconnected)``.

    Decoding happens against ``env.obs_state`` on purpose: with fog enabled the
    target slots index the *visible* enemy list (as masked in ``obs_state``), so
    using the ground-truth list here would fire at the wrong enemy.

    ``extra_actions`` are already-decoded ``ironzero/1`` orders that must be
    dispatched **in the same step**, after the policy's own orders. They carry the
    opening curriculum (see ``RwppDuelEnv._curriculum_actions``).

    Why they ride along instead of being sent as a second message
    ------------------------------------------------------------
    A second ``step`` message is a *second dispatch on a later tick*, not an extra
    order in the same decision. Measured on a masked-random Lake episode
    (``.runtime/extractor_trace.json``, 220 decisions): the curriculum re-issued
    ``build_extractor`` on every decision and the extractor's ``build_progress``
    stayed at exactly **0.0000** the whole time, because the policy's own tactical
    order (``attack_move``/``move_group``/``stop``, ~half of all sampled kinds)
    reached the builder first, cancelled the in-flight construction, and only then
    did the curriculum re-issue it. Progress was reset before it could accumulate,
    so the build never completed and nothing downstream could fire.

    Sharing one dispatch makes "last order wins" apply the way it does for a human
    player: the policy's order is queued first and the curriculum's last, so the
    builder keeps building while the rest of the army obeys the policy.

    Note the limit of this mechanism, measured rather than assumed: appending the
    curriculum order last does **not** prevent the policy's own tactical order from
    reaching the builder, because each entry in one dispatch is applied to whoever
    it addresses. On a masked-random Lake episode
    (``.runtime/group_collision.json``, 900 decisions) **289 of 995** policy orders
    (29%) targeted the builder's group or the broadcast group, and the factory
    stayed at ``build_progress`` 0.0 for the whole episode. Silently rewriting the
    policy's order (e.g. re-addressing it away from the builder) would be worse
    than the symptom: the executed action would no longer be the sampled action,
    so the PPO ratio and the advantage would both be computed against an action the
    environment never took. The exploration barrier is therefore closed for
    *deterministic* helpers and for a policy that already tends to leave the builder
    alone, and remains open for a uniformly random one -- which is a property of the
    task, not something a flag can paper over.

    What this costs, measured end to end
    ------------------------------------
    ``.runtime/ppo_interference.json`` sends one order per decision (which is what
    an untrained PPO policy does) and records that the order arrives while a
    structure is unfinished on **1949 of 1949** decisions. Final state: ``occupied
    = 1``, ``income = 26``, three own entities. The noop baseline -- same map, same
    seed family, same curriculum, but issuing *nothing* -- reaches ``occupied =
    3-4``, ``income = 42-50`` and a 93% win rate.

    So on this map the trained policy's measured contribution was **negative**:
    43.75% (7 wins / 16, ``runs/platform/rwpp-ppo-ppo-*``) against 93.75% for the
    do-nothing policy. The gap is not a tuning problem; it is the policy cancelling
    the construction that produces its own army. Two honest fixes exist and neither
    is a flag: (a) a two-slot action layout so an economy order and a tactical order
    can coexist (``decoding.ECONOMY_ACTION_KINDS`` already exists, unused), or (b)
    train with the builder masked out of the policy's tactical groups.
    """

    # Layout dispatch. This is the *only* place the main ``step`` path decodes the
    # policy's action, so it is the one seam that has to know about the two-slot
    # layout (``env.py``'s other ``decode_action`` call is the opening
    # curriculum's, which always speaks the single-slot vocabulary).
    #
    # Why an explicit dispatch instead of one decoder: ``decode_action`` validates
    # the vector length against ``len(config.action_nvec)`` (10 under the two-slot
    # layout, so *validation passes*) and then ``_decode_heads`` reads only
    # ``vector[0..4]``. A 10-head vector would therefore be silently decoded as
    # if its first five entries were a single slot -- the tactical slot dropped,
    # no exception raised, and PPO computing ratios against an order the
    # environment never executed. Measured: ``[5,0,0,0,0, 1,1,3,3,0]`` decoded to
    # one ``build_extractor`` and lost the ``attack_move``.
    if getattr(env.micro, "is_two_slot", False):
        actions = decode_action_multi(action, env.obs_state, env.micro)
    else:
        actions = decode_action(action, env.obs_state, env.micro)
    if extra_actions:
        actions = list(actions) + list(extra_actions)
    decision_id = f"d-{env.episode_id}-{env.episode_steps}"
    payload = {
        "protocol": "ironzero/1",
        "type": "step",
        "message_id": f"m-{decision_id}",
        "episode_id": env.episode_id,
        "tick": env.last_tick,
        "decision_id": decision_id,
        "actions": actions,
    }
    try:
        env._socket.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    except OSError:
        # The game died / the bridge was closed under us. Do not raise: let the
        # episode end and let the vector env restart this slot.
        disconnected = True
    else:
        disconnected = False
    return actions, disconnected


def absorb_message(env: "RwppDuelEnv", message: dict[str, Any]) -> dict[str, Any]:
    """Fold one bridge reply into ``state``/``obs_state``/``last_tick``/``masks``.

    A ``done`` message ends the episode; whatever it carries is merged over the
    previous state so the returned observation/masks describe the real terminal
    state instead of the previous decision's state. Masks are rebuilt from the
    (possibly fogged) state the policy actually saw, including on the last step.
    """

    if message.get("type") == "done":
        env.done = True
        env.state = merge_done_message(env.state, message)
        state = env.state
        env.obs_state = env._observed(state)
        env.last_tick = int(state.get("tick", env.last_tick))
    else:
        state = observation_to_state(message) if message else dict(env.state)
        env.state = state
        env.obs_state = env._observed(state)
        env.last_tick = int(state["tick"])
    env.masks = build_masks(env.obs_state, env.micro)
    return state


def strategy_payload(env: "RwppDuelEnv") -> dict[str, Any] | None:
    """JSON-safe view of the current directive (``None`` when absent/broken).

    The *resolved* macro plan is attached next to the raw directive payload, and
    the two are deliberately different objects: ``macro`` in the directive is what
    the commander asked for (possibly empty), ``resolved_macro`` is what the
    executor actually ran with after falling back and clamping. Logging only the
    first would make "the commander said nothing" and "the commander said
    something the executor overrode" indistinguishable in ``live.jsonl``.
    """

    if env.directive is None:
        return None
    to_payload = getattr(env.directive, "to_payload", None)
    if not callable(to_payload):
        return None
    try:
        payload = to_payload()
    except Exception:  # noqa: BLE001 - never break the env on logging
        return None
    if isinstance(payload, dict):
        resolved = getattr(env, "macro", None)
        resolved_payload = getattr(resolved, "to_payload", None)
        if callable(resolved_payload):
            try:
                payload["resolved_macro"] = resolved_payload()
            except Exception:  # noqa: BLE001 - logging must not break the env
                pass
        economy = getattr(env, "macro_state", None)
        if isinstance(economy, dict):
            payload["economy"] = dict(economy)
    return payload


def build_step_info(env: "RwppDuelEnv", ctx: dict[str, Any]) -> dict[str, Any]:
    """Pack the ``info`` dict returned by ``step()``.

    ``ctx`` carries the per-step locals: ``actions``, ``bridge_result``,
    ``disconnected``, ``own_alive``, ``enemy_alive``, ``distance`` and
    ``components``.
    """

    own_frac, enemy_frac = env_fracs(env)
    return {
        "episode_id": env.episode_id,
        "tick": int(env.state.get("tick", 0)),
        "result": env.last_result,
        "bridge_result": ctx["bridge_result"],
        "disconnected": bool(ctx["disconnected"]),
        "receipts": env.receipts,
        "masks": env.masks,
        "raw_state": env.state,
        "visible_enemies": len(env.obs_state.get("visible_enemy_entities", [])),
        "actions": ctx["actions"],
        "own_alive": ctx["own_alive"],
        "enemy_alive": ctx["enemy_alive"],
        "distance": ctx["distance"],
        "reward_breakdown": dict(ctx["components"]),
        "reward_components": dict(env.episode_components),
        # Why orders were refused this decision. ``invalid`` is one reward key,
        # so without this histogram a growing penalty cannot be attributed.
        "rejection_reasons": receipt_reasons(ctx.get("receipts") or []),
        "strategy": strategy_payload(env),
        "eval": {"own_hp_frac": own_frac, "enemy_hp_frac": enemy_frac},
        "max_refs": {
            "own": float(env.own_max_ref),
            "enemy": float(env.enemy_max_ref),
        },
        "port": env.port,
    }


def env_fracs(env: "RwppDuelEnv") -> tuple[float, float]:
    """``(own_hp_frac, enemy_hp_frac)`` measured against the episode-start refs."""

    from .reward import hp_fractions

    return hp_fractions(env.state, env.own_max_ref, env.enemy_max_ref)


def receipt_reasons(receipts: list[dict[str, Any]]) -> dict[str, int]:
    """Histogram of rejection reasons over a list of receipts.

    ``invalid`` is a single reward key, so a growing penalty says nothing about
    *which* order was refused. Counting reasons here (and logging them) is what
    turned "invalid went up" into an actual diagnosis. ``accepted`` receipts are
    skipped; a receipt with no reason is bucketed as ``UNKNOWN``.
    """

    counts: dict[str, int] = {}
    for receipt in receipts:
        if receipt.get("accepted", True):
            continue
        reason = str(receipt.get("rejected_reason") or "UNKNOWN")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


__all__ = [
    "absorb_message",
    "build_step_info",
    "env_fracs",
    "receipt_reasons",
    "send_step",
    "strategy_payload",
]
