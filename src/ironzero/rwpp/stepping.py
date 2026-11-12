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

from ..observations import build_masks
from .decoding import decode_action, merge_done_message, observation_to_state

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RwppDuelEnv


def send_step(env: "RwppDuelEnv", action: np.ndarray) -> tuple[list[dict[str, Any]], bool]:
    """Decode ``action``, send it to the bridge; return ``(actions, disconnected)``.

    Decoding happens against ``env.obs_state`` on purpose: with fog enabled the
    target slots index the *visible* enemy list (as masked in ``obs_state``), so
    using the ground-truth list here would fire at the wrong enemy.
    """

    actions = decode_action(action, env.obs_state, env.micro)
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
    """JSON-safe view of the current directive (``None`` when absent/broken)."""

    if env.directive is None:
        return None
    to_payload = getattr(env.directive, "to_payload", None)
    if not callable(to_payload):
        return None
    try:
        return to_payload()
    except Exception:  # noqa: BLE001 - never break the env on logging
        return None


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
