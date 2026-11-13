# SPDX-License-Identifier: Apache-2.0
"""Single real-game environment driven through the IronZero bridge.

The game itself is not resettable in-process, so ``reset()`` starts a fresh
headless process. That costs a few seconds of JVM startup, which is why
:class:`ironzero.training.vec_rwpp.VecRwppEnv` resets several instances in
parallel and keeps them running across PPO rollouts.

This module only orchestrates: process launch/pacing lives in
:mod:`ironzero.rwpp.process`, socket framing in :mod:`ironzero.rwpp.transport`,
message/action conversion in :mod:`ironzero.rwpp.decoding`, per-decision
plumbing (sending the step, absorbing the reply, packing ``info``) in
:mod:`ironzero.rwpp.stepping`, fog of war in :mod:`ironzero.rwpp.vision`,
potential terms in :mod:`ironzero.rwpp.reward` and the reward decomposition in
:mod:`ironzero.rwpp.reward_shaping`.
"""

from __future__ import annotations

import socket
from typing import Any, Callable

import numpy as np

from ..micro_config import MicroDuelConfig
from ..observations import build_masks, encode_observation
from .config import RwppGameConfig
from .decoding import observation_to_state
from .errors import RwppEnvError
from .process import _ProcLike, launch_rwpp_process
from .reward import army_distance_norm, army_hp_reference, combat_value_potential
from .reward_shaping import compute_step_shaping
from .session import await_observation, drain_verdict, shutdown_game, start_game
from .spaces import _BoxSpace, _MultiDiscreteSpace
from .stepping import absorb_message, build_step_info, send_step
from .strategy import refresh_strategy, strategy_potential, strategy_summary
from .transport import _LineReader
from .vision import apply_vision


class RwppDuelEnv:
    """Single real-game environment driven through the IronZero bridge."""

    metadata = {"render_modes": [], "name": "IronZeroRwppDuel-v0"}

    def __init__(
        self,
        game: RwppGameConfig | None = None,
        *,
        port: int = 5140,
        seed: int = 0,
        micro_config: MicroDuelConfig | None = None,
        launcher: Callable[[RwppGameConfig, int], _ProcLike] | None = None,
        episode_prefix: str = "rwpp",
        strategy: Any | None = None,
    ) -> None:
        self.game = game or RwppGameConfig()
        self.port = int(port)
        self.seed = int(seed)
        self.micro = micro_config or self.game.micro_config()
        self._launcher = launcher or launch_rwpp_process
        self.episode_prefix = episode_prefix
        self.strategy = strategy
        self.directive: Any | None = None
        self.last_strategy_potential: float | None = None

        self._process: _ProcLike | None = None
        self._socket: socket.socket | None = None
        self._reader: _LineReader | None = None
        self._log_handle = None
        self._episode_counter = 0
        self.episode_id = ""
        self.state: dict[str, Any] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.last_tick = 0
        self.last_potential = 0.0
        self.last_distance = 0.0
        # Previous economy rate, so the ``income`` reward term can pay for the
        # *growth* of income (a new extractor / an upgrade) instead of paying
        # the absolute rate on every one of thousands of decisions.
        self.last_income_rate = 0.0
        # Previous own-building count: the build reward is paid for a *finished*
        # structure, never for an accepted order (which a policy can repeat
        # every decision to farm reward).
        self.last_own_buildings = 0
        # Previous own *combat* unit count (buildings and builders excluded), so
        # ``bonus_produce`` can pay for a unit actually appearing instead of for
        # a queued order.
        self.last_own_combat = 0
        self.previous_own_alive = 0
        self.previous_enemy_alive = 0
        self.episode_components: dict[str, float] = {}
        self.receipts: list[dict[str, Any]] = []
        self.done = False
        self.last_result: str | None = None
        self.episode_steps = 0
        self.observation_space = _BoxSpace((self.micro.flat_obs_dim,))
        self.action_space = _MultiDiscreteSpace(self.micro.action_nvec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed = int(seed)
        self._episode_counter += 1
        self.episode_id = f"{self.episode_prefix}-{self.seed}-{self._episode_counter}"
        self._shutdown_game()
        self._start_game()
        message = self._await_observation(self.game.start_timeout + self.game.obs_timeout)
        if message is None:
            raise RwppEnvError(f"no observation from real game on port {self.port}")
        self.state = observation_to_state(message)
        # ``state`` stays the ground truth (used for termination / hp / reward);
        # ``obs_state`` is what the policy is allowed to see when fog is enabled.
        self.obs_state = self._observed(self.state)
        self.last_tick = int(self.state["tick"])
        self.masks = build_masks(self.obs_state, self.micro)
        # Reference totals: dead units vanish from the bridge stream, so reward and
        # evaluation fractions are measured against the episode-start army value.
        # Computed from ``self.state`` (ground truth: the bridge streams every
        # unit), NOT from ``self.obs_state`` (fogged policy view), otherwise a
        # fogged episode would normalise enemy hp against the few visible units.
        self.own_max_ref, self.enemy_max_ref = army_hp_reference(self.state, self.micro)
        # ``potential`` rewards the *combat value* difference, not the raw hp
        # difference: units fight at full strength until destroyed, so chip
        # damage across many units must not outscore a kill (see
        # ``reward.combat_value``).
        self.last_potential = combat_value_potential(
            self.state, self.own_max_ref, self.enemy_max_ref
        )
        self.last_distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        self.last_income_rate = float(
            (self.state.get("global_state") or {}).get("income") or 0.0
        )
        self.last_own_buildings = sum(
            1 for unit in self.state.get("own_entities", []) if unit.get("is_building")
        )
        self.last_own_combat = sum(
            1
            for unit in self.state.get("own_entities", [])
            if not unit.get("is_building") and not unit.get("is_builder")
        )
        self.previous_own_alive = sum(
            1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        self.previous_enemy_alive = sum(
            1
            for unit in self.state.get("visible_enemy_entities", [])
            if float(unit.get("hp") or 0.0) > 0.0
        )
        self.episode_components: dict[str, float] = {}
        self.receipts = []
        self.done = False
        self.last_result = None
        self.episode_steps = 0
        self._refresh_strategy()
        self.last_strategy_potential = self._strategy_potential()
        obs = encode_observation(self.obs_state, self.micro)
        info = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "masks": self.masks,
            "raw_state": self.state,
            "visible_enemies": len(self.obs_state.get("visible_enemy_entities", [])),
            "tick": self.state["tick"],
            "max_refs": {
                "own": float(self.own_max_ref),
                "enemy": float(self.enemy_max_ref),
            },
            "port": self.port,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._socket is None or self._reader is None:
            raise RwppEnvError("reset() must be called before step()")
        self.episode_steps += 1
        previous_potential = float(self.last_potential)
        previous_distance = float(self.last_distance)
        previous_income_rate = float(self.last_income_rate)
        previous_own_buildings = int(self.last_own_buildings)
        previous_own_combat = int(self.last_own_combat)
        previous_own_alive = int(self.previous_own_alive)
        previous_enemy_alive = int(self.previous_enemy_alive)
        receipts_before = len(self.receipts)
        # Decode against what the policy actually saw: with fog enabled the
        # target slots index the *visible* enemy list (as masked in obs_state),
        # so using the ground-truth list here would fire at the wrong enemy.
        actions, disconnected = send_step(self, action)

        # Always read, even when the send failed.
        #
        # The bridge is a free-running simulator: it advances on ``--frame-ms``
        # and emits an observation every ``DECISION_INTERVAL`` ticks without
        # waiting for a client ``step``. When the engine decides the match it
        # writes ``done`` and closes the socket, so our next ``sendall`` fails
        # with ``ConnectionResetError(10054)`` and the kernel *discards the
        # unread receive queue* -- verdict included.
        #
        # The old code turned that into ``message = None if disconnected else ...``:
        # the read was skipped entirely, the episode was booked as a timeout with
        # ``terminal = 0.0``, and the trainer reported ``win_rate = 0.0`` for all
        # 40 updates of a run whose bridge logs showed 38 victories and 5 defeats.
        #
        # ``_await_observation`` now also drains whatever already arrived and
        # prefers a decisive ``result``/``done`` over a stale frame, so the
        # verdict survives. After a real drop nothing is buffered and the result
        # is still an honest ``timeout`` (``_drain_verdict`` never invents one).
        message = self._await_observation(self.game.obs_timeout)
        if (message is None or not message.get("result")) and disconnected:
            # The step budget may have been eaten by the send failure + catch-up;
            # give the verdict one more bounded chance to surface.
            message = self._drain_verdict() or message
        receipts_this_step = self.receipts[receipts_before:]
        if message is None:
            disconnected = True
            message = {}
        # The bridge reports the native game result on the final observation and
        # on the ``done`` message; it is authoritative when present.
        bridge_result = message.get("result")
        state = absorb_message(self, message)
        if bridge_result is None:
            bridge_result = state.get("result")

        distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        potential = combat_value_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        own_alive = sum(1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0)
        enemy_alive = sum(
            1 for unit in self.state.get("visible_enemy_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        previous_strategy = self.last_strategy_potential
        self._refresh_strategy()
        strategy_now = self._strategy_potential()
        strategy_component = (
            float(self.game.strategy_coef) * (strategy_now - previous_strategy)
            if strategy_now is not None and previous_strategy is not None
            else 0.0
        )
        self.last_strategy_potential = strategy_now
        # ---- reward decomposition (docs 7.2/7.4) --------------------------
        shaping = compute_step_shaping(
            micro=self.micro,
            game=self.game,
            state=self.state,
            actions=actions,
            receipts=receipts_this_step,
            message=message,
            bridge_result=bridge_result,
            potential=potential,
            previous_potential=previous_potential,
            distance=distance,
            previous_distance=previous_distance,
            own_alive=own_alive,
            enemy_alive=enemy_alive,
            previous_own_alive=previous_own_alive,
            previous_enemy_alive=previous_enemy_alive,
            strategy_component=strategy_component,
            disconnected=disconnected,
            done=self.done,
            own_max_ref=self.own_max_ref,
            enemy_max_ref=self.enemy_max_ref,
            episode_components=self.episode_components,
            previous_income_rate=previous_income_rate,
            previous_own_buildings=previous_own_buildings,
            previous_own_combat=previous_own_combat,
        )
        components = shaping.components
        reward = shaping.reward
        terminated = shaping.terminated
        truncated = shaping.truncated
        if shaping.last_result is not None:
            self.last_result = shaping.last_result
        self.last_potential = potential
        self.last_distance = distance
        # A terminal ``done`` message has no ``global_state``; keep the last
        # known rate instead of resetting it to 0, otherwise the next step books
        # a phantom -rate "economy collapse".
        state_global = self.state.get("global_state") or {}
        if "income" in state_global:
            self.last_income_rate = float(state_global.get("income") or 0.0)
        self.last_own_buildings = sum(
            1 for unit in self.state.get("own_entities", []) if unit.get("is_building")
        )
        self.last_own_combat = sum(
            1
            for unit in self.state.get("own_entities", [])
            if not unit.get("is_building") and not unit.get("is_builder")
        )
        self.previous_own_alive = own_alive
        self.previous_enemy_alive = enemy_alive
        if terminated or truncated:
            self._shutdown_game()

        obs = encode_observation(self.obs_state, self.micro)
        info = build_step_info(
            self,
            {
                "actions": actions,
                "bridge_result": bridge_result,
                "disconnected": disconnected,
                "own_alive": own_alive,
                "enemy_alive": enemy_alive,
                "distance": distance,
                "components": components,
                # This decision's receipts only. The rejection histogram in
                # ``build_step_info`` reads it, and it is the only way to tell
                # *why* ``invalid`` was paid (see ``reward_shaping``).
                "receipts": receipts_this_step,
            },
        )
        return obs, reward, terminated, truncated, info

    def _observed(self, state: dict[str, Any]) -> dict[str, Any]:
        """State as the policy sees it (fog of war applied when configured)."""

        radius = float(self.game.vision_radius or 0.0)
        if radius <= 0.0:
            return state
        return apply_vision(dict(state), radius)

    # ------------------------------------------------------------------
    # Strategic layer (LLM or rule based; reward shaping only)
    # ------------------------------------------------------------------
    # The implementation lives in :mod:`ironzero.rwpp.strategy`; these methods
    # stay as delegates so the historical (private) names keep working.
    def _strategy_summary(self) -> dict[str, Any]:
        return strategy_summary(self)

    def _refresh_strategy(self) -> None:
        refresh_strategy(self)

    def _strategy_potential(self) -> float | None:
        return strategy_potential(self)

    def action_masks(self) -> dict[str, np.ndarray]:
        return self.masks

    def close(self) -> None:
        self._shutdown_game()

    def __enter__(self) -> "RwppDuelEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Process / bridge plumbing
    # ------------------------------------------------------------------
    def _start_game(self) -> None:
        start_game(self)

    def _await_observation(self, timeout: float) -> dict[str, Any] | None:
        return await_observation(self, timeout)

    def _drain_verdict(self) -> dict[str, Any] | None:
        return drain_verdict(self)

    def _shutdown_game(self) -> None:
        shutdown_game(self)


__all__ = ["RwppDuelEnv"]
