# SPDX-License-Identifier: Apache-2.0
"""Vectorised real-game environments for PPO sampling.

Each slot is an independent RWPP headless process + bridge port from a port
pool. ``reset`` and ``step`` fan out over a thread pool because both operations
are socket-bound (the game process itself runs single-threaded), and finished
episodes are restarted in parallel to hide the multi-second JVM startup cost.

Restarts are *not* awaited inside ``step``: a finished slot is returned to PPO as
a terminal transition immediately and relaunched in the background, so the
multi-second JVM boot of one slot never stalls the other slots' sampling.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from ..micro_config import MicroDuelConfig
from ..rwpp_env import RwppDuelEnv, RwppGameConfig, _ProcLike
from .vec_env import StepBatch


class VecRwppEnv:
    """Synchronous vector env over real Rusted Warfare 1.15 instances."""

    def __init__(
        self,
        num_envs: int = 2,
        *,
        game: RwppGameConfig | None = None,
        port_base: int = 5140,
        base_seed: int = 0,
        micro_config: MicroDuelConfig | None = None,
        episode_prefix: str = "rwpp-train",
        launcher: Callable[[RwppGameConfig, int], _ProcLike] | None = None,
        max_workers: int | None = None,
        persistent: bool = True,
        strategy: Any | None = None,
        async_reset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.game = game or RwppGameConfig()
        self.config = micro_config or self.game.micro_config()
        self.num_envs = int(num_envs)
        self.port_base = int(port_base)
        # ``persistent`` keeps a running game across PPO updates instead of
        # restarting it on every rollout (JVM startup costs seconds per instance).
        self.persistent = bool(persistent)
        # ``async_reset`` relaunches a finished slot in the background instead of
        # blocking ``step()`` on its JVM boot. The slot reports a neutral
        # observation (with all-false masks, so the policy never acts on it) and
        # is skipped for one decision while it boots.
        self.async_reset = bool(async_reset)
        self._started = False
        self.envs = [
            RwppDuelEnv(
                self.game,
                port=self.port_base + index,
                seed=base_seed + index,
                micro_config=self.config,
                launcher=launcher,
                episode_prefix=episode_prefix,
                strategy=strategy,
            )
            for index in range(self.num_envs)
        ]
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or max(2, self.num_envs),
            thread_name_prefix="rwpp-env",
        )
        # Background restarts get their *own* pool. Sharing one pool meant a
        # slot relaunching its JVM (seconds of blocking boot) consumed a worker
        # that ``step`` needed for a live slot, so the batch stalled on an
        # unrelated boot. Two pools keep stepping and booting independent.
        self._restart_pool = ThreadPoolExecutor(
            max_workers=max(1, self.num_envs),
            thread_name_prefix="rwpp-restart",
        )
        self.observations = np.zeros((self.num_envs, self.config.flat_obs_dim), dtype=np.float32)
        self.masks: list[dict[str, np.ndarray]] = [{} for _ in range(self.num_envs)]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []
        # Slots whose JVM is currently booting in the background. Guarded by
        # ``_pending_lock`` because the futures complete on pool threads.
        self._pending: dict[int, Future] = {}
        self._pending_lock = threading.Lock()
        # Per-slot rejection histogram. ``invalid`` is a single reward key, so a
        # growing penalty cannot be attributed without knowing *why* orders were
        # refused; this accumulates the reasons for the running episode.
        self._rejection_counts: list[dict[str, int]] = [
            {} for _ in range(self.num_envs)
        ]

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        if self.persistent and self._started:
            # Reuse running games: mid-episode state must survive PPO update
            # boundaries, otherwise every rollout would waste a fresh JVM.
            self._adopt_finished_restarts()
            return self.observations.copy(), self.masks, [{"reused": True} for _ in range(self.num_envs)]
        # Every boot happens at once: one JVM per slot, all in flight together.
        results = list(self._pool.map(lambda env: env.reset(), self.envs))
        infos: list[dict[str, Any]] = []
        for index, (obs, info) in enumerate(results):
            self.observations[index] = obs
            self.masks[index] = info["masks"]
            self.episode_returns[index] = 0.0
            self.episode_lengths[index] = 0
            infos.append(info)
        self._started = True
        return self.observations.copy(), self.masks, infos

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        expected = (self.num_envs, len(self.config.action_nvec))
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
        self._adopt_finished_restarts()
        active = self._active_indices()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        # One info dict per slot. A booting slot reports a zero-reward self-loop:
        # its observation/masks are left untouched, so the action the policy just
        # sampled for it stays the "next" action and the transition is a no-op
        # rather than a fabricated jump to an unrelated state.
        infos: list[dict[str, Any]] = [
            {"reset": None, "episode": None, "booting": index not in active}
            for index in range(self.num_envs)
        ]
        results: list[tuple[int, Any]] = []
        if active:
            pairs = [(index, self.envs[index], actions[index]) for index in active]
            stepped = list(
                self._pool.map(lambda pair: pair[1].step(pair[2]), pairs)
            )
            results = [(pair[0], item) for pair, item in zip(pairs, stepped)]
        finished: list[int] = []
        for index, (obs, reward, term, trunc, info) in results:
            self.episode_returns[index] += reward
            self.episode_lengths[index] += 1
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            infos[index] = info
            for reason, count in (info.get("rejection_reasons") or {}).items():
                bucket = self._rejection_counts[index]
                bucket[reason] = bucket.get(reason, 0) + int(count)
            if term or trunc:
                finished.append(index)
        if finished:
            self._retire_finished(finished, infos)
        for index, item in results:
            if index in finished:
                continue
            obs, _reward, _term, _trunc, info = item
            self.observations[index] = obs
            self.masks[index] = info["masks"]
        return StepBatch(
            observations=self.observations.copy(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def _active_indices(self) -> list[int]:
        """Slots that are not currently booting in the background."""

        with self._pending_lock:
            return [index for index in range(self.num_envs) if index not in self._pending]

    def _retire_finished(self, finished: list[int], infos: list[dict[str, Any]]) -> None:
        """Record the terminal episode and relaunch the slot in the background.

        The terminal transition is fully described by the data already collected
        (return / result / components), so nothing about the finished episode is
        lost by not awaiting its successor. The slot keeps its last observation
        and masks; while the background reset is in flight ``step`` turns it into
        a zero-reward self-loop (see :meth:`step`), so PPO never sees a synthetic
        observation for a slot that is not actually running.
        """

        for index in finished:
            info = infos[index]
            episode = {
                "return": float(self.episode_returns[index]),
                "decisions": int(self.episode_lengths[index]),
                "result": info.get("result"),
                "tick": int(info.get("tick", 0)),
                "port": int(info.get("port", 0)),
                "components": dict(info.get("reward_components") or {}),
                # Why the episode paid ``invalid``: {reason: count}.
                "rejections": dict(self._rejection_counts[index]),
                "eval": None,
            }
            self.completed_episodes.append(episode)
            infos[index]["episode"] = episode
            self.episode_returns[index] = 0.0
            self.episode_lengths[index] = 0
            self._rejection_counts[index] = {}
            if not self.async_reset:
                obs, reset_info = self.envs[index].reset()
                self.masks[index] = reset_info["masks"]
                self.observations[index] = obs
                infos[index]["reset"] = reset_info
                continue
            with self._pending_lock:
                self._pending[index] = self._restart_pool.submit(self._restart, index)

    def _restart(self, index: int) -> tuple[int, np.ndarray, dict[str, Any]]:
        """Background body of one slot restart (runs on a pool thread)."""

        obs, info = self.envs[index].reset()
        return index, obs, info

    def _adopt_finished_restarts(self) -> None:
        """Fold in any slot whose background JVM boot has completed."""

        with self._pending_lock:
            pending = dict(self._pending)
        for index, future in pending.items():
            if not future.done():
                continue
            with self._pending_lock:
                self._pending.pop(index, None)
            # ``Future.result`` re-raises a failure from the reset, which must
            # not kill the whole training run: the slot simply stays inactive
            # and is retried on the next boundary.
            try:
                _index, obs, info = future.result()
            except Exception:  # noqa: BLE001
                with self._pending_lock:
                    self._pending[index] = self._restart_pool.submit(self._restart, index)
                continue
            self.observations[index] = obs
            self.masks[index] = info["masks"]

    def drain_completed(self) -> list[dict[str, Any]]:
        completed, self.completed_episodes = self.completed_episodes, []
        return completed

    def close(self) -> None:
        # Settle in-flight restarts first: each one owns a game process, and
        # shutting the pool down under it would leave the JVM running detached.
        try:
            with self._pending_lock:
                pending = list(self._pending.values())
                self._pending.clear()
            for future in pending:
                try:
                    future.result(timeout=60.0)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        try:
            list(self._pool.map(lambda env: env.close(), self.envs))
        finally:
            self._started = False
            self._restart_pool.shutdown(wait=False, cancel_futures=True)
            self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = ["VecRwppEnv"]
