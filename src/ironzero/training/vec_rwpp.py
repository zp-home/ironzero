# SPDX-License-Identifier: Apache-2.0
"""Vectorised real-game environments for PPO sampling.

Each slot is an independent RWPP headless process + bridge port from a port
pool. ``reset`` and ``step`` fan out over a thread pool because both operations
are socket-bound (the game process itself runs single-threaded), and finished
episodes are restarted in parallel to hide the multi-second JVM startup cost.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
        self._started = False
        self.envs = [
            RwppDuelEnv(
                self.game,
                port=self.port_base + index,
                seed=base_seed + index,
                micro_config=self.config,
                launcher=launcher,
                episode_prefix=episode_prefix,
            )
            for index in range(self.num_envs)
        ]
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or max(2, self.num_envs),
            thread_name_prefix="rwpp-env",
        )
        self.observations = np.zeros((self.num_envs, self.config.flat_obs_dim), dtype=np.float32)
        self.masks: list[dict[str, np.ndarray]] = [{} for _ in range(self.num_envs)]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        if self.persistent and self._started:
            # Reuse running games: mid-episode state must survive PPO update
            # boundaries, otherwise every rollout would waste a fresh JVM.
            return self.observations.copy(), self.masks, [{"reused": True} for _ in range(self.num_envs)]
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
        results = list(
            self._pool.map(lambda pair: pair[0].step(pair[1]), zip(self.envs, actions))
        )
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        finished = [index for index, item in enumerate(results) if item[2] or item[3]]
        for index, (obs, reward, term, trunc, info) in enumerate(results):
            self.episode_returns[index] += reward
            self.episode_lengths[index] += 1
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            infos.append(info)
        if finished:
            # Restart finished episodes in parallel so JVM startup overlaps.
            restarted = list(
                self._pool.map(
                    lambda index: self.envs[index].reset(),
                    finished,
                )
            )
            for index, (obs, info) in zip(finished, restarted):
                episode = {
                    "return": float(self.episode_returns[index]),
                    "decisions": int(self.episode_lengths[index]),
                    "result": infos[index].get("result"),
                    "tick": int(infos[index].get("tick", 0)),
                    "port": int(infos[index].get("port", 0)),
                    "eval": None,
                }
                self.completed_episodes.append(episode)
                self.episode_returns[index] = 0.0
                self.episode_lengths[index] = 0
                self.masks[index] = info["masks"]
                self.observations[index] = obs
                infos[index]["reset"] = info
                infos[index]["episode"] = episode
        for index, item in enumerate(results):
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

    def drain_completed(self) -> list[dict[str, Any]]:
        completed, self.completed_episodes = self.completed_episodes, []
        return completed

    def close(self) -> None:
        try:
            list(self._pool.map(lambda env: env.close(), self.envs))
        finally:
            self._started = False
            self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = ["VecRwppEnv"]
