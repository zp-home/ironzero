# SPDX-License-Identifier: Apache-2.0
"""In-process vectorized micro-duel environments for PPO sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig


@dataclass(slots=True)
class StepBatch:
    observations: np.ndarray  # (B, obs_dim) float32
    rewards: np.ndarray  # (B,) float32
    terminated: np.ndarray  # (B,) bool
    truncated: np.ndarray  # (B,) bool
    infos: list[dict[str, Any]]


class VecMicroEnv:
    """A tiny synchronous vector env; deterministic and dependency-free."""

    def __init__(
        self,
        num_envs: int = 8,
        *,
        config: MicroDuelConfig | None = None,
        opponent: str = "scripted",
        base_seed: int = 0,
        episode_prefix: str = "train",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.config = config or DEFAULT_CONFIG
        self.num_envs = num_envs
        self.envs = [
            MicroDuelEnv(
                config=self.config,
                opponent=opponent,
                seed=base_seed + index,
                episode_prefix=episode_prefix,
            )
            for index in range(num_envs)
        ]
        self.observations = np.zeros((num_envs, self.config.flat_obs_dim), dtype=np.float32)
        self.masks: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        infos = []
        for index, env in enumerate(self.envs):
            obs, info = env.reset()
            self.observations[index] = obs
            self.masks[index] = info["masks"]
            self.episode_returns[index] = 0.0
            self.episode_lengths[index] = 0
            infos.append(info)
        return self.observations.copy(), self.masks, infos

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs, len(self.config.action_nvec)):
            raise ValueError(
                f"actions must have shape {(self.num_envs, len(self.config.action_nvec))}, got {actions.shape}"
            )
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for index, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(actions[index])
            self.episode_returns[index] += reward
            self.episode_lengths[index] += 1
            rewards[index] = reward
            terminated[index] = term
            truncated[index] = trunc
            done = term or trunc
            if done:
                info["episode"] = {
                    "return": float(self.episode_returns[index]),
                    "decisions": int(self.episode_lengths[index]),
                    "result": info.get("result"),
                    "eval": info.get("eval"),
                }
                self.completed_episodes.append(info["episode"])
                obs, reset_info = env.reset()
                self.episode_returns[index] = 0.0
                self.episode_lengths[index] = 0
                self.masks[index] = reset_info["masks"]
                info["reset"] = reset_info
            else:
                self.masks[index] = info["masks"]
            self.observations[index] = obs
            infos.append(info)
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
        for env in self.envs:
            env.close()


__all__ = ["VecMicroEnv", "StepBatch"]
