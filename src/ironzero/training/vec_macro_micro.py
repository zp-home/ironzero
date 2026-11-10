# SPDX-License-Identifier: Apache-2.0
"""Hierarchical vector environment combining Macro Controller with Micro PPO.

In this environment:
- The MacroController autonomously manages economy (extractors, factories, production).
- Micro PPO policy controls combat groups (move, attack, focus-fire).
- Together they enable end-to-end training and inference in an RTS setting with economy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..strategy.macro_controller import MacroController
from .vec_env import StepBatch


class VecMacroMicroEnv:
    """Synchronous vectorized environment executing Macro decisions + Micro PPO actions."""

    def __init__(
        self,
        num_envs: int = 4,
        *,
        config: MicroDuelConfig | None = None,
        opponent: str = "scripted",
        base_seed: int = 0,
        episode_prefix: str = "macro_train",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        # 激活经济与建造
        self.config = config or MicroDuelConfig(
            enable_economy=True,
            starting_funds=1500.0,
            base_income_per_second=4.0,
            extractor_income_per_second=10.0,
            extractor_t2_income_per_second=22.0,
            extractor_cost=700.0,
            factory_cost=800.0,
            tank_cost=250.0,
        )
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
        self.macro_controllers = [
            MacroController(
                team=0,
                extractor_cost=self.config.extractor_cost,
                extractor_upgrade_cost=self.config.extractor_upgrade_cost,
                factory_cost=self.config.factory_cost,
                tank_cost=self.config.tank_cost,
            )
            for _ in range(num_envs)
        ]
        self.observations = np.zeros((num_envs, self.config.flat_obs_dim), dtype=np.float32)
        self.masks: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []

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

    def step(self, micro_actions: np.ndarray) -> StepBatch:
        micro_actions = np.asarray(micro_actions, dtype=np.int64)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for index, env in enumerate(self.envs):
            raw_obs = env.state
            # 1. 宏观控制器产生经济与建造动作
            macro_actions = self.macro_controllers[index].plan_macro_actions(raw_obs)
            # 2. PPO 解码战术微操动作
            tactical_actions = env.decode_actions(micro_actions[index])
            # 3. 合并动作序列下发给引擎
            all_actions = macro_actions + tactical_actions

            obs, reward, term, trunc, info = env.step_actions(all_actions)
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
                    "components": dict(info.get("reward_components") or {}),
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
        batch = list(self.completed_episodes)
        self.completed_episodes.clear()
        return batch

    def close(self) -> None:
        for env in self.envs:
            env.close()
