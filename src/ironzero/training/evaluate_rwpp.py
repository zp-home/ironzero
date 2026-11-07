# SPDX-License-Identifier: Apache-2.0
"""Evaluation of a policy inside real Rusted Warfare 1.15 games."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..micro_config import MicroDuelConfig
from ..rwpp_env import RwppDuelEnv, RwppGameConfig
from .evaluate import summarize
from .policy import masks_to_tensors


def evaluate_rwpp_policy(
    policy: torch.nn.Module,
    *,
    game: RwppGameConfig,
    episodes: int = 2,
    micro_config: MicroDuelConfig | None = None,
    port_base: int = 5200,
    seed: int = 70_000,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run ``episodes`` full real-game episodes with the policy in the loop."""

    micro = micro_config or game.micro_config()
    device = torch.device(device)
    policy.eval()
    results: list[str] = []
    returns: list[float] = []
    own_hp: list[float] = []
    enemy_hp: list[float] = []
    decisions: list[int] = []
    env = RwppDuelEnv(
        game,
        port=port_base,
        seed=seed,
        micro_config=micro,
        episode_prefix="rwpp-eval",
    )
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            total = 0.0
            steps = 0
            done = False
            final = info
            while not done:
                mask_tensors = masks_to_tensors([info["masks"]], device)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    actions, _, _ = policy.act(obs_tensor, mask_tensors, deterministic=deterministic)
                action = actions[0].cpu().numpy()
                obs, reward, terminated, truncated, final = env.step(action)
                total += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            evaluation = final.get("eval") or {}
            results.append(str(final.get("result")))
            returns.append(total)
            own_hp.append(float(evaluation.get("own_hp_frac", 0.0)))
            enemy_hp.append(float(evaluation.get("enemy_hp_frac", 0.0)))
            decisions.append(steps)
    finally:
        env.close()
    evaluation = summarize(results, returns, own_hp, enemy_hp, decisions)
    evaluation["source"] = "rwpp"
    return evaluation


__all__ = ["evaluate_rwpp_policy"]
