# SPDX-License-Identifier: Apache-2.0
"""Evaluation helpers: baseline and policy rollouts on the micro-duel task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .policy import masks_to_tensors


def sample_masked_action(masks: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly random but legal action tuple."""

    def pick(key: str) -> int:
        valid = np.flatnonzero(masks[key])
        if len(valid) == 0:
            return 0
        return int(rng.choice(valid))

    return np.array(
        [pick("kind"), pick("group"), pick("x"), pick("y"), pick("target")], dtype=np.int64
    )


def evaluate_policy(
    policy: torch.nn.Module,
    *,
    episodes: int = 24,
    config: MicroDuelConfig | None = None,
    opponent: str = "scripted",
    seed: int = 10_000,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    device = torch.device(device)
    policy.eval()
    results: list[str] = []
    returns: list[float] = []
    own_hp: list[float] = []
    enemy_hp: list[float] = []
    decisions: list[int] = []
    for episode in range(episodes):
        env = MicroDuelEnv(config=cfg, opponent=opponent, seed=seed + episode, episode_prefix="eval")
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False
        while not done:
            mask_tensors = masks_to_tensors([info["masks"]], device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            actions, _, _ = policy.act(obs_tensor, mask_tensors, deterministic=deterministic)
            action = actions[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        evaluation = info.get("eval") or {}
        results.append(str(info.get("result")))
        returns.append(float(total))
        own_hp.append(float(evaluation.get("own_hp_frac", 0.0)))
        enemy_hp.append(float(evaluation.get("enemy_hp_frac", 0.0)))
        decisions.append(steps)
        env.close()
    return summarize(results, returns, own_hp, enemy_hp, decisions)


def evaluate_random(
    *,
    episodes: int = 24,
    config: MicroDuelConfig | None = None,
    opponent: str = "scripted",
    seed: int = 20_000,
) -> dict[str, Any]:
    cfg = config or DEFAULT_CONFIG
    rng = np.random.default_rng(seed)
    results: list[str] = []
    returns: list[float] = []
    own_hp: list[float] = []
    enemy_hp: list[float] = []
    decisions: list[int] = []
    for episode in range(episodes):
        env = MicroDuelEnv(config=cfg, opponent=opponent, seed=seed + episode, episode_prefix="random")
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = sample_masked_action(info["masks"], rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        evaluation = info.get("eval") or {}
        results.append(str(info.get("result")))
        returns.append(float(total))
        own_hp.append(float(evaluation.get("own_hp_frac", 0.0)))
        enemy_hp.append(float(evaluation.get("enemy_hp_frac", 0.0)))
        decisions.append(steps)
        env.close()
    return summarize(results, returns, own_hp, enemy_hp, decisions)


def summarize(
    results: list[str],
    returns: list[float],
    own_hp: list[float],
    enemy_hp: list[float],
    decisions: list[int],
) -> dict[str, Any]:
    count = max(1, len(results))
    return {
        "episodes": len(results),
        "win_rate": sum(1 for r in results if r == "victory") / count,
        "loss_rate": sum(1 for r in results if r == "defeat") / count,
        "timeout_rate": sum(1 for r in results if r == "timeout") / count,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_own_hp_frac": float(np.mean(own_hp)) if own_hp else 0.0,
        "mean_enemy_hp_frac": float(np.mean(enemy_hp)) if enemy_hp else 0.0,
        "mean_decisions": float(np.mean(decisions)) if decisions else 0.0,
    }


__all__ = ["sample_masked_action", "evaluate_policy", "evaluate_random", "summarize"]
