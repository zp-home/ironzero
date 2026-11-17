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
    """Draw a uniformly random but legal action tuple.

    Single-slot layout only: the five keys below are the ``MASK_KEYS`` names of the
    5-head action space. A two-slot policy (``MicroDuelConfig.action_slots == 2``,
    10 heads keyed ``econ_*``/``tac_*``) must **not** call this -- it would raise
    ``KeyError: 'kind'``. Nothing on the two-slot path uses it: this is the offline
    random *baseline* for the mock micro-duel engine, and
    ``RwppPPOTrainer.baseline()`` returns ``{}`` (a random baseline is meaningless
    on the real game), so the real-game trainer never reaches this function.
    """

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


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95% CI).

    Evaluation episodes are expensive on the real game (one fresh JVM each), so
    reports must show how much a win rate could move with a handful more games.
    The Wilson interval is well behaved for the tiny sample sizes used here
    (unlike the normal approximation, it does not collapse to zero width at
    ``p = 0`` or ``p = 1``).
    """

    total = int(total)
    if total <= 0:
        return (0.0, 1.0)
    successes = min(max(0, int(successes)), total)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return (max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width)))


def summarize(
    results: list[str],
    returns: list[float],
    own_hp: list[float],
    enemy_hp: list[float],
    decisions: list[int],
) -> dict[str, Any]:
    count = max(1, len(results))
    wins = sum(1 for r in results if r == "victory")
    win_low, win_high = wilson_interval(wins, len(results))
    loss_low, loss_high = wilson_interval(sum(1 for r in results if r == "defeat"), len(results))
    timeout_low, timeout_high = wilson_interval(
        sum(1 for r in results if r == "timeout"), len(results)
    )
    return {
        "episodes": len(results),
        "wins": wins,
        "losses": sum(1 for r in results if r == "defeat"),
        "timeouts": sum(1 for r in results if r == "timeout"),
        "win_rate": wins / count,
        "loss_rate": sum(1 for r in results if r == "defeat") / count,
        "timeout_rate": sum(1 for r in results if r == "timeout") / count,
        "win_rate_ci95": [win_low, win_high],
        "loss_rate_ci95": [loss_low, loss_high],
        "timeout_rate_ci95": [timeout_low, timeout_high],
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_own_hp_frac": float(np.mean(own_hp)) if own_hp else 0.0,
        "mean_enemy_hp_frac": float(np.mean(enemy_hp)) if enemy_hp else 0.0,
        "mean_decisions": float(np.mean(decisions)) if decisions else 0.0,
    }


def wilson_lower_for_target(target: float, episodes: int) -> float:
    """Smallest win count whose Wilson lower bound still clears ``target``.

    Used to size evaluation runs honestly. With the episode counts the platform
    actually used (``eval_episodes=4``) even a clean 4/4 sweep has a Wilson lower
    bound of only 0.510, so "win rate > 70%" could never be *demonstrated* no
    matter how good the policy was.

    Returns the required number of wins for ``episodes`` games, or ``episodes +
    1`` when the target is unreachable at that sample size.
    """

    episodes = max(1, int(episodes))
    for wins in range(0, episodes + 1):
        low, _ = wilson_interval(wins, episodes)
        if low >= float(target):
            return float(wins)
    return float(episodes + 1)


def minimum_episodes_for_target(target: float, max_episodes: int = 200) -> int:
    """Smallest ``episodes`` at which *all wins* would clear ``target``.

    Scans one episode at a time rather than a coarse candidate list: the
    threshold is not where a step-4 list happens to land. A clean sweep's Wilson
    lower bound is 0.7008 at 9 games and 0.6756 at 8, so the honest answer for a
    0.70 target is 9 -- a coarse list starting at 4/8/12 would report 12 and
    triples nothing but the evaluation bill.
    """

    target = float(target)
    for episodes in range(1, max(1, int(max_episodes)) + 1):
        low, _ = wilson_interval(episodes, episodes)
        if low >= target:
            return episodes
    return max(1, int(max_episodes))


__all__ = [
    "sample_masked_action",
    "evaluate_policy",
    "evaluate_random",
    "summarize",
    "wilson_interval",
    "wilson_lower_for_target",
    "minimum_episodes_for_target",
]
