# SPDX-License-Identifier: Apache-2.0
"""Independent inference and evaluation runner for hierarchical RTS (Macro + Micro PPO)."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from ..strategy.macro_controller import MacroController
from .evaluate import summarize
from .policy import masks_to_tensors
from .ppo import load_policy


def evaluate_macro_micro_policy(
    policy: torch.nn.Module,
    *,
    episodes: int = 5,
    config: MicroDuelConfig | None = None,
    opponent: str = "scripted",
    seed: int = 10_000,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run full evaluation with MacroController handling economy and policy handling micro."""
    cfg = config or MicroDuelConfig(
        enable_economy=True,
        starting_funds=1500.0,
        base_income_per_second=5.0,
        extractor_income_per_second=10.0,
        extractor_t2_income_per_second=22.0,
        extractor_cost=700.0,
        factory_cost=800.0,
        tank_cost=250.0,
    )
    device = torch.device(device)
    policy.eval()

    results: list[str] = []
    returns: list[float] = []
    own_hp: list[float] = []
    enemy_hp: list[float] = []
    decisions: list[int] = []

    for episode in range(episodes):
        env = MicroDuelEnv(config=cfg, opponent=opponent, seed=seed + episode, episode_prefix="macro_eval")
        macro = MacroController(
            team=0,
            extractor_cost=cfg.extractor_cost,
            extractor_upgrade_cost=cfg.extractor_upgrade_cost,
            factory_cost=cfg.factory_cost,
            tank_cost=cfg.tank_cost,
        )
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False

        while not done:
            # 1. 宏观决策
            macro_actions = macro.plan_macro_actions(env.state)
            # 2. 微观神经网络推理决策
            mask_tensors = masks_to_tensors([info["masks"]], device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                actions, _, _ = policy.act(obs_tensor, mask_tensors, deterministic=deterministic)
            tactical_actions = env.decode_actions(actions[0].cpu().numpy())

            # 3. 合并并在环境中执行一步
            all_actions = macro_actions + tactical_actions
            obs, reward, terminated, truncated, info = env.step_actions(all_actions)
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

    summary = summarize(results, returns, own_hp, enemy_hp, decisions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference & evaluate Macro+Micro RTS policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    args = parser.parse_args()

    policy, _ = load_policy(args.checkpoint, device="cpu")
    summary = evaluate_macro_micro_policy(policy, episodes=args.episodes)
    print("Evaluation Result:", summary)


if __name__ == "__main__":
    main()
