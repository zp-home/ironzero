# SPDX-License-Identifier: Apache-2.0
"""Train end-to-end PPO policy under the macro-micro hierarchical environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ironzero.micro_config import MicroDuelConfig
from ironzero.training.ppo import PPOConfig, PPOTrainer
from ironzero.training.vec_macro_micro import VecMacroMicroEnv


def train_macro_micro(
    *,
    total_updates: int = 15,
    num_envs: int = 4,
    rollout_steps: int = 64,
    out_dir: str = "runs/macro_micro_ppo",
) -> PPOTrainer:
    micro_cfg = MicroDuelConfig(
        enable_economy=True,
        starting_funds=1500.0,
        base_income_per_second=5.0,
        extractor_income_per_second=10.0,
        extractor_cost=700.0,
        factory_cost=800.0,
        tank_cost=250.0,
        decision_interval=30,
    )
    vec = VecMacroMicroEnv(
        num_envs=num_envs,
        config=micro_cfg,
        opponent="scripted",
    )
    ppo_cfg = PPOConfig(
        total_updates=total_updates,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        epochs=3,
        minibatch_size=64,
        lr=3e-4,
        out_dir=out_dir,
        eval_every=5,
        eval_episodes=4,
    )
    trainer = PPOTrainer(
        config=ppo_cfg,
        micro_config=micro_cfg,
        vec=vec,
    )
    trainer.train()
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO in Macro-Micro RTS environment.")
    parser.add_argument("--updates", type=int, default=10, help="Total PPO updates")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--out-dir", type=str, default="runs/macro_micro_ppo", help="Output path")
    args = parser.parse_args()

    train_macro_micro(
        total_updates=args.updates,
        num_envs=args.envs,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
