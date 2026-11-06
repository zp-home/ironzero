# SPDX-License-Identifier: Apache-2.0
"""Command line entry point for PPO training on the micro-duel task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..micro_config import MicroDuelConfig
from .ppo import PPOConfig, PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IronZero PPO on the micro-duel reference task")
    parser.add_argument("--updates", type=int, default=120, help="number of PPO updates")
    parser.add_argument("--envs", type=int, default=8, help="parallel in-process environments")
    parser.add_argument("--rollout-steps", type=int, default=128, help="decisions per env per update")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "idle", "random"])
    parser.add_argument("--out", default="runs/micro_duel", help="output directory")
    parser.add_argument("--live", default="", help="JSONL live event stream path (dashboard)")
    parser.add_argument("--decision-interval", type=int, default=None, help="override ticks per decision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    micro = MicroDuelConfig()
    if args.decision_interval is not None:
        micro = MicroDuelConfig(decision_interval=args.decision_interval)
    config = PPOConfig(
        total_updates=args.updates,
        num_envs=args.envs,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.lam,
        entropy_coef=args.entropy,
        hidden=args.hidden,
        device=args.device,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        opponent=args.opponent,
        out_dir=args.out,
    )
    trainer = PPOTrainer(config, micro_config=micro, live_path=args.live or None)
    metrics = trainer.train()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "final_eval": metrics.get("final_eval"),
                "random_baseline": metrics.get("random_baseline"),
                "total_seconds": metrics.get("total_seconds"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(metrics.get("final_eval"), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
