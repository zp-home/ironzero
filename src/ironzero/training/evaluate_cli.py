# SPDX-License-Identifier: Apache-2.0
"""Evaluate a saved PPO checkpoint against the micro-duel baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..micro_config import MicroDuelConfig
from .evaluate import evaluate_policy, evaluate_random
from .ppo import load_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an IronZero micro-duel checkpoint")
    parser.add_argument("checkpoint", help="path to checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "idle", "random"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy, micro = load_policy(args.checkpoint, device=args.device)
    if not isinstance(micro, MicroDuelConfig):
        micro = MicroDuelConfig()
    policy_report = evaluate_policy(
        policy,
        episodes=args.episodes,
        config=micro,
        opponent=args.opponent,
        seed=args.seed,
        device=args.device,
        deterministic=not args.stochastic,
    )
    random_report = evaluate_random(
        episodes=args.episodes,
        config=micro,
        opponent=args.opponent,
        seed=args.seed + 1,
    )
    report = {"policy": policy_report, "random_baseline": random_report}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(args.checkpoint).with_name("eval_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
