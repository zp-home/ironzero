# SPDX-License-Identifier: Apache-2.0
"""Command line entry point: PPO with a policy in the loop of the real game.

The default scenario is the ``Artillery`` challenge map: the engine pre-places
22 player units against 25 AI units, so an episode is a real battle rather than
an economy skirmish.

Example (2 real game instances, 6 PPO updates):

    $env:PYTHONPATH = "src"
    python -m ironzero.training.train_rwpp --updates 6 --envs 2 --episode-ticks 900
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..micro_config import MicroDuelConfig
from ..rwpp_env import DEFAULT_RUNTIME, DEFAULT_JAVA_HOME, RwppGameConfig
from .ppo import PPOConfig
from .rwpp_ppo import RwppPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO training against real Rusted Warfare 1.15 games (RWPP bridge)"
    )
    parser.add_argument("--updates", type=int, default=4, help="PPO updates")
    parser.add_argument("--envs", type=int, default=2, help="parallel real game instances")
    parser.add_argument("--rollout-steps", type=int, default=8, help="decisions per env per update")
    parser.add_argument("--episode-ticks", type=int, default=3600,
                        help="simulated ticks per real-game episode (3600 = 60 s; a decided battle ends earlier)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2, help="real-game evaluation episodes")
    parser.add_argument("--port-base", type=int, default=5140, help="first bridge port for training envs")
    parser.add_argument("--eval-port-base", type=int, default=5200)
    parser.add_argument("--map", default="maps/challenge/Artillery.tmx")
    parser.add_argument("--map-name", default="Artillery")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--engagement", type=float, default=2.0, help="dense reward for closing distance to the enemy army")
    parser.add_argument("--frame-ms", type=int, default=2,
                        help="wall-clock ms per simulated frame: 16=real time, 2=~8x faster, 0=unpaced (too fast for the client)")
    parser.add_argument("--vision-radius", type=float, default=260.0,
                        help="observation fog radius in map pixels (<=0 disables fog)")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--out", default="runs/rwpp_ppo")
    parser.add_argument("--live", default="", help="JSONL live event stream (dashboard)")
    parser.add_argument("--use-llm", action="store_true",
                        help="enable the LLM strategic layer (falls back to the rule strategy)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    game = RwppGameConfig(
        runtime=Path(args.runtime),
        java_home=Path(args.java_home),
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.episode_ticks),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        engagement_coef=float(args.engagement),
        frame_ms=max(0, int(args.frame_ms)),
        vision_radius=float(args.vision_radius),
    )
    micro = game.micro_config()
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
        out_dir=args.out,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = Path(args.live) if args.live else out_dir / "live.jsonl"

    trainer = RwppPPOTrainer(
        config,
        game=game,
        micro_config=micro,
        live_path=str(live),
        port_base=args.port_base,
        eval_port_base=args.eval_port_base,
        strategy=_build_strategy(args.use_llm),
    )
    print(
        json.dumps(
            {
                "mode": "real-game-ppo",
                "device": str(trainer.device),
                "gpu": _gpu_name(),
                "envs": args.envs,
                "episode_ticks": args.episode_ticks,
                "obs_dim": micro.flat_obs_dim,
                "action_heads": list(micro.action_nvec),
                "strategy": getattr(trainer.vec.envs[0], "strategy", None).__class__.__name__
                if getattr(trainer.vec.envs[0], "strategy", None) is not None
                else "none",
                "out": str(out_dir),
                "live": str(live),
            },
            ensure_ascii=False,
        )
    )
    started = time.time()
    try:
        metrics = trainer.train()
    finally:
        try:
            trainer.vec.close()
        except Exception:  # noqa: BLE001
            pass
    elapsed = round(time.time() - started, 1)
    summary = {
        "mode": "real-game-ppo",
        "device": str(trainer.device),
        "updates": len(metrics.get("updates", [])),
        "episodes": sum(record.get("episodes", 0) for record in metrics.get("updates", [])),
        "total_seconds": elapsed,
        "final_eval": metrics.get("final_eval"),
        "config": asdict(config),
        "game": {
            "map": game.map_name,
            "map_path": game.map_relative,
            "map_size_px": [game.map_width_px, game.map_height_px],
            "episode_ticks": game.episode_ticks,
            "frame_ms": game.frame_ms,
            "vision_radius": game.vision_radius,
            "runtime": str(game.runtime),
        },
        "micro": asdict(micro),
    }
    (out_dir / "rwpp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("updates", "episodes", "total_seconds", "final_eval")}, ensure_ascii=False))
    return 0


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    return "cpu-only"


def _build_strategy(use_llm: bool):
    """Build the strategic layer; rule strategy when the LLM is unavailable."""

    try:
        from ..strategy.llm_strategy import build_strategy
    except Exception:  # noqa: BLE001 - strategy package is optional
        return None
    try:
        strategy = build_strategy(use_llm)
        print(f"[strategy] using {strategy.__class__.__name__} (llm={use_llm})")
        return strategy
    except Exception as exc:  # noqa: BLE001
        print(f"[strategy] build failed ({exc}); continuing without a strategy layer")
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
