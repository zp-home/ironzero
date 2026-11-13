# SPDX-License-Identifier: Apache-2.0
"""Command line entry point: PPO with a policy in the loop of the real game.

The default scenario is the legacy ``Artillery`` challenge map (22 player units
against 25 AI units). Defaults keep that old contract; the fixed IronZero25
25-vs-25 arena is selected with the explicit entity/terrain/map flags:

    $env:PYTHONPATH = "src"
    python -m ironzero.training.train_rwpp `
      --map "maps/ironzero/IronZero25.tmx" --map-name "IronZero25" `
      --map-width 2700 --map-height 2500 `
      --units-per-team 25 --max-entities 25 --target-slots 26 `
      --terrain "configs/maps/IronZero25.terrain.json" `
      --win-mode annihilation --unlock-ai `
      --envs 2 --updates 100 --frame-ms 1 --out runs/rwpp_ppo_25

Observation/action contract for that command: obs_dim 771 (15 global + 2x25x10
entity + 4x8x8 grid) and action heads ``(10, 5, 8, 8, 26)``, where the kind
head is ``len(ACTION_KINDS)``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
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
    # 360 decisions per episode: 0.999^360 = 0.698 keeps the terminal reward
    # inside the horizon (0.99^360 = 0.027 hid it completely).
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2,
                        help="real-game evaluation episodes per eval (sequential fresh games; keep <= 8)")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True,
                        help="write checkpoint_best.pt whenever the eval win rate improves")
    parser.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="run evaluation in a non-blocking background thread without pausing training")
    parser.add_argument("--port-base", type=int, default=5140, help="first bridge port for training envs")
    parser.add_argument("--eval-port-base", type=int, default=5200)
    parser.add_argument("--map", default="maps/challenge/Artillery.tmx")
    parser.add_argument("--map-name", default="Artillery")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--units-per-team", type=int, default=DEFAULT_CONFIG.units_per_team,
                        help="units per side in the observation/action contract (25 for the IronZero25 map)")
    parser.add_argument("--max-entities", type=int, default=DEFAULT_CONFIG.max_entities,
                        help="entity slots per team block in the observation (>= units-per-team)")
    parser.add_argument("--target-slots", type=int, default=0,
                        help="target head size; 0 = max-entities + 1 (25 enemies + no-target)")
    parser.add_argument("--terrain", default="",
                        help="terrain passability JSON (TerrainGrid map.json format); adds a 4th map channel")
    parser.add_argument("--win-mode", default="",
                        help="bridge win condition passed through as --win-mode (e.g. annihilation); omitted by default")
    parser.add_argument("--unlock-ai", action="store_true",
                        help="let the built-in AI play; passed through as --unlock-ai to the bridge")
    parser.add_argument("--ai-pilot", action="store_true",
                        help="let the built-in AI drive the controlled team (behaviour-cloning teacher)")
    parser.add_argument("--record-ai-orders", action="store_true",
                        help="record the AI pilot's orders as BC labels (implies --ai-pilot)")
    parser.add_argument("--ai-difficulty", type=int, default=None, choices=[-2, -1, 0, 1, 2, 3],
                        help="pin every AI team's difficulty (-2..3); omitted keeps the engine "
                             "default, which is 0 (正常, x1.00 income, no AI cheat bonus)")
    parser.add_argument("--engagement", type=float, default=2.0, help="dense reward for closing distance to the enemy army")
    parser.add_argument("--frame-ms", type=int, default=2,
                        help="wall-clock ms per simulated frame: 16=real time, 2=~8x faster, 0=unpaced (too fast for the client)")
    parser.add_argument("--physics-ms", type=int, default=16,
                        help="pin the engine's simulated delta (16 = real time, so --ticks counts "
                             "real game ticks); --frame-ms then only paces the wall clock")
    parser.add_argument("--vision-radius", type=float, default=260.0,
                        help="observation fog radius in map pixels (<=0 disables fog)")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--out", default="runs/rwpp_ppo")
    parser.add_argument("--init-checkpoint", default="",
                        help="initialize the PPO policy from a checkpoint (e.g. a behavior-cloning warm start)")
    parser.add_argument("--live", default="", help="JSONL live event stream (dashboard)")
    parser.add_argument("--use-llm", action="store_true",
                        help="enable the LLM strategic layer (falls back to the rule strategy)")
    return parser


def build_game(args: argparse.Namespace) -> RwppGameConfig:
    """Build the real-game config from parsed CLI args (kept testable)."""

    return RwppGameConfig(
        runtime=Path(args.runtime),
        java_home=Path(args.java_home),
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.episode_ticks),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        units_per_team=int(args.units_per_team),
        max_entities=int(args.max_entities),
        target_slots=int(args.target_slots),
        terrain_path=Path(args.terrain) if args.terrain else None,
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
        ai_pilot=bool(args.ai_pilot) or bool(args.record_ai_orders),
        record_ai_orders=bool(args.record_ai_orders),
        ai_difficulty=(int(args.ai_difficulty) if args.ai_difficulty is not None else None),
        engagement_coef=float(args.engagement),
        frame_ms=max(0, int(args.frame_ms)),
        physics_ms=max(0, int(args.physics_ms)),
        vision_radius=float(args.vision_radius),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    game = build_game(args)
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
        save_best=bool(args.save_best),
        async_eval=bool(args.async_eval),
        out_dir=args.out,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    live = Path(args.live) if args.live else out_dir / "live.jsonl"
    if args.eval_episodes > 8:
        print(
            f"[warn] --eval-episodes {args.eval_episodes} launches one fresh game per episode; "
            "training-loop evaluation above 8 games is usually too slow"
        )
    if micro.max_entities < micro.units_per_team:
        print(
            f"[warn] max_entities {micro.max_entities} < units_per_team {micro.units_per_team}: "
            "some units will not be visible in the observation"
        )

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
                "units_per_team": micro.units_per_team,
                "max_entities": micro.max_entities,
                "target_slots": micro.target_slots,
                "map_channels": micro.map_channels,
                "terrain": micro.terrain_path,
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
    if args.init_checkpoint:
        import torch

        payload = torch.load(Path(args.init_checkpoint), map_location="cpu", weights_only=False)
        state_dict = payload.get("model") or payload.get("state_dict")
        if state_dict is None:
            raise SystemExit(f"--init-checkpoint has no model weights: {args.init_checkpoint}")
        loaded = _load_compatible_weights(trainer.policy, state_dict)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "init_update": payload.get("update"),
                    "initialized": True,
                    "loaded_parameters": len(loaded),
                },
                ensure_ascii=False,
            )
        )
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
        "best_eval": metrics.get("best_eval"),
        "best_checkpoint": metrics.get("best_checkpoint"),
        "config": asdict(config),
        "game": {
            "map": game.map_name,
            "map_path": game.map_relative,
            "map_size_px": [game.map_width_px, game.map_height_px],
            "episode_ticks": game.episode_ticks,
            "frame_ms": game.frame_ms,
            "physics_ms": game.physics_ms,
            "vision_radius": game.vision_radius,
            "units_per_team": micro.units_per_team,
            "max_entities": micro.max_entities,
            "target_slots": micro.target_slots,
            "terrain": micro.terrain_path,
            "win_mode": game.win_mode,
            "unlock_ai": bool(game.unlock_ai),
            "ai_pilot": bool(game.ai_pilot),
            "record_ai_orders": bool(game.record_ai_orders),
            "ai_difficulty": game.ai_difficulty,
            "runtime": str(game.runtime),
        },
        "micro": asdict(micro),
    }
    (out_dir / "rwpp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("updates", "episodes", "total_seconds", "final_eval", "best_checkpoint")
            },
            ensure_ascii=False,
        )
    )
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


def _load_compatible_weights(policy, state_dict: dict) -> list[str]:
    """Load weights flexibly, adapting mismatched head/input dimensions.

    The action space has grown over time (kind head 5 -> 10, target head 9 ->
    26, ...). Any head whose width changed is migrated by copying the shared
    leading rows/columns, so a warm start keeps every weight it *can* reuse
    instead of silently discarding the whole tensor. Without this, a checkpoint
    from an older action space loaded "successfully" while the kind head stayed
    randomly initialised.
    """

    import torch

    model_dict = policy.state_dict()
    loaded_keys = []
    for name, param in model_dict.items():
        if name not in state_dict:
            continue
        src = state_dict[name]
        if src.shape == param.shape:
            model_dict[name] = src
            loaded_keys.append(name)
            continue
        if src.dim() != param.dim():
            continue
        if src.dim() == 2:
            out_ch = min(param.shape[0], src.shape[0])
            in_ch = min(param.shape[1], src.shape[1])
            param_copy = param.clone()
            param_copy[:out_ch, :in_ch] = src[:out_ch, :in_ch]
            model_dict[name] = param_copy
            loaded_keys.append(f"{name} (adapted {tuple(src.shape)} -> {tuple(param.shape)})")
        elif src.dim() == 1:
            n = min(param.shape[0], src.shape[0])
            param_copy = param.clone()
            param_copy[:n] = src[:n]
            model_dict[name] = param_copy
            loaded_keys.append(f"{name} (adapted {tuple(src.shape)} -> {tuple(param.shape)})")

    missing = [name for name in model_dict if name not in loaded_keys]
    policy.load_state_dict(model_dict)
    if missing:
        loaded_keys.append(f"WARNING: kept random init for {missing}")
    return loaded_keys


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
