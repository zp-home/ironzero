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
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..micro_config import (
    DEFAULT_CONFIG,
    MicroDuelConfig,
    WIDE_MAX_ENTITIES,
)
from ..rwpp_env import DEFAULT_RUNTIME, DEFAULT_JAVA_HOME, RwppGameConfig
from .ppo import PPOConfig, _config_from_saved
from .rwpp_ppo import RwppPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO training against real Rusted Warfare 1.15 games (RWPP bridge)"
    )
    parser.add_argument("--updates", type=int, default=4, help="PPO updates")
    parser.add_argument("--envs", type=int, default=2, help="parallel real game instances")
    parser.add_argument("--rollout-steps", type=int, default=8, help="decisions per env per update")
    parser.add_argument("--episode-ticks", type=int, default=0,
                        help="simulated-tick cap per episode; 0 (default) derives it from the "
                             "map diagonal and player count via recommended_episode_ticks(), "
                             "then scales it by --episode-tick-multiplier. This is a safety "
                             "fuse, not the win condition -- the engine decides the result, "
                             "and reaching the cap instead pays timeout_penalty")
    parser.add_argument("--episode-tick-multiplier", type=float,
                        default=RwppGameConfig().episode_tick_multiplier,
                        help="scale the DERIVED tick cap (ignored when --episode-ticks is set). "
                             "The derived cap is consumed in full and is too short for the "
                             "built-in AI to finish a march; see RwppGameConfig's docstring. "
                             "Raising this without also cutting --produce-bonus makes winning "
                             "LESS attractive, because the production flow grows with it")
    parser.add_argument("--timeout-penalty", type=float,
                        default=RwppGameConfig().timeout_penalty,
                        help="penalty for hitting the tick cap without the engine deciding "
                             "the match; keeps stalling from being free")
    parser.add_argument("--win-reward", type=float, default=RwppGameConfig().win_reward,
                        help="terminal reward for a won match. Derived: must exceed the "
                             "production flow the episode can pay (see RwppGameConfig)")
    parser.add_argument("--loss-reward", type=float, default=RwppGameConfig().loss_reward,
                        help="terminal reward for a lost match (normally the negative of --win-reward)")
    parser.add_argument("--produce-bonus", type=float, default=RwppGameConfig().produce_bonus,
                        help="one-off reward per new combat unit. Keep small: this is an "
                             "unbounded FLOW and it competes with the one-off terminal reward")
    parser.add_argument("--build-bonus", type=float, default=RwppGameConfig().build_bonus,
                        help="one-off reward per finished building")
    parser.add_argument("--kill-bonus", type=float, default=RwppGameConfig().kill_bonus,
                        help="reward per baseline combat unit destroyed (scaled by threat value). "
                             "Bounded so a full enemy army pays less than one won match")
    parser.add_argument("--loss-penalty", type=float, default=RwppGameConfig().loss_penalty,
                        help="penalty per own baseline unit destroyed")
    parser.add_argument("--num-teams", type=int, default=RwppGameConfig().num_teams,
                        help="team count used to size a derived --episode-ticks cap")
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
    parser.add_argument("--max-entities", type=int, default=None,
                        help=(
                            "entity slots per team block in the observation "
                            f"(>= units-per-team). Default: {WIDE_MAX_ENTITIES} "
                            "unless --units-per-team is larger, because 8 was "
                            "measured to hide 26% of the enemy army (including "
                            "100% of its artillery). Pass 8 for the historical "
                            "contract."
                        ))
    parser.add_argument("--target-slots", type=int, default=0,
                        help="target head size; 0 = max-entities + 1 so every visible enemy can be named")
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
    parser.add_argument("--enemy-ai-difficulty", type=int, default=None, choices=[-2, -1, 0, 1, 2, 3],
                        help="pin only the enemy teams' difficulty, leaving the player/teacher side "
                             "alone; use it to give the BC teacher real pressure to react to")
    parser.add_argument("--engagement", type=float, default=RwppGameConfig().engagement_coef,
                        help="dense reward for closing distance to the enemy army "
                             f"(default {RwppGameConfig().engagement_coef}: enough to break the "
                             "initial gradient, small enough not to outbid fighting)")
    parser.add_argument("--frame-ms", type=int, default=0,
                        help="wall-clock ms per simulated frame: 16=real time, 1=~10x, "
                             "0=unpaced (~26x; needs --physics-ms>0 so the delta stays pinned)")
    parser.add_argument("--physics-ms", type=int, default=16,
                        help="pin the engine's simulated delta (16 = real time, so --ticks counts "
                             "real game ticks); --frame-ms then only paces the wall clock")
    parser.add_argument("--fast-forward", type=int, default=1,
                        help="engine game speed (1..32): every tick advances N times the game "
                             "time, so a match needs 1/N the ticks. 1 = off. Changes fidelity.")
    parser.add_argument("--lockstep", action=argparse.BooleanOptionalAction, default=True,
                        help="block after every observation until the policy answers, so an "
                             "unpaced engine cannot drop frames (control period stays 10 ticks)")
    parser.add_argument("--lockstep-timeout-ms", type=int, default=30000,
                        help="max lockstep wait before the bridge falls back to free running")
    parser.add_argument("--vision-radius", type=float, default=260.0,
                        help="observation fog radius in map pixels (<=0 disables fog)")
    parser.add_argument("--no-opening-extractor", dest="opening_extractor",
                        action="store_false", default=True,
                        help="disable the opening-extractor curriculum (the env normally "
                             "queues one extractor at reset to break the exploration barrier)")
    parser.add_argument("--no-opening-factory", dest="opening_factory",
                        action="store_false", default=True,
                        help="disable the opening-factory curriculum (the env normally "
                             "queues one land factory once the opening extractor is "
                             "finished; without it 'bonus_produce' can never fire)")
    parser.add_argument("--no-opening-production", dest="opening_production",
                        action="store_false", default=True,
                        help="disable the opening-production curriculum (the env normally "
                             "queues one combat unit once a factory is finished, so "
                             "'bonus_produce' can leave 0.000)")
    parser.add_argument("--action-slots", type=int, choices=(1, 2), default=1,
                        help="action layout: 1 = the historical single slot (one order per "
                             "decision), 2 = two slots (one economy order AND one tactical order "
                             "per decision). Default 1 keeps every existing checkpoint loadable "
                             "and bit-identical; a 2-slot run cannot resume a 1-slot checkpoint "
                             "because the head count changes 5 -> 10.")
    parser.add_argument("--curriculum-mode", choices=("orders", "mask", "off"), default="orders",
                        help="how the opening curriculum participates: 'orders' (default) lets it "
                             "append its own orders to the dispatch -- the historical behaviour; "
                             "'mask' makes it influence only the action masks and never emit an "
                             "order, so the policy is the sole source of every command; 'off' "
                             "disables it entirely. Use 'mask' with --action-slots 2 to measure "
                             "what the policy does once the curriculum stops acting for it.")
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

    # Resolve the visibility budget here rather than in argparse, because the
    # default depends on another argument: ``max_entities`` must be at least
    # ``units_per_team`` (a 25v25 run needs 25 slots) and must never fall back to
    # the historical 8 on a real economy map, where 8 was measured to hide 26% of
    # the enemy army and 100% of its artillery.
    units_per_team = int(args.units_per_team)
    if args.max_entities is None:
        max_entities = max(WIDE_MAX_ENTITIES, units_per_team)
    else:
        max_entities = int(args.max_entities)
    if args.target_slots:
        target_slots = int(args.target_slots)
    else:
        # +1 for the "no target" fallback, so every visible enemy is nameable.
        target_slots = max_entities + 1

    return RwppGameConfig(
        runtime=Path(args.runtime),
        java_home=Path(args.java_home),
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.episode_ticks),
        episode_tick_multiplier=float(args.episode_tick_multiplier),
        timeout_penalty=float(args.timeout_penalty),
        win_reward=(float(args.win_reward) if args.win_reward is not None else None),
        loss_reward=(float(args.loss_reward) if args.loss_reward is not None else None),
        produce_bonus=float(args.produce_bonus),
        build_bonus=float(args.build_bonus),
        kill_bonus=float(args.kill_bonus),
        loss_penalty=float(args.loss_penalty),
        num_teams=max(1, int(args.num_teams)),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        units_per_team=units_per_team,
        max_entities=max_entities,
        target_slots=target_slots,
        terrain_path=Path(args.terrain) if args.terrain else None,
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
        ai_pilot=bool(args.ai_pilot) or bool(args.record_ai_orders),
        record_ai_orders=bool(args.record_ai_orders),
        ai_difficulty=(int(args.ai_difficulty) if args.ai_difficulty is not None else None),
        enemy_ai_difficulty=(
            int(args.enemy_ai_difficulty) if args.enemy_ai_difficulty is not None else None
        ),
        engagement_coef=float(args.engagement),
        action_slots=int(args.action_slots),
        curriculum_mode=str(args.curriculum_mode),
        frame_ms=max(0, int(args.frame_ms)),
        physics_ms=max(0, int(args.physics_ms)),
        fast_forward=max(1, int(args.fast_forward)),
        lockstep=bool(args.lockstep),
        lockstep_timeout_ms=int(args.lockstep_timeout_ms),
        vision_radius=float(args.vision_radius),
        opening_extractor=bool(getattr(args, "opening_extractor", True)),
        opening_factory=bool(getattr(args, "opening_factory", True)),
        opening_production=bool(getattr(args, "opening_production", True)),
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
                "episode_ticks": game.resolved_episode_ticks(),
                "episode_ticks_explicit": args.episode_ticks,
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
        loaded = _load_compatible_weights(
            trainer.policy,
            state_dict,
            # ``_config_from_saved`` rebuilds a :class:`MicroDuelConfig` from a dict
            # of *config fields*. It must be handed the checkpoint's
            # ``micro_config`` sub-dict, not the whole payload.
            #
            # This call used to pass ``payload`` itself. Every key it was looking
            # for (``global_dim``, ``type_features``, ``entity_features``, ...)
            # lives under ``payload["micro_config"]``, so the filtered dict came
            # back empty and the function silently returned ``DEFAULT_CONFIG``.
            # The weight migration was therefore always told the *source* layout
            # was the 8-entity / 3-channel / type_features=9 default, no matter
            # what the checkpoint actually was. For a warm start from a matching
            # 25v25 checkpoint that means ``_column_source`` computed a remap for a
            # layout that never existed, and the transferred trunk columns were
            # bound to the wrong features -- the failure mode that looks like
            # "loaded 99% of parameters" while the policy behaves like noise.
            source_micro=_config_from_saved(dict(payload.get("micro_config") or {})),
            target_micro=trainer.micro,
        )
        transfer = getattr(trainer.policy, "load_transfer_summary", {}) or {}
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "init_update": payload.get("update"),
                    "initialized": True,
                    "loaded_parameters": len(loaded),
                    "transfer": transfer,
                },
                ensure_ascii=False,
            )
        )
        if transfer.get("warning"):
            # Loud on purpose: a "warm start" that silently kept most weights
            # random is worse than no warm start at all, because the curve then
            # looks like PPO failed when the initialisation was never applied.
            print(f"[warmstart] {transfer['warning']}", file=sys.stderr, flush=True)
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
            "episode_ticks": game.resolved_episode_ticks(),
            "episode_ticks_explicit": game.episode_ticks,
            "episode_tick_multiplier": game.episode_tick_multiplier,
            "timeout_penalty": game.timeout_penalty,
            # Reward weights recorded so a run's objective can be audited without
            # guessing. The terminal values live on the micro config because that
            # is what ``compute_step_shaping`` reads.
            "win_reward": micro.win_reward,
            "loss_reward": micro.loss_reward,
            "produce_bonus": game.produce_bonus,
            "build_bonus": game.build_bonus,
            "kill_bonus": game.kill_bonus,
            "loss_penalty": game.loss_penalty,
            "num_teams": game.num_teams,
            "frame_ms": game.frame_ms,
            "physics_ms": game.physics_ms,
            "fast_forward": game.fast_forward,
            "lockstep": bool(game.lockstep),
            "lockstep_timeout_ms": game.lockstep_timeout_ms,
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
            "enemy_ai_difficulty": game.enemy_ai_difficulty,
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


def _load_compatible_weights(policy, state_dict: dict, source_micro=None, target_micro=None) -> list[str]:
    """Load weights flexibly, adapting mismatched head/input dimensions.

    The action space has grown over time (kind head 5 -> 10, target head 9 ->
    26, ...). Any head whose width changed is migrated by copying the shared
    leading rows/columns, so a warm start keeps every weight it *can* reuse
    instead of silently discarding the whole tensor. Without this, a checkpoint
    from an older action space loaded "successfully" while the kind head stayed
    randomly initialised.

    ``source_micro``/``target_micro`` feed the observation-layout re-map in
    :func:`..training.ppo._migrate_state_dict`. Without them a plain prefix copy
    is used, which is wrong whenever a block was *inserted* (``global_dim``
    12 -> 15): the retained columns would then be bound to different features
    than the ones the checkpoint was trained on.

    **The transfer ratio is reported, not buried.** A "warm start" that only
    carries a quarter of its parameters is not a warm start, but the old code
    said so on a single hidden warning line. Measured on the Web default
    (``init_mode=warmstart`` on ``[p2]Lake (2p)``): the host resolved
    ``bc_ironzero25`` (25v25, obs_dim 768) for a 2v2 map (obs_dim 511), so
    ``trunk.0.weight``, ``heads.0`` and ``heads.4`` were all shape-incompatible
    and only **26% of the BC parameters transferred -- 65.4% of the policy stayed
    randomly initialised**, with nothing on screen saying so. The returned
    summary now carries the exact numbers and ``train_rwpp`` prints them, so the
    dashboard can show "本次实际迁移 26%".
    """

    from .ppo import _migrate_state_dict

    model_dict = _migrate_state_dict(
        policy.state_dict(), state_dict, source_layout=source_micro, target_layout=target_micro
    )
    loaded_keys: list[str] = []
    exact = 0
    adapted = 0
    kept_random = 0
    total_params = 0
    transferred_params = 0
    informative = 0
    target = policy.state_dict()
    for name, param in target.items():
        total_params += param.numel()
        src = state_dict.get(name)
        if src is None:
            kept_random += param.numel()
            continue
        if tuple(src.shape) == tuple(param.shape):
            loaded_keys.append(name)
            exact += 1
            transferred_params += param.numel()
            informative += param.numel()
        else:
            kept_random += param.numel()
            # How many entries of THIS target tensor the migration actually
            # fills from the checkpoint. A 2-D layer keeps min(rows) x min(cols)
            # of the old block; a wider head keeps its overlapping rows.
            kept = 0
            if src.dim() == param.dim() == 2:
                kept = min(int(src.shape[0]), int(param.shape[0])) * min(
                    int(src.shape[1]), int(param.shape[1])
                )
            elif src.dim() == param.dim() == 1:
                kept = min(int(src.shape[0]), int(param.shape[0]))
            informative += kept
            if name == "trunk.0.weight":
                # This one is re-mapped column-by-column through
                # ``_column_source``, not per-tensor: a (256, 768) BC trunk
                # feeding a (256, 511) Lake trunk keeps 508 of 511 columns
                # (the 3 that are genuinely new stay zero). Report the *real*
                # number instead of min(cols), which would overstate it.
                informed_cols = _informed_trunk_columns(source_micro, target_micro, param.shape[1])
                if informed_cols is not None:
                    remapped = informed_cols * int(param.shape[0])
                    informative += remapped - kept
                    kept = remapped
            if src.dim() == param.dim() and src.dim() in (1, 2):
                adapted += 1
                loaded_keys.append(
                    f"{name} (adapted {tuple(src.shape)} -> {tuple(param.shape)}, "
                    f"{kept}/{param.numel()} entries from checkpoint)"
                )
            else:
                loaded_keys.append(
                    f"{name} (shape {tuple(src.shape)} vs {tuple(param.shape)}: random init)"
                )
    policy.load_state_dict(model_dict)
    ratio = (transferred_params / total_params) if total_params else 0.0
    summary = {
        "exact": exact,
        "adapted": adapted,
        "kept_random_params": kept_random,
        "total_params": total_params,
        "transferred_params": int(transferred_params),
        "transfer_ratio": round(ratio, 4),
        # Coverage answers a different question than the ratio above: "what
        # fraction of the *target* policy was actually shaped by the checkpoint?"
        # A wide tensor can be adapted (its overlapping block copied) while the
        # great majority of its entries stay at random init -- e.g. a
        # ``trunk.0.weight`` of (256, 511) receiving 256x4 re-mapped columns is
        # 0.8% informative while counting as "fully adapted". On the measured
        # Web-default case (bc_ironzero25 -> Lake 2v2) coverage is what exposes
        # the problem: ~26%, not 99%.
        "informative_params": int(informative),
        "coverage": round((informative / total_params) if total_params else 0.0, 4),
    }
    policy.load_transfer_summary = summary  # type: ignore[attr-defined]
    if summary["coverage"] < 0.5:
        summary["warning"] = (
            "warm start is only "
            f"{summary['coverage']:.0%} informative ({summary['transfer_ratio']:.0%} of "
            "tensors touched): the checkpoint was trained on a different "
            "observation/action layout, so most of the policy stays random. "
            "Check units_per_team / max_entities / target_slots / map_channels."
        )
    return loaded_keys


def _informed_trunk_columns(source_micro, target_micro, target_cols: int) -> int | None:
    """How many target trunk-input columns the checkpoint actually fills.

    ``trunk.0.weight`` is the one tensor whose migration is *column by column*
    (``ppo._column_source``), so its real coverage cannot be read off the tensor
    shapes. Returns ``None`` when the two layouts are unavailable, meaning the
    caller should fall back to the shape-based estimate.
    """

    if source_micro is None or target_micro is None:
        return None
    try:
        from .ppo import _column_source

        remap = _column_source(source_micro, target_micro)
    except Exception:  # noqa: BLE001 - never fail a warm start over reporting
        return None
    if not remap:
        return None
    informed = sum(1 for col in remap[:target_cols] if col is not None)
    return informed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
