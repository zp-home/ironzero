# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained PPO checkpoint on the real Rusted Warfare 1.15 scenario.

This is the reproducible "how good is the model actually?" entry point: it
loads the checkpoint's own ``micro_config`` (observation dim, action heads,
25-vs-25 army size, terrain channel, fog radius), runs N deterministic games
sequentially on fresh headless instances, and writes a report with per-episode
outcomes plus Wilson 95% confidence intervals under ``runs/<out>/``.

Example (20 games on the fixed IronZero25 map)::

    $env:PYTHONPATH = "src"
    python scripts/eval_rwpp_policy.py `
      --checkpoint runs/rwpp_ppo/checkpoint_best.pt `
      --runtime ".runtime\\rwpp-runtime-1.15" `
      --java-home "E:\\jdk25copy" `
      --map "maps/ironzero/IronZero25.tmx" --map-name "IronZero25" `
      --map-width 2700 --map-height 2500 `
      --episodes 20 --frame-ms 1 --out runs/eval_rwpp

The map path/size only affect how the game is launched; the observation
contract always comes from the checkpoint, so a checkpoint trained on a
different map fails loudly instead of silently reading a misaligned vector.
Add ``--record`` to also write an ``ironzero.replay/1`` replay of one game.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from ironzero.micro_config import MicroDuelConfig  # noqa: E402
from ironzero.rwpp_env import (  # noqa: E402
    DEFAULT_JAVA_HOME,
    DEFAULT_RUNTIME,
    RwppGameConfig,
)
from ironzero.training.evaluate_rwpp import evaluate_rwpp_policy  # noqa: E402
from ironzero.training.ppo import load_policy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint on real Rusted Warfare 1.15 games"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt / checkpoint_best.pt")
    parser.add_argument("--runtime", default=str(REPO_ROOT / ".runtime" / "rwpp-runtime-1.15"))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--map", default="maps/ironzero/IronZero25.tmx")
    parser.add_argument("--map-name", default="IronZero25")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--episodes", type=int, default=20, help="sequential fresh games")
    parser.add_argument("--episode-ticks", type=int, default=3600)
    parser.add_argument("--frame-ms", type=int, default=1,
                        help="wall-clock ms per simulated frame (1 = fast, 16 = real time)")
    parser.add_argument("--port-base", type=int, default=5420,
                        help="bridge port for the first evaluation game (avoid 5140/5200/5401/5410)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--out", default="runs/eval_rwpp")
    parser.add_argument("--terrain", default="",
                        help="override the checkpoint's terrain JSON path (same obs_dim only)")
    parser.add_argument("--vision-radius", type=float, default=None,
                        help="override the checkpoint's fog radius (default: from checkpoint)")
    parser.add_argument("--win-mode", default="", help="bridge --win-mode passthrough (e.g. annihilation)")
    parser.add_argument("--unlock-ai", action="store_true", help="bridge --unlock-ai passthrough")
    parser.add_argument("--record", action="store_true",
                        help="also write one ironzero.replay/1 replay via play_rwpp_policy")
    return parser


def _resolve_micro(policy: torch.nn.Module, micro: MicroDuelConfig, args: argparse.Namespace) -> MicroDuelConfig:
    """Apply optional terrain/fog overrides while keeping the checkpoint layout."""

    updates: dict[str, object] = {}
    if args.terrain:
        candidate_path = Path(args.terrain)
        if not candidate_path.is_absolute() and not candidate_path.is_file():
            candidate_path = REPO_ROOT / candidate_path
        if not candidate_path.is_file():
            raise SystemExit(f"--terrain file not found: {args.terrain}")
        updates["terrain_path"] = str(candidate_path.resolve())
        updates["map_channels"] = max(int(micro.map_channels), 4)
    if args.vision_radius is not None:
        updates["vision_radius"] = float(args.vision_radius)
    if not updates:
        return micro
    candidate = replace(micro, **updates)
    obs_dim = int(getattr(policy, "obs_dim", candidate.flat_obs_dim))
    if candidate.flat_obs_dim != obs_dim:
        raise SystemExit(
            f"override changes obs_dim {obs_dim} -> {candidate.flat_obs_dim}: the checkpoint "
            "policy cannot consume it (retrain with the terrain channel enabled)"
        )
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    device = torch.device(args.device)
    policy, micro = load_policy(args.checkpoint, device=device)
    policy.eval()
    micro = _resolve_micro(policy, micro, args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    game = RwppGameConfig(
        runtime=Path(args.runtime),
        java_home=Path(args.java_home),
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.episode_ticks),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        units_per_team=int(micro.units_per_team),
        max_entities=int(micro.max_entities),
        target_slots=int(micro.target_slots),
        terrain_path=Path(micro.terrain_path) if micro.terrain_path else None,
        frame_ms=max(0, int(args.frame_ms)),
        vision_radius=float(micro.vision_radius or 0.0),
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
    )
    print(
        json.dumps(
            {
                "mode": "eval-rwpp",
                "checkpoint": str(args.checkpoint),
                "obs_dim": micro.flat_obs_dim,
                "action_heads": list(micro.action_nvec),
                "units_per_team": micro.units_per_team,
                "max_entities": micro.max_entities,
                "target_slots": micro.target_slots,
                "map_channels": micro.map_channels,
                "terrain": micro.terrain_path,
                "vision_radius": micro.vision_radius,
                "episodes": args.episodes,
                "map": args.map,
                "out": str(out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def report_episode(record: dict) -> None:
        print(json.dumps({"episode": record}, ensure_ascii=False), flush=True)

    started = time.time()
    evaluation = evaluate_rwpp_policy(
        policy,
        game=game,
        episodes=int(args.episodes),
        micro_config=micro,
        port_base=int(args.port_base),
        seed=int(args.seed),
        device=device,
        deterministic=True,
        report_path=out_dir / "eval_report.json",
        on_episode=report_episode,
    )
    report_path = Path(evaluation.get("report_path") or out_dir / "eval_report.json")
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": evaluation["episodes"],
        "wins": evaluation["wins"],
        "losses": evaluation["losses"],
        "timeouts": evaluation["timeouts"],
        "win_rate": round(float(evaluation["win_rate"]), 4),
        "win_rate_ci95": [round(v, 4) for v in evaluation["win_rate_ci95"]],
        "loss_rate": round(float(evaluation["loss_rate"]), 4),
        "timeout_rate": round(float(evaluation["timeout_rate"]), 4),
        "mean_return": round(float(evaluation["mean_return"]), 4),
        "mean_own_hp_frac": round(float(evaluation["mean_own_hp_frac"]), 4),
        "mean_enemy_hp_frac": round(float(evaluation["mean_enemy_hp_frac"]), 4),
        "mean_decisions": round(float(evaluation["mean_decisions"]), 2),
        "total_seconds": round(time.time() - started, 1),
        "report": str(report_path),
    }
    if args.record:
        try:
            summary["replay"] = _record_replay(args, out_dir)
        except Exception as exc:  # noqa: BLE001 - recording must not lose the eval
            print(f"[warn] replay recording failed: {exc}", file=sys.stderr)
            summary["replay"] = None
    (out_dir / "eval_summary.json").write_text(
        json.dumps({"summary": summary, "report": evaluation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _load_play_module():
    path = REPO_ROOT / "scripts" / "play_rwpp_policy.py"
    spec = importlib.util.spec_from_file_location("play_rwpp_policy", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_replay(args: argparse.Namespace, out_dir: Path) -> str:
    """Render one deterministic game as an ``ironzero.replay/1`` file."""

    play = _load_play_module()
    replay_args = [
        "--checkpoint", str(args.checkpoint),
        "--port", str(int(args.port_base) + 1),
        "--ticks", str(int(args.episode_ticks)),
        "--map", args.map,
        "--map-name", args.map_name,
        "--map-width", str(args.map_width),
        "--map-height", str(args.map_height),
        "--frame-ms", str(max(1, int(args.frame_ms))),
        "--device", args.device,
        "--out-root", str(out_dir / "record"),
    ]
    if args.terrain:
        replay_args.extend(["--terrain", args.terrain])
    if args.win_mode:
        replay_args.extend(["--win-mode", args.win_mode])
    if args.unlock_ai:
        replay_args.append("--unlock-ai")
    code = play.main(replay_args)
    if code not in (0, None):
        raise RuntimeError(f"play_rwpp_policy exited with {code}")
    replay_dir = out_dir / "record"
    replays = sorted(replay_dir.rglob("replay-*.json"))
    return str(replays[-1]) if replays else str(replay_dir)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
