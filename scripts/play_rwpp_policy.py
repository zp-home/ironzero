# SPDX-License-Identifier: Apache-2.0
"""Play a trained PPO checkpoint inside a real Rusted Warfare 1.15 game.

This is the "model inference" path: the policy network drives the player's units
in a real headless game, frames are streamed as JSONL for the dashboard sandbox,
and the episode is written as an ``ironzero.replay/1`` file.

    python scripts/play_rwpp_policy.py --checkpoint runs/rwpp_ppo/checkpoint.pt \
        --ticks 3600 --stream runs/platform/rwpp-live-x/frames.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from ironzero.rwpp_env import (  # noqa: E402
    DEFAULT_JAVA_HOME,
    DEFAULT_RUNTIME,
    RwppDuelEnv,
    RwppGameConfig,
    hp_fractions,
)
from ironzero.training.evaluate import summarize  # noqa: E402
from ironzero.training.ppo import load_policy  # noqa: E402
from ironzero.training.policy import masks_to_tensors  # noqa: E402


def map_id_for(relative: str) -> str:
    import base64

    return base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")


def state_to_frame(
    state: dict,
    tick: int,
    episode: int = 0,
    *,
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> dict:
    units = []
    for entity in state.get("own_entities", []) + state.get("visible_enemy_entities", []):
        team = entity.get("team_id", -1)
        if team not in (0, 1):
            continue
        is_building = bool(entity.get("is_building", False))
        is_builder = bool(entity.get("is_builder", False))
        is_factory = bool(entity.get("is_factory", False))
        max_hp = float(entity.get("max_hp", 1.0))
        unit_type = str(entity.get("unit_type") or "")
        if not unit_type:
            if is_builder:
                unit_type = "builder"
            elif is_factory:
                unit_type = "factory"
            elif is_building:
                unit_type = "command_center" if max_hp >= 2500.0 else "building"
            else:
                unit_type = "tank"
        units.append(
            {
                "id": entity.get("entity_id"),
                "team": team,
                "group": 0,
                "x": entity.get("x", 0.0),
                "y": entity.get("y", 0.0),
                "hp": entity.get("hp", 0.0),
                "max_hp": max_hp,
                "order": "idle",
                "target": None,
                "is_building": is_building,
                "is_builder": is_builder,
                "is_factory": is_factory,
                "unit_type": unit_type,
            }
        )
    # Normalise against the episode-start full-army hp, not the alive-unit
    # maximum: dead units vanish from the stream, so without the reference a
    # damaged army that lost units would still show hp_frac == 1.0.
    own_frac, enemy_frac = hp_fractions(state, own_max_ref, enemy_max_ref)
    return {
        "episode": episode,
        "decision": tick // 10,
        "tick": tick,
        "time": round(tick / 60.0, 3),
        "reward": 0.0,
        "units": units,
        "events": [],
        "stats": {
            "own_hp_frac": round(own_frac, 4),
            "enemy_hp_frac": round(enemy_frac, 4),
            "own_alive": sum(1 for unit in units if unit["team"] == 0),
            "enemy_alive": sum(1 for unit in units if unit["team"] == 1),
        },
        "result": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a PPO checkpoint in a real RWPP game")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5130)
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--map", default="maps/challenge/Artillery.tmx")
    parser.add_argument("--map-name", default="Artillery")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--frame-ms", type=int, default=16, help="16 = real time playback, 2 = fast")
    parser.add_argument("--terrain", default="",
                        help="override the checkpoint's terrain JSON path (same obs_dim only)")
    parser.add_argument("--win-mode", default="", help="bridge --win-mode passthrough (e.g. annihilation)")
    parser.add_argument("--unlock-ai", action="store_true", help="bridge --unlock-ai passthrough")
    parser.add_argument("--stream", default="")
    parser.add_argument("--out-root", default="runs/platform")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    policy, micro = load_policy(args.checkpoint, device=device)
    policy.eval()
    # Optional terrain override: only legal when it keeps the checkpoint's
    # observation layout (same obs_dim), otherwise the policy cannot consume it.
    if args.terrain:
        terrain_path = Path(args.terrain)
        if not terrain_path.is_absolute() and not terrain_path.is_file():
            terrain_path = REPO_ROOT / terrain_path
        if not terrain_path.is_file():
            raise SystemExit(f"--terrain file not found: {args.terrain}")
        candidate = replace(
            micro,
            terrain_path=str(terrain_path.resolve()),
            map_channels=max(int(micro.map_channels), 4),
        )
        if candidate.flat_obs_dim != int(getattr(policy, "obs_dim", micro.flat_obs_dim)):
            raise SystemExit(
                f"--terrain changes obs_dim {micro.flat_obs_dim} -> {candidate.flat_obs_dim}; "
                "this checkpoint was trained without the terrain channel"
            )
        micro = candidate

    stamp = time.strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:6]
    job_dir = Path(args.out_root) / f"real-rwpp-{stamp}-{unique}"
    replay_dir = job_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    game = RwppGameConfig(
        runtime=DEFAULT_RUNTIME,
        java_home=DEFAULT_JAVA_HOME,
        map_relative=args.map,
        map_name=args.map_name,
        episode_ticks=int(args.ticks),
        map_width_px=float(args.map_width),
        map_height_px=float(args.map_height),
        units_per_team=int(micro.units_per_team),
        max_entities=int(micro.max_entities),
        target_slots=int(micro.target_slots),
        terrain_path=Path(micro.terrain_path) if micro.terrain_path else None,
        frame_ms=max(0, int(args.frame_ms)),
        # Inference parity: the policy was trained (and evaluated) with this fog
        # radius, so the replay must not hand it a different observation.
        vision_radius=float(micro.vision_radius or 0.0),
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
    )
    env = RwppDuelEnv(game, port=args.port, seed=0, micro_config=micro, episode_prefix="policy")
    stream = None
    if args.stream:
        stream_path = Path(args.stream)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream = stream_path.open("w", encoding="utf-8")
        stream.write(json.dumps({"type": "start", "job": job_dir.name, "checkpoint": args.checkpoint}) + "\n")
        stream.flush()
    frames: list[dict] = []
    started = time.time()
    result = "timeout"
    total_reward = 0.0
    own_ref: float | None = None
    enemy_ref: float | None = None
    try:
        obs, info = env.reset()
        own_ref = float(getattr(env, "own_max_ref", 0.0)) or None
        enemy_ref = float(getattr(env, "enemy_max_ref", 0.0)) or None
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            frame = state_to_frame(
                env.state,
                int(env.last_tick),
                own_max_ref=own_ref,
                enemy_max_ref=enemy_ref,
            )
            frames.append(frame)
            if stream is not None:
                stream.write(json.dumps({"type": "frame", **frame}, ensure_ascii=False) + "\n")
                stream.flush()
            mask_tensors = masks_to_tensors([info["masks"]], device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                actions, _, _ = policy.act(obs_tensor, mask_tensors, deterministic=True)
            action = actions[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        result = str(info.get("result") or "timeout")
        # The env returns the real bridge result in info["result"]; write it on the
        # terminal frame so the replay/stream shows the actual outcome.
        final_frame = state_to_frame(
            env.state,
            int(env.last_tick),
            own_max_ref=own_ref,
            enemy_max_ref=enemy_ref,
        )
        final_frame["result"] = result
        frames.append(final_frame)
        if stream is not None:
            stream.write(json.dumps({"type": "frame", **final_frame}, ensure_ascii=False) + "\n")
            stream.write(
                json.dumps(
                    {
                        "type": "done",
                        "result": result,
                        "frames": len(frames),
                        "terminated": result in ("victory", "defeat"),
                        "truncated": result == "timeout",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
    finally:
        env.close()
        if stream is not None:
            stream.close()

    meta = {
        "schema": "ironzero.replay/1",
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_path": str(args.checkpoint),
        "opponent": "ai",
        "seed": 0,
        "episodes": 1,
        "mode": "policy",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sim_mode": "terrain",
        "map": {
            "id": map_id_for(args.map),
            "name": args.map_name,
            "width_px": float(micro.map_width),
            "height_px": float(micro.map_height),
            "tile_width": 20,
            "tile_height": 20,
        },
        "vision_radius": float(micro.vision_radius or 0.0),
        "fog": float(micro.vision_radius or 0.0) > 0.0,
        "result": result,
        "terminated": result in ("victory", "defeat"),
        "truncated": result == "timeout",
        "return": round(total_reward, 4),
        "source": "rwpp-policy",
    }
    replay_path = replay_dir / f"replay-policy-{stamp}-{unique}.json"
    replay_path.write_text(
        json.dumps(
            {"schema": "ironzero.replay/1", "meta": meta, "episodes": [{"episode": 0}], "frames": frames},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    final_own_frac, final_enemy_frac = hp_fractions(env.state, own_ref, enemy_ref)
    summary = summarize(
        [result],
        [total_reward],
        [final_own_frac],
        [final_enemy_frac],
        [len(frames)],
    )
    print(
        json.dumps(
            {
                "replay": str(replay_path),
                "frames": len(frames),
                "seconds": round(time.time() - started, 1),
                "result": result,
                "return": round(total_reward, 4),
                "eval": summary,
            },
            ensure_ascii=False,
        )
    )
    return 0 if frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
