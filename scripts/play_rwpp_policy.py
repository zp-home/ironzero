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


def state_to_frame(state: dict, tick: int, episode: int = 0) -> dict:
    units = []
    for entity in state.get("own_entities", []) + state.get("visible_enemy_entities", []):
        team = entity.get("team_id", -1)
        if team not in (0, 1):
            continue
        units.append(
            {
                "id": entity.get("entity_id"),
                "team": team,
                "group": 0,
                "x": entity.get("x", 0.0),
                "y": entity.get("y", 0.0),
                "hp": entity.get("hp", 0.0),
                "max_hp": entity.get("max_hp", 1.0),
                "order": "idle",
                "target": None,
            }
        )
    own_frac, enemy_frac = hp_fractions(state)
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
        frame_ms=max(0, int(args.frame_ms)),
        vision_radius=0.0,  # the checkpoint was trained with fog internally? keep full obs for eval parity
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
    try:
        obs, info = env.reset()
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            frame = state_to_frame(env.state, int(env.last_tick))
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
        final_frame = state_to_frame(env.state, int(env.last_tick))
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
            "width_px": args.map_width,
            "height_px": args.map_height,
            "tile_width": 20,
            "tile_height": 20,
        },
        "vision_radius": 0,
        "fog": False,
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
    summary = summarize(
        [result],
        [total_reward],
        [hp_fractions(env.state)[0]],
        [hp_fractions(env.state)[1]],
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
