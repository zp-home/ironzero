# SPDX-License-Identifier: Apache-2.0
"""Record inference episodes (or random rollouts) into replay JSON files.

This is the worker the dashboard spawns for the sandbox tab::

    python -m ironzero.training.record --checkpoint runs/micro_duel/checkpoint.pt \
        --episodes 3 --opponent scripted --out-dir runs/platform/infer-xxx/replays \
        --stream runs/platform/infer-xxx/frames.jsonl

It can also run without a checkpoint (``--mode random``) so the platform has a
working demo before any model is trained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG
from ..mock_engine import MicroDuelAdapter, make_terrain_config
from ..replay import build_frame, build_meta, episode_meta, write_replay
from ..terrain import TerrainGrid
from .evaluate import sample_masked_action
from .policy import masks_to_tensors
from .ppo import load_policy


class JsonlStream:
    """Append-only JSONL writer used for live dashboard streaming."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def emit(self, payload: dict) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record IronZero micro-duel replays")
    parser.add_argument("--checkpoint", default="", help="checkpoint.pt path; empty or 'random' = random policy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "idle", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sample", "random", "scripted"])
    parser.add_argument("--out-dir", default="runs/replays")
    parser.add_argument("--stream", default="", help="optional JSONL path for live frames")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--prefix", default="replay")
    parser.add_argument("--interpolate", type=int, default=1, help="record every N decisions (1 = every frame)")
    parser.add_argument("--map-id", default="", help="selected dashboard map id, stored in the replay meta")
    parser.add_argument(
        "--sim-mode",
        default="arena",
        choices=["arena", "terrain"],
        help="arena = abstract 1000x1000 micro arena (default); terrain = real map collision + A*",
    )
    parser.add_argument(
        "--map-json",
        default="",
        help="path to the backend-generated map.json (required for --sim-mode terrain)",
    )
    parser.add_argument("--fog", action="store_true", help="enable fog of war (vision filtering)")
    parser.add_argument("--vision-radius", type=float, default=260.0, help="fog of war vision radius")
    return parser


def _load_terrain(path: str) -> TerrainGrid:
    if not path:
        raise ValueError("--sim-mode terrain requires --map-json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TerrainGrid.from_payload(payload)


def _build_env(
    *,
    config,
    opponent: str,
    seed: int,
    episode: int,
    terrain: TerrainGrid | None,
) -> MicroDuelEnv:
    adapter = None
    if terrain is not None:
        adapter = MicroDuelAdapter(config=config, opponent=opponent, terrain=terrain)
    return MicroDuelEnv(
        config=config,
        opponent=opponent,
        adapter=adapter,
        seed=seed,
        episode_prefix=f"infer-{episode}",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG
    policy = None
    checkpoint = args.checkpoint or ""
    if args.mode == "scripted":
        policy_mode = "scripted"
        device = torch.device("cpu")
    elif checkpoint and checkpoint.lower() not in {"random", "none"}:
        device = torch.device(args.device)
        policy, config = load_policy(checkpoint, device=device)
        policy_mode = "sample" if args.mode == "sample" else "greedy"
    else:
        policy_mode = "random"
        device = torch.device("cpu")

    terrain: TerrainGrid | None = None
    if args.sim_mode == "terrain":
        try:
            terrain = _load_terrain(args.map_json)
        except (OSError, ValueError) as exc:
            print(f"[record] terrain mode error: {exc}", file=sys.stderr)
            return 2
        # Real map pixel size replaces the arena size; unit stats/episode length
        # stay identical (the observation encoder is rescaled by the same config).
        config = make_terrain_config(terrain, config)

    fog = bool(args.fog)
    if fog:
        config = replace(config, vision_radius=float(args.vision_radius))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = JsonlStream(args.stream or None)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = build_meta(
        checkpoint=checkpoint or None,
        opponent=args.opponent,
        seed=args.seed,
        config=config,
        episodes=args.episodes,
        mode=policy_mode,
        created_at=created_at,
    )
    meta["sim_mode"] = args.sim_mode
    meta["fog"] = fog
    meta["vision_radius"] = float(config.vision_radius if fog else args.vision_radius)
    if args.map_id:
        meta["map_id"] = args.map_id
    if terrain is not None:
        meta["map"] = {
            "id": terrain.map_id,
            "name": terrain.name,
            "width_px": terrain.width_px,
            "height_px": terrain.height_px,
            "tile_width": terrain.tile_width,
            "tile_height": terrain.tile_height,
        }
    stream.emit({"type": "start", "meta": meta})
    episode_summaries: list[dict] = []
    rng = np.random.default_rng(args.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    for episode in range(args.episodes):
        env = _build_env(
            config=config,
            opponent=args.opponent,
            seed=args.seed + episode,
            episode=episode,
            terrain=terrain,
        )
        obs, info = env.reset()
        frames: list[dict] = []
        total_reward = 0.0
        decisions = 0
        done = False
        while not done and decisions < args.max_decisions:
            if policy_mode == "scripted":
                engine = getattr(env.adapter, "engine", None)
                protocol_actions = engine.scripted_actions(0) if engine is not None else []
                obs, reward, terminated, truncated, info = env.step_actions(protocol_actions)
            else:
                if policy_mode == "random":
                    action = sample_masked_action(info["masks"], rng)
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    mask_tensors = masks_to_tensors([info["masks"]], device)
                    actions, _, _ = policy.act(
                        obs_tensor, mask_tensors, deterministic=(policy_mode == "greedy")
                    )
                    action = actions[0].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            decisions += 1
            state = info["raw_state"]
            frame = build_frame(
                state,
                episode=episode,
                decision=decisions,
                events=state.get("events", []),
                config=config,
            )
            frame["reward_total"] = round(total_reward, 4)
            keep = (decisions % max(1, args.interpolate) == 0) or terminated or truncated
            if keep:
                frames.append(frame)
                stream.emit({"type": "frame", **frame})
            done = terminated or truncated
        env.close()
        filename = f"{args.prefix}-{stamp}-ep{episode:02d}-{uuid.uuid4().hex[:6]}.json"
        replay_path = out_dir / filename
        summary = episode_meta(
            index=episode,
            seed=args.seed + episode,
            result=info.get("result"),
            total_reward=total_reward,
            decisions=decisions,
        )
        write_replay(replay_path, meta={**meta, **summary}, frames=frames, episodes=[summary])
        episode_summaries.append({**summary, "replay_path": str(replay_path), "filename": filename})
        stream.emit(
            {
                "type": "episode_end",
                "episode": episode,
                "result": info.get("result"),
                "return": round(total_reward, 4),
                "decisions": decisions,
                "replay_path": str(replay_path),
                "filename": filename,
            }
        )
    stream.emit({"type": "done", "episodes": episode_summaries})
    stream.close()
    print(json.dumps({"episodes": episode_summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
