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
from pathlib import Path

import numpy as np
import torch

from ..env import MicroDuelEnv
from ..micro_config import DEFAULT_CONFIG
from ..replay import build_frame, build_meta, episode_meta, write_replay
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
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sample", "random"])
    parser.add_argument("--out-dir", default="runs/replays")
    parser.add_argument("--stream", default="", help="optional JSONL path for live frames")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--prefix", default="replay")
    parser.add_argument("--interpolate", type=int, default=1, help="record every N decisions (1 = every frame)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG
    policy = None
    checkpoint = args.checkpoint or ""
    if checkpoint and checkpoint.lower() not in {"random", "none"}:
        device = torch.device(args.device)
        policy, config = load_policy(checkpoint, device=device)
        policy_mode = "sample" if args.mode == "sample" else "greedy"
    else:
        policy_mode = "random"
        device = torch.device("cpu")

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
    stream.emit({"type": "start", "meta": meta})
    episode_summaries: list[dict] = []
    rng = np.random.default_rng(args.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    for episode in range(args.episodes):
        env = MicroDuelEnv(
            config=config,
            opponent=args.opponent,
            seed=args.seed + episode,
            episode_prefix=f"infer-{episode}",
        )
        obs, info = env.reset()
        frames: list[dict] = []
        total_reward = 0.0
        decisions = 0
        done = False
        while not done and decisions < args.max_decisions:
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
