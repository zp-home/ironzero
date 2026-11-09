# SPDX-License-Identifier: Apache-2.0
"""Behavior-cloning warm start for the IronZero 25-vs-25 real-game arena.

Why this exists
---------------
The scripted focus-fire controller is a very strong teacher on the fixed
``IronZero25`` map (it wins every probe game), but plain PPO from a random
policy spends a long time discovering "approach -> focus one target" on a
25-slot target head. This script distills the teacher into the PPO actor with
supervised cross-entropy, producing a ``checkpoint.pt`` that
``train_rwpp --init-checkpoint`` can then fine-tune with PPO in the real game.

The resulting artifact is *not* a hand-written policy: it is a neural network
that has to generalize the teacher's target selection from observations.

Example::

    $env:PYTHONPATH = "src"
    python scripts/bc_warmstart_rwpp.py `
      --runtime "%TEMP%\\ironzero-rwpp-runtime-1.15" --java-home "E:\\jdk25copy" `
      --episodes 24 --envs 3 --transitions 6000 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig  # noqa: E402
from ironzero.rwpp_env import RwppGameConfig  # noqa: E402
from ironzero.training.policy import MASK_KEYS, MaskedActorCritic  # noqa: E402
from ironzero.training.vec_rwpp import VecRwppEnv  # noqa: E402

BROADCAST = 4
STRATEGIES = ("enemy_centroid", "nearest", "lowhp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC warm start for IronZero25 PPO")
    parser.add_argument("--runtime", default=str(REPO_ROOT / ".runtime" / "rwpp-runtime-1.15"))
    parser.add_argument("--java-home", default=r"E:\jdk25copy")
    parser.add_argument("--map", default="maps/ironzero/IronZero25.tmx")
    parser.add_argument("--map-name", default="IronZero25")
    parser.add_argument("--map-width", type=float, default=2700.0)
    parser.add_argument("--map-height", type=float, default=2500.0)
    parser.add_argument("--units-per-team", type=int, default=25)
    parser.add_argument("--max-entities", type=int, default=25)
    parser.add_argument("--target-slots", type=int, default=26)
    parser.add_argument("--terrain", default="configs/maps/IronZero25.terrain.json")
    parser.add_argument("--episode-ticks", type=int, default=4800)
    parser.add_argument("--frame-ms", type=int, default=1)
    parser.add_argument("--vision-radius", type=float, default=0.0)
    parser.add_argument("--envs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--transitions", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port-base", type=int, default=5480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/bc_ironzero25")
    parser.add_argument("--win-mode", default="annihilation")
    parser.add_argument("--unlock-ai", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_game(args: argparse.Namespace) -> tuple[RwppGameConfig, MicroDuelConfig]:
    terrain = Path(args.terrain) if args.terrain else None
    micro = replace(
        DEFAULT_CONFIG,
        map_width=float(args.map_width),
        map_height=float(args.map_height),
        episode_ticks=int(args.episode_ticks),
        units_per_team=int(args.units_per_team),
        max_entities=int(args.max_entities),
        target_slots=int(args.target_slots),
        vision_radius=float(args.vision_radius),
        map_channels=4 if terrain is not None else 3,
        terrain_path=str(terrain) if terrain is not None else None,
    )
    game = RwppGameConfig(
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
        terrain_path=terrain,
        win_mode=str(args.win_mode or ""),
        unlock_ai=bool(args.unlock_ai),
        frame_ms=max(0, int(args.frame_ms)),
        vision_radius=float(args.vision_radius),
    )
    return game, micro


def _centroid(units: list[dict[str, Any]]) -> tuple[float, float]:
    if not units:
        return (0.0, 0.0)
    return (
        sum(float(unit.get("x") or 0.0) for unit in units) / len(units),
        sum(float(unit.get("y") or 0.0) for unit in units) / len(units),
    )


def teacher_action(state: dict[str, Any], micro: MicroDuelConfig, strategy: str) -> np.ndarray:
    enemies = state.get("visible_enemy_entities") or []
    if not enemies:
        gx = _grid(micro.map_width, micro.position_grid)
        gy = _grid(micro.map_height, micro.position_grid)
        return np.array([1, BROADCAST, gx, gy, 0], dtype=np.int64)
    own_cx, own_cy = _centroid(state.get("own_entities") or [])
    if strategy == "nearest":
        target = min(enemies, key=lambda u: (float(u["x"]) - own_cx) ** 2 + (float(u["y"]) - own_cy) ** 2)
    elif strategy == "lowhp":
        target = min(
            enemies,
            key=lambda u: (
                float(u.get("hp") or 0.0),
                (float(u["x"]) - own_cx) ** 2 + (float(u["y"]) - own_cy) ** 2,
            ),
        )
    else:
        ex, ey = _centroid(enemies)
        target = min(enemies, key=lambda u: (float(u["x"]) - ex) ** 2 + (float(u["y"]) - ey) ** 2)
    return np.array([2, BROADCAST, 0, 0, enemies.index(target)], dtype=np.int64)


def _grid(size: float, grid: int) -> int:
    return max(0, min(grid - 1, int(round(0.5 * (grid - 1)))))


def collect(args: argparse.Namespace, game: RwppGameConfig, micro: MicroDuelConfig) -> dict[str, Any]:
    vec = VecRwppEnv(
        int(args.envs),
        game=game,
        port_base=int(args.port_base),
        base_seed=int(args.seed),
        micro_config=micro,
        episode_prefix="bc-" + str(int(time.time()) % 100000),
        persistent=True,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[dict[str, np.ndarray]] = []
    started = time.time()
    try:
        obs, mask_list, _infos = vec.reset()
        decision = 0
        while len(observations) < int(args.transitions):
            batch_actions = np.zeros((vec.num_envs, len(micro.action_nvec)), dtype=np.int64)
            for index, env in enumerate(vec.envs):
                strategy = STRATEGIES[(decision + index) % len(STRATEGIES)]
                batch_actions[index] = teacher_action(env.state, micro, strategy)
                observations.append(np.asarray(obs[index], dtype=np.float32))
                actions.append(batch_actions[index].copy())
                masks.append(
                    {key: np.asarray(mask_list[index][key], dtype=bool).copy() for key in MASK_KEYS}
                )
            batch = vec.step(batch_actions)
            obs = batch.observations
            mask_list = vec.masks
            decision += 1
            if decision % 200 == 0:
                print(
                    json.dumps(
                        {
                            "decisions": int(decision),
                            "transitions": len(observations),
                            "completed_episodes": len(vec.completed_episodes),
                            "elapsed_s": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        completed = list(vec.completed_episodes)
    finally:
        vec.close()
    return {
        "observations": np.stack(observations),
        "actions": np.stack(actions),
        "masks": masks,
        "completed_episodes": completed,
        "collect_seconds": round(time.time() - started, 1),
    }


def train_bc(
    dataset: dict[str, Any],
    micro: MicroDuelConfig,
    args: argparse.Namespace,
) -> tuple[MaskedActorCritic, dict[str, Any]]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    mask_arrays = {
        key: torch.as_tensor(np.stack([mask[key] for mask in dataset["masks"]]), dtype=torch.bool, device=device)
        for key in MASK_KEYS
    }
    total = observations.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    indices = torch.randperm(total, generator=generator)
    split = max(1, int(total * 0.9))
    train_idx = indices[:split].to(device)
    eval_idx = indices[split:].to(device)
    policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec, hidden=int(args.hidden)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    batch_size = 512
    stats = {"loss": 0.0, "batches": 0}
    for epoch in range(int(args.epochs)):
        policy.train()
        shuffled = train_idx[torch.randperm(train_idx.numel(), device="cpu").to(device)]
        epoch_loss = 0.0
        for start in range(0, shuffled.numel(), batch_size):
            batch = shuffled[start : start + batch_size]
            logprob, entropy, _value = policy.evaluate(
                observations[batch],
                {key: mask_arrays[key][batch] for key in MASK_KEYS},
                actions[batch],
            )
            loss = -logprob.mean() - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            epoch_loss += float(loss.item())
            stats["batches"] += 1
        stats["loss"] = epoch_loss / max(1, (shuffled.numel() + batch_size - 1) // batch_size)
        if epoch % 4 == 0 or epoch == int(args.epochs) - 1:
            print(json.dumps({"epoch": epoch, "loss": round(stats["loss"], 4)}, ensure_ascii=False), flush=True)
    agreement = _teacher_agreement(policy, observations[eval_idx], actions[eval_idx], mask_arrays, eval_idx, micro)
    return policy, {**stats, "train_samples": int(train_idx.numel()), "eval_samples": int(eval_idx.numel()), **agreement}


@torch.no_grad()
def _teacher_agreement(
    policy: MaskedActorCritic,
    eval_obs: torch.Tensor,
    eval_actions: torch.Tensor,
    mask_arrays: dict[str, torch.Tensor],
    eval_idx: torch.Tensor,
    micro: MicroDuelConfig,
) -> dict[str, Any]:
    policy.eval()
    logits, _ = policy.forward(eval_obs)
    matched_total = 0
    active_total = 0
    per_head: dict[str, float] = {}
    for head_index, key in enumerate(MASK_KEYS):
        mask = mask_arrays[key][eval_idx]
        predicted = torch.argmax(logits[head_index].masked_fill(~mask, -1e9), dim=-1)
        match = predicted == eval_actions[:, head_index]
        per_head[key] = round(float(match.float().mean().item()), 4)
    # Active-head agreement, mirroring the policy's hierarchical log-prob.
    kinds = eval_actions[:, 0]
    active = torch.zeros_like(eval_actions, dtype=torch.bool)
    active[:, 0] = True
    active[:, 1] = True
    active[:, 2] = kinds <= 1
    active[:, 3] = kinds <= 1
    active[:, 4] = kinds == 2
    for head_index in range(eval_actions.shape[1]):
        predicted = torch.argmax(
            logits[head_index].masked_fill(~mask_arrays[MASK_KEYS[head_index]][eval_idx], -1e9), dim=-1
        )
        match = (predicted == eval_actions[:, head_index]) & active[:, head_index]
        matched_total += int(match.sum().item())
        active_total += int(active[:, head_index].sum().item())
    return {
        "active_head_agreement": round(matched_total / max(1, active_total), 4),
        "per_head_agreement": per_head,
        "obs_dim": int(micro.flat_obs_dim),
        "action_heads": list(micro.action_nvec),
    }


def main() -> int:
    args = parse_args()
    game, micro = build_game(args)
    print(
        json.dumps(
            {
                "mode": "bc-warmstart",
                "obs_dim": micro.flat_obs_dim,
                "action_heads": list(micro.action_nvec),
                "units_per_team": micro.units_per_team,
                "map": args.map,
                "envs": args.envs,
                "transitions": args.transitions,
                "out": args.out,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dataset = collect(args, game, micro)
    print(
        json.dumps(
            {
                "collected": int(dataset["observations"].shape[0]),
                "completed_episodes": len(dataset["completed_episodes"]),
                "collect_seconds": dataset["collect_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    policy, report = train_bc(dataset, micro, args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "micro_config": asdict(micro),
            "ppo_config": {"hidden": int(args.hidden)},
            "update": 0,
            "metrics": [],
            "bc": report,
        },
        checkpoint,
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "map": args.map,
            "terrain": args.terrain,
            "completed_results": [
                episode.get("result") for episode in dataset["completed_episodes"]
            ],
        }
    )
    (out_dir / "bc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(checkpoint), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
