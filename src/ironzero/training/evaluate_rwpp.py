# SPDX-License-Identifier: Apache-2.0
"""Evaluation of a policy inside real Rusted Warfare 1.15 games.

Evaluation is the expensive part of the loop (one fresh headless game per
episode), so the report keeps the *per-episode* outcome, the Wilson 95%
confidence interval of the win rate, and - when a path is given - writes the
whole report as JSON under ``runs/`` for offline comparison.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from ..micro_config import MicroDuelConfig
from ..rwpp_env import RwppDuelEnv, RwppGameConfig
from .evaluate import summarize
from .policy import masks_to_tensors


def _run_one_episode(
    *,
    policy: torch.nn.Module,
    game: RwppGameConfig,
    micro: MicroDuelConfig,
    port: int,
    seed: int,
    episode: int,
    device: torch.device,
    deterministic: bool,
) -> dict[str, Any]:
    """Play one full episode in its own game process. Thread-safe by isolation."""

    env = RwppDuelEnv(
        game,
        port=port,
        seed=seed,
        micro_config=micro,
        episode_prefix="rwpp-eval",
    )
    total = 0.0
    steps = 0
    try:
        obs, info = env.reset(seed=seed)
        done = False
        final = info
        while not done:
            mask_tensors = masks_to_tensors([final["masks"]], device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                actions, _, _ = policy.act(
                    obs_tensor, mask_tensors, deterministic=deterministic
                )
            action = actions[0].cpu().numpy()
            obs, reward, terminated, truncated, final = env.step(action)
            total += float(reward)
            steps += 1
            done = bool(terminated or truncated)
    finally:
        env.close()

    evaluation = final.get("eval") or {}
    return {
        "episode": int(episode),
        "seed": int(seed),
        "result": str(final.get("result")),
        "bridge_result": final.get("bridge_result"),
        "return": float(total),
        "decisions": int(steps),
        "ticks": int(final.get("tick", 0)),
        "own_hp_frac": float(evaluation.get("own_hp_frac", 0.0)),
        "enemy_hp_frac": float(evaluation.get("enemy_hp_frac", 0.0)),
        "own_alive": final.get("own_alive"),
        "enemy_alive": final.get("enemy_alive"),
    }


def evaluate_rwpp_policy(
    policy: torch.nn.Module,
    *,
    game: RwppGameConfig,
    episodes: int = 2,
    micro_config: MicroDuelConfig | None = None,
    port_base: int = 5200,
    seed: int = 70_000,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    report_path: str | Path | None = None,
    update: int | None = None,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
    parallel: int | None = None,
) -> dict[str, Any]:
    """Run ``episodes`` full real-game episodes with the policy in the loop.

    Episodes run **in parallel by default**: one fresh game process per episode,
    each on its own bridge port (``port_base + episode_index``). Evaluation used
    to be strictly sequential, which made a 4-episode pass take ~112 s while the
    10 training updates it gated took ~15 s -- so three of every four background
    evaluations were dropped as "still running" and the reported curve was
    mostly interpolated. Set ``parallel=1`` for the old sequential behaviour when
    the machine cannot host several game processes at once.

    The returned dict is :func:`~ironzero.training.evaluate.summarize`
    augmented with ``source``, ``game``, ``deterministic``, the Wilson 95%
    intervals and ``episode_results`` (one record per game, in episode order).
    """

    micro = micro_config or game.micro_config()
    device = torch.device(device)
    # Evaluation must see the same fog the checkpoint was trained with even when
    # the caller forgot to copy it into the game config (e.g. a CLI built from
    # checkpoint metadata).
    if float(micro.vision_radius or 0.0) > 0.0 and float(game.vision_radius or 0.0) <= 0.0:
        game = replace(game, vision_radius=float(micro.vision_radius))
    policy.eval()
    started = time.time()
    total_episodes = max(1, int(episodes))
    # One fresh game process per episode, all in flight at once. Running the
    # episodes sequentially made a 4-episode evaluation take ~112 s while the
    # 10 training updates it was gating took ~15 s, so 3 of every 4 background
    # evaluations were dropped as "still running". Each worker owns a distinct
    # bridge port: two games on one port die with BindException.
    workers = min(total_episodes, max(1, int(parallel or total_episodes)))
    seeds = [int(seed) + index for index in range(total_episodes)]
    records: list[dict[str, Any] | None] = [None] * total_episodes

    def _job(index: int) -> None:
        record = _run_one_episode(
            policy=policy,
            game=game,
            micro=micro,
            port=int(port_base) + index,
            seed=seeds[index],
            episode=index,
            device=device,
            deterministic=deterministic,
        )
        records[index] = record
        if on_episode is not None:
            on_episode(record)

    if workers <= 1:
        for index in range(total_episodes):
            _job(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Surface the first worker exception instead of silently returning a
            # partial report: a half-played evaluation must not look like a win.
            for future in [pool.submit(_job, index) for index in range(total_episodes)]:
                future.result()

    episode_results = [record for record in records if record is not None]
    results = [str(record["result"]) for record in episode_results]
    returns = [float(record["return"]) for record in episode_results]
    own_hp = [float(record["own_hp_frac"]) for record in episode_results]
    enemy_hp = [float(record["enemy_hp_frac"]) for record in episode_results]
    decisions = [int(record["decisions"]) for record in episode_results]
    evaluation = summarize(results, returns, own_hp, enemy_hp, decisions)
    evaluation.update(
        {
            "source": "rwpp",
            "deterministic": bool(deterministic),
            "map": game.map_name,
            "map_path": game.map_relative,
            "vision_radius": float(game.vision_radius or 0.0),
            "frame_ms": int(game.frame_ms),
            "elapsed_s": round(time.time() - started, 2),
            "episode_results": episode_results,
        }
    )
    if update is not None:
        evaluation["update"] = int(update)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        evaluation["report_path"] = str(path)
    return evaluation


__all__ = ["evaluate_rwpp_policy"]
