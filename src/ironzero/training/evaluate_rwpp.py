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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from ..micro_config import MicroDuelConfig
from ..rwpp_env import RwppDuelEnv, RwppGameConfig
from .evaluate import summarize
from .policy import masks_to_tensors


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
) -> dict[str, Any]:
    """Run ``episodes`` full real-game episodes with the policy in the loop.

    Episodes run sequentially on one env instance (a fresh game process per
    reset). The returned dict is :func:`~ironzero.training.evaluate.summarize`
    augmented with ``source``, ``game``, ``deterministic``, the Wilson 95%
    intervals and ``episode_results`` (one record per game).
    """

    micro = micro_config or game.micro_config()
    device = torch.device(device)
    # Evaluation must see the same fog the checkpoint was trained with even when
    # the caller forgot to copy it into the game config (e.g. a CLI built from
    # checkpoint metadata).
    if float(micro.vision_radius or 0.0) > 0.0 and float(game.vision_radius or 0.0) <= 0.0:
        game = replace(game, vision_radius=float(micro.vision_radius))
    policy.eval()
    results: list[str] = []
    returns: list[float] = []
    own_hp: list[float] = []
    enemy_hp: list[float] = []
    decisions: list[int] = []
    episode_results: list[dict[str, Any]] = []
    started = time.time()
    env = RwppDuelEnv(
        game,
        port=port_base,
        seed=seed,
        micro_config=micro,
        episode_prefix="rwpp-eval",
    )
    try:
        for episode in range(int(episodes)):
            episode_seed = seed + episode
            obs, info = env.reset(seed=episode_seed)
            total = 0.0
            steps = 0
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
            evaluation = final.get("eval") or {}
            result = str(final.get("result"))
            record = {
                "episode": int(episode),
                "seed": int(episode_seed),
                "result": result,
                "bridge_result": final.get("bridge_result"),
                "return": float(total),
                "decisions": int(steps),
                "ticks": int(final.get("tick", 0)),
                "own_hp_frac": float(evaluation.get("own_hp_frac", 0.0)),
                "enemy_hp_frac": float(evaluation.get("enemy_hp_frac", 0.0)),
                "own_alive": final.get("own_alive"),
                "enemy_alive": final.get("enemy_alive"),
            }
            results.append(result)
            returns.append(float(total))
            own_hp.append(float(record["own_hp_frac"]))
            enemy_hp.append(float(record["enemy_hp_frac"]))
            decisions.append(int(steps))
            episode_results.append(record)
            if on_episode is not None:
                on_episode(record)
    finally:
        env.close()
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
