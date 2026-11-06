# SPDX-License-Identifier: Apache-2.0
"""PPO with GAE and per-head action masks for the micro-duel task."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .evaluate import evaluate_policy, evaluate_random
from .policy import MASK_KEYS, MaskedActorCritic, masks_to_tensors
from .vec_env import VecMicroEnv

@dataclass
class PPOConfig:
    total_updates: int = 120
    num_envs: int = 8
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 256
    device: str = "auto"
    seed: int = 0
    eval_every: int = 20
    eval_episodes: int = 24
    opponent: str = "scripted"
    out_dir: str = "runs/micro_duel"
    log_every: int = 5

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Rollout:
    observations: torch.Tensor  # (T, B, D)
    actions: torch.Tensor  # (T, B, 5)
    logprobs: torch.Tensor  # (T, B)
    values: torch.Tensor  # (T, B)
    rewards: torch.Tensor  # (T, B)
    dones: torch.Tensor  # (T, B)
    masks: dict[str, torch.Tensor]  # (T, B, n)
    last_values: torch.Tensor  # (B,)


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig | None = None,
        *,
        micro_config: MicroDuelConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.micro = micro_config or DEFAULT_CONFIG
        self.device = self.config.resolved_device()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.vec = VecMicroEnv(
            self.config.num_envs,
            config=self.micro,
            opponent=self.config.opponent,
            base_seed=self.config.seed,
        )
        self.policy = MaskedActorCritic(
            self.micro.flat_obs_dim,
            nvec=self.micro.action_nvec,
            hidden=self.config.hidden,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)
        self.history: list[dict[str, Any]] = []
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect(self) -> Rollout:
        cfg = self.config
        obs, masks, _infos = self.vec.reset()
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        mask_history: dict[str, list[torch.Tensor]] = {key: [] for key in MASK_KEYS}
        for _ in range(cfg.rollout_steps):
            mask_tensors = masks_to_tensors(masks, self.device)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, logprob, value = self.policy.act(obs_tensor, mask_tensors)
            batch = self.vec.step(action.cpu().numpy())
            observations.append(obs_tensor.cpu())
            actions.append(action.cpu())
            logprobs.append(logprob.cpu())
            values.append(value.cpu())
            rewards.append(torch.as_tensor(batch.rewards, dtype=torch.float32))
            dones.append(torch.as_tensor(batch.terminated | batch.truncated, dtype=torch.float32))
            for key in MASK_KEYS:
                mask_history[key].append(mask_tensors[key].cpu())
            obs, masks = batch.observations, self.vec.masks
        with torch.no_grad():
            last_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_masks = masks_to_tensors(masks, self.device)
            _, last_values = self.policy.forward(last_obs)
            _ = last_masks
        return Rollout(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            logprobs=torch.stack(logprobs),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            masks={key: torch.stack(mask_history[key]) for key in MASK_KEYS},
            last_values=last_values.cpu(),
        )

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------
    def compute_gae(self, rollout: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        rewards = rollout.rewards
        values = rollout.values
        dones = rollout.dones
        steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(rewards.shape[1], dtype=torch.float32)
        for step in reversed(range(steps)):
            if step == steps - 1:
                next_value = rollout.last_values
            else:
                next_value = values[step + 1]
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + cfg.gamma * next_value * non_terminal - values[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values
        return returns, advantages

    def update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self.config
        returns, advantages = self.compute_gae(rollout)
        steps, num_envs = advantages.shape
        flat_obs = rollout.observations.reshape(-1, self.micro.flat_obs_dim)
        flat_actions = rollout.actions.reshape(-1, len(self.micro.action_nvec))
        flat_logprobs = rollout.logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_masks = {
            key: value.reshape(steps * num_envs, *value.shape[2:]) for key, value in rollout.masks.items()
        }
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        total = flat_obs.shape[0]
        indices = np.arange(total)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0, "updates": 0}
        for _epoch in range(cfg.epochs):
            np.random.shuffle(indices)
            for start in range(0, total, cfg.minibatch_size):
                batch = indices[start : start + cfg.minibatch_size]
                obs_batch = flat_obs[batch].to(self.device)
                actions_batch = flat_actions[batch].to(self.device)
                old_logprob = flat_logprobs[batch].to(self.device)
                advantages_batch = flat_advantages[batch].to(self.device)
                returns_batch = flat_returns[batch].to(self.device)
                masks_batch = {key: flat_masks[key][batch].to(self.device) for key in MASK_KEYS}
                new_logprob, entropy, value = self.policy.evaluate(obs_batch, masks_batch, actions_batch)
                ratio = torch.exp(new_logprob - old_logprob)
                surrogate_1 = ratio * advantages_batch
                surrogate_2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = 0.5 * ((value - returns_batch) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_mean.item())
                stats["clip_fraction"] += float(((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item())
                stats["updates"] += 1
        divisor = max(1, stats["updates"])
        return {key: value / divisor for key, value in stats.items() if key != "updates"}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        cfg = self.config
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        random_baseline = evaluate_random(
            episodes=cfg.eval_episodes, config=self.micro, opponent=cfg.opponent
        )
        print(f"[baseline] random: {json.dumps(random_baseline, ensure_ascii=False)}")
        metrics: dict[str, Any] = {
            "config": asdict(cfg),
            "device": str(self.device),
            "random_baseline": random_baseline,
            "updates": [],
            "evaluations": [],
        }
        for update in range(1, cfg.total_updates + 1):
            rollout = self.collect()
            update_stats = self.update(rollout)
            completed = self.vec.drain_completed()
            returns = [episode["return"] for episode in completed]
            results = [episode.get("result") for episode in completed]
            record = {
                "update": update,
                **update_stats,
                "episodes": len(completed),
                "mean_return": float(np.mean(returns)) if returns else 0.0,
                "win_rate": (
                    sum(1 for r in results if r == "victory") / len(results) if results else 0.0
                ),
                "elapsed_s": round(time.time() - self.start_time, 2),
            }
            metrics["updates"].append(record)
            if update % cfg.log_every == 0 or update == 1:
                print(
                    "[train] update {update:03d} return={mean_return:+.3f} win={win_rate:.2f} "
                    "pi={policy_loss:.3f} v={value_loss:.3f} H={entropy:.3f} t={elapsed_s}s".format(
                        update=update,
                        mean_return=record["mean_return"],
                        win_rate=record["win_rate"],
                        policy_loss=record["policy_loss"],
                        value_loss=record["value_loss"],
                        entropy=record["entropy"],
                        elapsed_s=record["elapsed_s"],
                    )
                )
            if update % cfg.eval_every == 0 or update == cfg.total_updates:
                evaluation = evaluate_policy(
                    self.policy,
                    episodes=cfg.eval_episodes,
                    config=self.micro,
                    opponent=cfg.opponent,
                    seed=50_000 + update,
                    device=self.device,
                )
                evaluations = {"update": update, **evaluation}
                metrics["evaluations"].append(evaluations)
                print(
                    "[eval ] update {update:03d} win={win_rate:.2f} loss={loss_rate:.2f} "
                    "timeout={timeout_rate:.2f} return={mean_return:+.3f} own_hp={mean_own_hp_frac:.2f}".format(
                        update=update, **evaluation
                    )
                )
                self.save(out_dir, update, metrics)
        final_eval = metrics["evaluations"][-1] if metrics["evaluations"] else None
        metrics["final_eval"] = final_eval
        metrics["total_seconds"] = round(time.time() - self.start_time, 2)
        self.save(out_dir, cfg.total_updates, metrics)
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        return metrics

    # ------------------------------------------------------------------
    def save(self, out_dir: Path, update: int, metrics: dict[str, Any]) -> Path:
        path = out_dir / "checkpoint.pt"
        torch.save(
            {
                "model": self.policy.state_dict(),
                "micro_config": asdict(self.micro),
                "ppo_config": asdict(self.config),
                "update": update,
                "metrics": metrics.get("evaluations", []),
            },
            path,
        )
        best = out_dir / "checkpoint_latest.pt"
        torch.save(
            {
                "model": self.policy.state_dict(),
                "micro_config": asdict(self.micro),
                "ppo_config": asdict(self.config),
                "update": update,
            },
            best,
        )
        return path


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> tuple[MaskedActorCritic, MicroDuelConfig]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    micro = MicroDuelConfig(**payload["micro_config"])
    policy = MaskedActorCritic(micro.flat_obs_dim, nvec=micro.action_nvec)
    policy.load_state_dict(payload["model"])
    policy.to(torch.device(device))
    policy.eval()
    return policy, micro


__all__ = ["PPOConfig", "PPOTrainer", "Rollout", "load_policy"]
