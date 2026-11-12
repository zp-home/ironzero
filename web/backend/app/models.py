# SPDX-License-Identifier: Apache-2.0
"""Pydantic request bodies for the platform backend.

Field names, defaults and constraints are part of the public API contract: the
dashboard posts these payloads verbatim, so nothing here may be renamed or
retuned.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    name: str = Field(default="", max_length=80)
    updates: int = Field(default=200, ge=1, le=5000)
    envs: int = Field(default=16, ge=1, le=64)
    rollout_steps: int = Field(default=64, ge=8, le=512)
    opponent: str = Field(default="scripted", pattern="^(scripted|idle|random)$")
    seed: int = Field(default=0, ge=0)
    eval_every: int = Field(default=25, ge=1, le=1000)
    eval_episodes: int = Field(default=16, ge=1, le=128)
    lr: float = Field(default=3e-4, gt=0, le=1e-1)
    entropy: float = Field(default=0.01, ge=0.0, le=1.0)
    epochs: int = Field(default=4, ge=1, le=16)


class InferRequest(BaseModel):
    checkpoint: str = Field(default="", max_length=512)
    episodes: int = Field(default=3, ge=1, le=50)
    opponent: str = Field(default="scripted", pattern="^(scripted|idle|random)$")
    seed: int = Field(default=42, ge=0)
    mode: str = Field(default="greedy", pattern="^(greedy|sample|random|scripted)$")
    max_decisions: int = Field(default=400, ge=10, le=2000)
    map_id: str = Field(default="", max_length=256)
    sim_mode: str = Field(default="arena", pattern="^(arena|terrain)$")
    fog: bool = Field(default=False)
    vision_radius: float = Field(default=260.0, gt=0.0, le=5000.0)


class RwppRecordRequest(BaseModel):
    ticks: int = Field(default=3600, ge=120, le=216000)
    policy: str = Field(default="scripted", pattern="^(scripted|none|model|ppo)$")
    checkpoint: str | None = Field(default=None, max_length=512)
    # 2 ms/tick is ~8x real time; the policy only decides every 10 ticks, so the
    # old 16 ms (= real time) default just made every episode take minutes.
    frame_ms: int = Field(default=2, ge=0, le=16)
    map_name: str | None = Field(default=None, max_length=128)
    map_path: str | None = Field(default=None, max_length=256)
    map_width: float = Field(default=2600.0, gt=0.0, le=20000.0)
    map_height: float = Field(default=2600.0, gt=0.0, le=20000.0)
    episodes: int = Field(default=1, ge=1, le=50)
    # Concurrent headless game processes. The host-side service is the real
    # authority (``--max-parallel``, default 10, enforced with a 400 for a
    # larger value); this bound only has to be wide enough not to reject a
    # legitimate request before it reaches that check.
    concurrency: int = Field(default=1, ge=1, le=16)
    vision_radius: float = Field(default=0.0, ge=0.0, le=2000.0)


class RwppPpoRequest(BaseModel):
    """Strategy-in-the-loop PPO training on real games."""

    updates: int = Field(default=4, ge=1, le=500)
    envs: int = Field(default=2, ge=1, le=16)
    rollout_steps: int = Field(default=20, ge=1, le=512)
    episode_ticks: int = Field(default=3600, ge=120, le=216000)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    map_name: str = Field(default="Artillery", max_length=128)
    map_path: str = Field(default="maps/challenge/Artillery.tmx", max_length=256)
    map_width: float = Field(default=2700.0, gt=0.0, le=20000.0)
    map_height: float = Field(default=2500.0, gt=0.0, le=20000.0)
    frame_ms: int = Field(default=2, ge=0, le=16, description="wall-clock ms per simulated frame; 2 is ~8x faster than real time")
    vision_radius: float = Field(default=260.0, ge=0.0, le=2000.0, description="observation fog radius; <=0 disables fog")
    eval_every: int = Field(default=0, ge=0, le=500, description="0 = evaluate only on the final update")
    eval_episodes: int = Field(default=1, ge=0, le=16)
    use_llm: bool = Field(default=False, description="enable the LLM strategic layer (rule fallback if unconfigured)")
    units_per_team: int | None = Field(default=None, ge=1, le=25, description="双方坦克数量（1..25）")
    init_checkpoint: str | None = Field(default=None, max_length=512, description="初始策略 Checkpoint 路径")
    init_mode: str | None = Field(default=None, max_length=64, description="策略初始化模式: warmstart / checkpoint / scratch")
