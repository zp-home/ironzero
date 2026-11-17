# SPDX-License-Identifier: Apache-2.0
"""Pydantic request bodies for the platform backend.

Field names, defaults and constraints are part of the public API contract: the
dashboard posts these payloads verbatim, so nothing here may be renamed or
retuned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    # Wall-clock ms per simulated frame. Measured on 6000 real ticks with the
    # delta pinned (`--physics-ms 16`): 16 ms = 1x (real time), 2 ms = 6.5x,
    # 1 ms = 10.7x, 0 ms = 26.5x (no sleep at all). 0 is legal *only* because the
    # delta is pinned; `frame_ms` then paces the wall clock and cannot change
    # what an episode simulates.
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
    # --- acceleration (host forwards both verbatim) ---------------------
    # Bound enforced in ``routers/rwpp.py`` as a 400 with the shared wording, so
    # the dashboard sees the same message a direct curl to the host would.
    fast_forward: int = Field(default=1, description="engine gameSpeed 1..32; 1 = off")
    lockstep: bool | None = Field(default=None, description="bridge lockstep; NULL = host decides per runner")
    lockstep_timeout_ms: int = Field(default=30000, ge=1000, le=600000)


class RwppPpoRequest(BaseModel):
    """Strategy-in-the-loop PPO training on real games.

    Every field here is forwarded verbatim to the host-side service
    (``scripts/rwpp_service/launch.py``). ``extra="forbid"`` is deliberate: when
    this model was missing the arena / AI-teacher / physics fields, pydantic's
    default ``ignore`` policy *silently dropped* them, so the dashboard could ask
    for ``ai_pilot`` or ``physics_ms``, get HTTP 200, and train something else
    entirely. An unknown key is now a 422 at the boundary instead of a mystery.
    """

    model_config = ConfigDict(extra="forbid")

    updates: int = Field(default=4, ge=1, le=500)
    envs: int = Field(default=2, ge=1, le=16)
    rollout_steps: int = Field(default=20, ge=1, le=512)
    # 0 = derive the cap from the map diagonal and player count
    # (``recommended_episode_ticks``). A fixed small constant cut every
    # original-map episode off before the engine could decide anything: measured,
    # the earliest verdict ever observed on ``[p2]Lake (2p)`` was tick 46972 while
    # the form used to send 4800/3600, so ``terminal`` was structurally 0.
    episode_ticks: int = Field(default=0, ge=0, le=216000)
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

    # --- arena preset, resolved host-side -------------------------------
    arena: bool | None = Field(default=None, description="force the arena/preset path (inferred from the map name otherwise)")
    max_entities: int | None = Field(default=None, ge=1, le=64, description="entity slots per team block in the observation")
    target_slots: int | None = Field(default=None, ge=1, le=65, description="target head size; 0/NULL = max_entities + 1")
    terrain: str | None = Field(default=None, max_length=512, description="terrain passability JSON (adds a 4th map channel)")
    win_mode: str | None = Field(default=None, max_length=32, description="bridge win condition, e.g. annihilation")
    unlock_ai: bool | None = Field(default=None, description="let the built-in AI play (arena needs this)")

    # --- pacing and adjudication ---------------------------------------
    # physics_ms pins the simulated delta (16 = real time); frame_ms then only
    # paces the wall clock. The trainer's own default has moved more than once,
    # so a web-launched run must be able to pin it instead of inheriting.
    physics_ms: int | None = Field(default=None, ge=0, le=100, description="engine simulated ms per frame; 16 = real time")
    # Engine-native acceleration. `fast_forward=N` makes one tick advance N times
    # the game time (the engine clamps delta to 3x, so N is the only way past 3x);
    # a match then needs 1/N the ticks, at the cost of decision granularity.
    #
    # No `ge`/`le` pair here on purpose: a pydantic bound answers **422** with
    # pydantic's own sentence, while the contract (and the host service, and
    # therefore a direct curl) answers **400 `fast_forward must be 1..32`**.
    # `routers/rwpp.py::_check_acceleration` produces that, so every entry point
    # reports the same status and text.
    fast_forward: int = Field(default=1, description="engine gameSpeed 1..32; 1 = off, N = Nx game time per tick")
    # Lockstep on by default for training: the policy must answer every
    # observation, otherwise an unpaced engine (frame_ms=0) drops decisions the
    # policy never made (measured: only ~14% of decisions landed at frame_ms=1
    # with 8 envs before this existed).
    lockstep: bool | None = Field(default=None, description="bridge lockstep; NULL = host default (on for /ppo)")
    lockstep_timeout_ms: int = Field(default=30000, ge=1000, le=600000, description="max lockstep wait before the bridge falls back to free running")
    timeout_penalty: float | None = Field(default=None, ge=0.0, le=10.0, description="penalty for hitting the tick cap undecided")
    num_teams: int | None = Field(default=None, ge=1, le=16, description="team count used to derive the episode tick cap")

    # --- built-in-AI teacher / opponent --------------------------------
    ai_pilot: bool | None = Field(default=None, description="hand the controlled team to the built-in AI (BC teacher)")
    record_ai_orders: bool | None = Field(default=None, description="record the AI pilot's orders as BC labels")
    ai_difficulty: int | None = Field(default=None, ge=-2, le=3, description="pin every AI team's difficulty")
    enemy_ai_difficulty: int | None = Field(default=None, ge=-2, le=3, description="pin only the enemy teams' difficulty")
