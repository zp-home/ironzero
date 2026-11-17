# SPDX-License-Identifier: Apache-2.0
"""由 HTTP payload 创建任务：``/infer`` ``/train`` ``/ppo`` 的准入与装配。

``StartMixin`` 混入 :class:`rwpp_service.handler.Handler`，方法体与拆分前
``Handler._start_ppo`` / ``Handler._start`` / ``Handler._ppo_running`` 等价，
只是把字段解析、参数校验、任务记录模板抽成了模块级纯函数。
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .paths import REPO_ROOT
from .paths import _persist_job, _strip_container_prefix
from .maps import DEFAULT_ENEMY_AI_DIFFICULTY, resolve_map_dimensions
from .ppo import resolve_init_state, resolve_preset, run_ppo
from .proc import FAST_FORWARD_MAX, FAST_FORWARD_MIN, LOCKSTEP_TIMEOUT_DEFAULT_MS
from .runners import run_infer, run_train
from . import state
from .state import JOBS, JOBS_LOCK, PORT_SLOTS

__all__ = [
    "StartMixin",
    "acceleration_error",
    "ppo_bounds_error",
    "resolve_acceleration",
    "resolve_map_geometry",
    "resolve_enemy_ai_difficulty",
]

#: ``/ppo`` 的 6 段数值/枚举校验，返回 ``(message, status)`` 或 ``None``。
_PPO_CHECKS = (
    ("frame_ms must be 0..16", lambda v: 0 <= v["frame_ms"] <= 16),
    ("vision_radius must be 0..2000", lambda v: 0.0 <= v["vision_radius"] <= 2000.0),
    ("updates must be 1..500", lambda v: 1 <= v["updates"] <= 500),
    ("envs must be 1..16", lambda v: 1 <= v["envs"] <= 16),
    ("rollout_steps must be 1..512", lambda v: 1 <= v["rollout_steps"] <= 512),
    # 0 means "derive the cap from the map diagonal and player count"
    # (``recommended_episode_ticks``); anything else is an explicit override.
    ("episode_ticks must be 0 or 120..216000",
     lambda v: v["episode_ticks"] == 0 or 120 <= v["episode_ticks"] <= 216000),
)


def _optional_bool(value, default: bool) -> bool:
    """Read a JSON-ish boolean, tolerating ``"false"``/``"0"``/``0``.

    ``bool("false")`` is ``True``, which would silently turn lockstep *on* for a
    caller that sent the string form. The dashboard sends real booleans, but this
    endpoint is also reachable with curl and from scripts.
    """

    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def resolve_acceleration(payload: dict, *, lockstep_default: bool) -> dict:
    """Read the two acceleration switches out of a request payload.

    ``fast_forward`` (1..32, default 1) is the engine-native ``gameSpeed``:
    one tick then advances N times the game time, so a match needs 1/N the
    ticks. It changes fidelity, unlike ``frame_ms``, which is pure pacing once
    ``physics_ms`` pins the delta -- so it stays opt-in and explicit.

    ``lockstep`` makes the bridge block after every observation until the client
    answers, which is what a policy-driven episode wants (no dropped decisions)
    and what a recorder does *not* want (it only sends an order every N
    observations, so blocking on every frame would stall the engine). Hence the
    caller supplies ``lockstep_default``: on for ``/ppo`` and model inference,
    off for recording and collection.
    """

    fast_forward = _optional_int(payload.get("fast_forward"), FAST_FORWARD_MIN)
    if fast_forward is None:
        fast_forward = FAST_FORWARD_MIN
    lockstep_explicit = payload.get("lockstep") not in (None, "")
    lockstep = _optional_bool(payload.get("lockstep"), lockstep_default)
    lockstep_timeout_ms = _optional_int(
        payload.get("lockstep_timeout_ms"), LOCKSTEP_TIMEOUT_DEFAULT_MS
    )
    if lockstep_timeout_ms is None:
        lockstep_timeout_ms = LOCKSTEP_TIMEOUT_DEFAULT_MS
    return {
        "fast_forward": fast_forward,
        "lockstep": lockstep,
        # True when the caller stated ``lockstep`` itself. The distinction
        # matters for /infer: its default depends on which runner the episode
        # needs (model player -> on, scripted recorder -> off), so an absent key
        # must stay "the runner decides" instead of being frozen into a bool.
        "lockstep_explicit": lockstep_explicit,
        "lockstep_timeout_ms": lockstep_timeout_ms,
    }


def acceleration_error(raw_fast_forward) -> str | None:
    """``fast_forward must be 1..32`` when out of range, else ``None``.

    Takes the **raw** payload value (not the coerced one) so a non-numeric value
    is refused too instead of being silently replaced by the default -- that is
    the same class of bug as a dropped request key: the dashboard asks for
    something, gets ``200``, and runs something else.

    Kept out of :func:`ppo_bounds_error` so the PPO bounds contract (and its
    per-field order) is untouched, and so every route shares one message instead
    of each inventing its own wording.
    """

    if raw_fast_forward is None or raw_fast_forward == "":
        return None
    try:
        speed = int(raw_fast_forward)
    except (TypeError, ValueError):
        return "fast_forward must be 1..32"
    if not FAST_FORWARD_MIN <= speed <= FAST_FORWARD_MAX:
        return "fast_forward must be 1..32"
    return None


def _optional_int(value, default: int | None) -> int | None:
    """把 payload 里的可选整数收敛成 ``int``；缺失/空值/非法值一律回退 ``default``。

    ``/ppo`` 的 payload 来自 JS，字段常常是 ``null``、``""`` 或者字符串数字，
    直接 ``int()`` 会抛 ``ValueError`` 把整个请求打成 500。
    """

    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ppo_bounds_error(values: dict) -> str | None:
    """按原顺序校验 PPO 数值区间；``device`` 单独校验以保持原始次序。"""

    for message, check in _PPO_CHECKS:
        if not check(values):
            return message
    return None


def resolve_map_geometry(
    payload: dict,
    map_name,
    map_path,
    *,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """``(width_px, height_px)``：payload 显式给了就用，否则按地图查。

    ``frame_ms`` / ``fast_forward`` 之类的默认值写死无所谓，地图尺寸不行：观测层
    用像素尺寸归一化每个单位坐标，所以在 ``4000x3200`` 的地图上跑
    ``2700x2500`` 会把整场战斗压进地图左三分之一，而且**不会报任何错**。原来
    ``_start_ppo`` 里的 ``2700x2500`` 正是这么来的 —— 它是 Artillery 的尺寸，被
    当成了所有地图的默认值（``Lake`` 真正的 ``2600x2600`` 也被写成了
    ``2600x2500``，y 方向差 100 px）。

    优先级：``payload`` 里的正数 > 地图目录 > ``fallback``。``fallback`` 是为了
    ``IronZero25`` 这类不在 2p 目录里的竞技场图，它们的尺寸由调用方的历史默认
    值决定。非正数（``0``、负数、``"abc"``）一律视为"没给"，因为一个 0 宽的
    地图只会让归一化除零。
    """

    def _positive(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not (number > 0.0) or number != number:  # NaN fails ``==``
            return None
        return number

    explicit_width = _positive(payload.get("map_width"))
    explicit_height = _positive(payload.get("map_height"))
    if explicit_width is not None and explicit_height is not None:
        return (explicit_width, explicit_height)
    resolved_width, resolved_height = resolve_map_dimensions(
        map_path, map_name, fallback=fallback
    )
    return (
        explicit_width if explicit_width is not None else resolved_width,
        explicit_height if explicit_height is not None else resolved_height,
    )


def resolve_enemy_ai_difficulty(payload: dict) -> int | None:
    """对手难度：显式值优先，否则 :data:`DEFAULT_ENEMY_AI_DIFFICULTY`（1）。

    缺省为什么不是 ``None``：``None`` 会让引擎走 ``SettingsEngine.aiDifficulty``，
    那个全局默认是 **0**（正常，x1.00，没有任何经济加成）。也就是说"没选"实际
    等价于"最容易的一档"，而界面上的难度输入框看着像是有值的。1 = 困难
    （x1.40 经济），对手真的会攒兵进攻，策略才有压力可学。

    越界值原样返回，由调用方的 400 校验拒绝（保持既有的错误信息与顺序）。
    """

    value = _optional_int(payload.get("enemy_ai_difficulty"), None)
    if value is None:
        return DEFAULT_ENEMY_AI_DIFFICULTY
    return value


class StartMixin:
    """任务创建路由的共享实现（依赖 host 类提供 ``_send``）。"""

    def _start_ppo(self, payload: dict) -> None:
        updates = int(payload.get("updates", 4))
        envs = int(payload.get("envs", 2))
        rollout_steps = int(payload.get("rollout_steps", 20))
        episode_ticks = int(payload.get("episode_ticks", 0))
        device = str(payload.get("device", "auto"))
        map_name = str(payload.get("map_name") or "Artillery")
        map_path = str(payload.get("map_path") or "maps/challenge/Artillery.tmx")
        # Dimensions are resolved from the map, not defaulted to one map's size.
        # An explicit payload value still wins (a caller may smoke-test a crop);
        # see ``resolve_map_geometry`` for why the constant had to go. This first
        # pass uses the *requested* map; the second one below re-resolves after
        # ``resolve_preset`` in case it redirected the request to another file
        # (the arena path regenerates ``maps/ironzero/IronZero<units>.tmx``).
        map_width, map_height = resolve_map_geometry(
            payload, map_name, map_path, fallback=(2700.0, 2500.0)
        )
        # ``/ppo`` pins ``--physics-ms`` unconditionally, so 0 (unpaced) is legal
        # here as well; the dashboard now sends it explicitly. The *default* stays
        # the historical 2 so a direct caller sees no unrequested behaviour change,
        # and ``_optional_int`` keeps a JSON ``null`` from raising (the old
        # ``int(payload.get(...))`` turned that into a 500).
        frame_ms = _optional_int(payload.get("frame_ms"), 2)
        if frame_ms is None:
            frame_ms = 2
        physics_ms = _optional_int(payload.get("physics_ms"), 16)
        # /ppo trains a policy, so lockstep is on unless asked otherwise: one
        # decision per observation, never a dropped frame.
        accel = resolve_acceleration(payload, lockstep_default=True)
        fast_forward = accel["fast_forward"]
        lockstep = accel["lockstep"]
        lockstep_timeout_ms = accel["lockstep_timeout_ms"]
        vision_radius = float(payload.get("vision_radius", 260.0))
        eval_every = int(payload.get("eval_every", 0))
        eval_episodes = int(payload.get("eval_episodes", 1))
        use_llm = bool(payload.get("use_llm", False))
        ai_pilot = bool(payload.get("ai_pilot", False))
        record_ai_orders = bool(payload.get("record_ai_orders", False))
        ai_difficulty = _optional_int(payload.get("ai_difficulty"), None)
        # Enemy difficulty defaults to 1 (困难, x1.40 income) instead of None.
        # ``None`` means "engine global", whose default is 0 (x1.00, no bonus):
        # omitting the key silently equalled the weakest real opponent while the
        # form still showed a difficulty box. See ``resolve_enemy_ai_difficulty``.
        enemy_ai_difficulty = resolve_enemy_ai_difficulty(payload)
        preset = resolve_preset(payload, map_name, map_path)
        map_name = preset["map_name"]
        map_path = preset["map_path"]
        # ``resolve_preset`` may have redirected the request (the arena path
        # regenerates ``maps/ironzero/IronZero<units>.tmx``), so resolve again
        # against the map that will actually be loaded. An explicit payload
        # dimension still wins inside ``resolve_map_geometry``.
        map_width, map_height = resolve_map_geometry(
            payload, map_name, map_path, fallback=(map_width, map_height)
        )
        arena = preset["arena"]
        units_per_team = preset["units_per_team"]
        max_entities = preset["max_entities"]
        target_slots = preset["target_slots"]
        terrain = preset["terrain"]
        win_mode = preset["win_mode"]
        unlock_ai = preset["unlock_ai"]
        init_state = resolve_init_state(payload, units_per_team)
        init_checkpoint = init_state["checkpoint"]

        error = ppo_bounds_error(
            {
                "frame_ms": frame_ms,
                "vision_radius": vision_radius,
                "updates": updates,
                "envs": envs,
                "rollout_steps": rollout_steps,
                "episode_ticks": episode_ticks,
            }
        )
        if error:
            self._send({"error": error}, 400)
            return
        accel_error = acceleration_error(payload.get("fast_forward"))
        if accel_error:
            self._send({"error": accel_error}, 400)
            return
        if not 0 <= physics_ms <= 100:
            self._send({"error": "physics_ms must be 0..100"}, 400)
            return
        # An arena request whose map could not be prepared must be refused here,
        # with the real reason. Starting it anyway only moves the failure into the
        # trainer subprocess, where all the user ever sees is "failed".
        if preset.get("arena_error"):
            self._send({"error": preset["arena_error"]}, 400)
            return
        if init_state.get("error"):
            self._send({"error": init_state["error"]}, 400)
            return
        if ai_difficulty is not None and not -2 <= ai_difficulty <= 3:
            self._send({"error": "ai_difficulty must be -2..3"}, 400)
            return
        if enemy_ai_difficulty is not None and not -2 <= enemy_ai_difficulty <= 3:
            self._send({"error": "enemy_ai_difficulty must be -2..3"}, 400)
            return
        if device not in {"auto", "cpu", "cuda"}:
            self._send({"error": "device must be auto/cpu/cuda"}, 400)
            return
        with JOBS_LOCK:
            running_ppo = [
                job for job in JOBS.values() if job.get("kind") == "ppo" and job.get("status") == "running"
            ]
            running_real = [
                job
                for job in JOBS.values()
                if job.get("kind") in ("infer", "train") and job.get("status") == "running"
            ]
        if running_ppo:
            running_id = running_ppo[0]["id"]
            self._send({"error": f"当前已有 PPO 训练任务正在进行 ({running_id})，请先停止它再启动新训练", "running_job_id": running_id}, 409)
            return
        if running_real:
            running_id = running_real[0]["id"]
            self._send({"error": f"当前已有实机任务正在进行 ({running_id})，请先停止它或等待完成", "running_job_id": running_id}, 409)
            return
        job_id = f"ppo-{uuid.uuid4().hex[:8]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "kind": "ppo",
                "status": "running",
                "updates_total": updates,
                "updates_done": 0,
                "envs": envs,
                "rollout_steps": rollout_steps,
                "episode_ticks": episode_ticks,
                "device": device,
                "map_name": map_name,
                "map_path": map_path,
                # Recorded because the *resolved* size is what the observation
                # normalises by; the dashboard used to have no way to tell that
                # a request ran with the wrong geometry.
                "map_width": map_width,
                "map_height": map_height,
                "frame_ms": frame_ms,
                "physics_ms": physics_ms,
                # Acceleration actually requested, so the dashboard can show it
                # and ``recover.py`` can read it back after a restart (same path
                # as ``frame_ms``).
                "fast_forward": fast_forward,
                "lockstep": lockstep,
                "lockstep_timeout_ms": lockstep_timeout_ms,
                "vision_radius": vision_radius,
                "eval_every": eval_every,
                "eval_episodes": eval_episodes,
                "use_llm": use_llm,
                "ai_pilot": ai_pilot,
                "record_ai_orders": record_ai_orders,
                "ai_difficulty": ai_difficulty,
                "enemy_ai_difficulty": enemy_ai_difficulty,
                "arena": arena,
                "units_per_team": units_per_team,
                "max_entities": max_entities,
                "target_slots": target_slots,
                "terrain": terrain,
                "win_mode": win_mode,
                "unlock_ai": unlock_ai,
                # The *resolved* starting weights, not the requested ones: the
                # dashboard needs to show which checkpoint training actually
                # began from, and "warmstart" resolves to a different file per
                # tank count.
                "init_mode": init_state["mode"],
                "init_checkpoint": init_checkpoint,
                # Non-fatal note about the real starting point (e.g. "asked for
                # warmstart but no BC weight exists for this tank count, so this
                # run starts from scratch"). Without it the dashboard shows
                # "预热策略" for a run that is actually random-initialised.
                "init_warning": init_state.get("warning") or "",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_ts": time.time(),
                "elapsed": 0.0,
                "latest_update": None,
                "evaluations": [],
                "summary": None,
                "live_file": None,
                "out_dir": None,
                "error": None,
                "last_log": "",
            }
            # Persist immediately: a restart in the next few seconds must still
            # know this job exists (see ``recover.load_persisted_jobs``).
            _persist_job(job_id, dict(JOBS[job_id]))

        def runner() -> None:
            run_ppo(
                job_id,
                updates=updates,
                envs=envs,
                rollout_steps=rollout_steps,
                episode_ticks=episode_ticks,
                device=device,
                map_name=map_name,
                map_path=map_path,
                map_width=map_width,
                map_height=map_height,
                frame_ms=frame_ms,
                physics_ms=physics_ms,
                fast_forward=fast_forward,
                lockstep=lockstep,
                lockstep_timeout_ms=lockstep_timeout_ms,
                vision_radius=vision_radius,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                use_llm=use_llm,
                units_per_team=units_per_team,
                max_entities=max_entities,
                target_slots=target_slots,
                terrain=terrain,
                win_mode=win_mode,
                unlock_ai=unlock_ai,
                ai_pilot=ai_pilot,
                record_ai_orders=record_ai_orders,
                ai_difficulty=ai_difficulty,
                enemy_ai_difficulty=enemy_ai_difficulty,
                init_checkpoint=init_checkpoint,
            )

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": "ppo"})

    def _start(self, kind: str, payload: dict) -> None:
        ticks = int(payload.get("ticks", 3600))
        policy = str(payload.get("policy", "scripted"))
        if policy not in ("scripted", "none", "model", "ppo"):
            self._send({"error": "policy must be scripted|none|model"}, 400)
            return
        checkpoint = payload.get("checkpoint")
        map_name = payload.get("map_name")
        map_path = payload.get("map_path")
        # Same rule as ``/ppo``: never fall back to one map's pixel size. The old
        # ``2600x2600`` constant was Lake's *tile count* squared by accident and
        # is wrong for eight of the nine stock maps.
        map_width, map_height = resolve_map_geometry(
            payload, map_name, map_path, fallback=(2600.0, 2600.0)
        )
        # ``/infer`` / ``/record`` / ``/train`` keep the historical 2 ms default.
        # Unlike ``/ppo`` this path does not pin ``--physics-ms``, so ``frame_ms``
        # is still the engine delta and 0 would be clamped back to 1 with a
        # warning. The dashboard always sends an explicit value (0 = unpaced);
        # this default only affects direct callers.
        frame_ms = _optional_int(payload.get("frame_ms"), 2)
        if frame_ms is None:
            frame_ms = 2
        episodes = int(payload.get("episodes", 1))
        concurrency = int(payload.get("concurrency", 1))
        is_model = bool(policy in ("model", "ppo") and checkpoint)
        # /train collects data with the scripted policy, /infer may also drive a
        # checkpoint. Lockstep follows the runner unless the caller said
        # otherwise: a model player must answer every observation, the recorder
        # only sends an order every N observations.
        accel = resolve_acceleration(payload, lockstep_default=is_model)
        fast_forward = accel["fast_forward"]
        lockstep_choice: bool | None
        if accel["lockstep_explicit"]:
            lockstep_choice = accel["lockstep"]
        elif is_model:
            lockstep_choice = True
        else:
            lockstep_choice = False
        lockstep_timeout_ms = accel["lockstep_timeout_ms"]
        accel_error = acceleration_error(payload.get("fast_forward"))
        if accel_error:
            self._send({"error": accel_error}, 400)
            return
        if kind == "train" and not 1 <= episodes <= 50:
            self._send({"error": "episodes out of range"}, 400)
            return
        if not 1 <= concurrency <= state.MAX_PARALLEL:
            self._send({"error": f"concurrency must be 1..{state.MAX_PARALLEL}"}, 400)
            return
        if not 120 <= ticks <= 216000:
            self._send({"error": "ticks must be 120..216000"}, 400)
            return
        if policy in ("model", "ppo") and not checkpoint:
            self._send({"error": "policy=model requires checkpoint"}, 400)
            return
        if policy in ("model", "ppo") and checkpoint:
            # Same container-prefix rule as ``/ppo`` (``resolve_init_state``): the
            # platform lists weights as ``/app/runs/...`` and the host has no
            # ``/app``. The backend already rewrites the prefix, but a caller that
            # talks to this service directly (curl, a script) would otherwise get
            # "checkpoint not found" for a perfectly valid weight.
            checkpoint = _strip_container_prefix(checkpoint)
            candidate = Path(str(checkpoint))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if not candidate.exists():
                self._send({"error": f"checkpoint not found: {checkpoint}"}, 400)
                return
            checkpoint = str(candidate)
        if self._ppo_running():
            with JOBS_LOCK:
                running_ppo = [job["id"] for job in JOBS.values() if job.get("kind") == "ppo" and job.get("status") == "running"]
            running_id = running_ppo[0] if running_ppo else "unknown"
            self._send({"error": f"当前有 PPO 训练任务正在进行 ({running_id})，请先停止它再启动实机任务", "running_job_id": running_id}, 409)
            return
        if PORT_SLOTS.qsize() <= 0:
            self._send({"error": "all parallel slots are busy"}, 409)
            return
        job_id = f"rwpp-{uuid.uuid4().hex[:8]}"
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "ticks": ticks,
                "policy": policy,
                "checkpoint": checkpoint,
                "frame_ms": frame_ms,
                # Acceleration is part of the job record so the dashboard can
                # show it and ``recover.py`` can rebuild it from the persisted
                # JSON after a restart (same path as ``frame_ms``).
                "fast_forward": fast_forward,
                "lockstep": lockstep_choice,
                "lockstep_timeout_ms": lockstep_timeout_ms,
                "concurrency": concurrency if kind == "train" else 1,
                "episodes_total": episodes if kind == "train" else 1,
                "episodes_done": 0,
                "episodes_metrics": [],
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_ts": time.time(),
                "elapsed": 0.0,
                "ticks_total": 0,
                "replays": [],
                "replay_path": None,
                "frames_file": f"runs/platform/rwpp-live-{job_id}/frames.jsonl" if kind == "infer" else None,
                "frames": 0,
                "port": None,
                "map_name": map_name,
                "map_path": map_path,
                "map_width": map_width,
                "map_height": map_height,
            }
            _persist_job(job_id, dict(JOBS[job_id]))

        def runner() -> None:
            if kind == "train":
                run_train(
                    job_id,
                    episodes,
                    ticks,
                    policy,
                    map_name,
                    concurrency,
                    map_path=map_path,
                    map_width=map_width,
                    map_height=map_height,
                    frame_ms=frame_ms,
                    fast_forward=fast_forward,
                    lockstep=bool(lockstep_choice),
                    lockstep_timeout_ms=lockstep_timeout_ms,
                )
            else:
                run_infer(
                    job_id,
                    ticks,
                    policy,
                    map_name,
                    map_path=map_path,
                    map_width=map_width,
                    map_height=map_height,
                    checkpoint=checkpoint,
                    frame_ms=frame_ms,
                    fast_forward=fast_forward,
                    # ``None`` = "the runner decides"; only an explicit request
                    # from the caller is forwarded as a hard bool.
                    lockstep=(bool(lockstep_choice) if accel["lockstep_explicit"] else None),
                    lockstep_timeout_ms=lockstep_timeout_ms,
                )

        threading.Thread(target=runner, daemon=True).start()
        self._send({"job_id": job_id, "status": "running", "kind": kind})

    def _ppo_running(self) -> bool:
        with JOBS_LOCK:
            return any(
                job.get("kind") == "ppo" and job.get("status") == "running"
                for job in JOBS.values()
            )
