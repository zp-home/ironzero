# SPDX-License-Identifier: Apache-2.0
"""``/ppo`` 执行体：拉起源码树里的 ``ironzero.training.train_rwpp`` 子进程并轮询。

数据流：

1. ``run_ppo`` 拼出 trainer 命令行（含 arena 观测/动作契约、地形、胜负条件），
   把 stdout 重定向到 ``<out>/train.log``；
2. trainer 每个 update / eval 往 ``<out>/live.jsonl`` 追加一行 JSON；
3. 本模块每 2 s 读一次 live 文件，把 ``updates_done`` / ``latest_update`` /
   ``evaluations`` / ``elapsed`` 写回 JOBS（等价于 SSE 的轮询替代）；
4. 进程退出后读取 ``rwpp_summary.json`` 与日志尾，落终态 status。

停止：``stop_job`` 会 kill ``PPO_PROCESSES[job_id]`` 并置 ``PPO_STOP``，因此终态
区分 ``stopped`` / ``done`` / ``failed``。

另外两个预设收敛函数（原本内联在 ``Handler._start_ppo`` 里）也住在这里，因为
它们只服务 ``/ppo``：

* :func:`resolve_preset` —— arena 判定 + 单位数/实体数/地形/胜负条件收敛。
* :func:`resolve_init_checkpoint` —— ``init_mode`` / ``init_checkpoint`` 收敛。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import REPO_ROOT, RUNS, _relative, _strip_container_prefix, _update
from .proc import _accel_flags
from .state import PPO_PROCESSES, PPO_STOP

__all__ = [
    "build_ppo_command",
    "run_ppo",
    "_read_live_events",
    "resolve_preset",
    "resolve_init_state",
    "resolve_init_checkpoint",
    "resolve_bc_checkpoint",
]


def _read_live_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return events
    return events


def build_ppo_command(
    *,
    job_dir: Path,
    live_path: Path,
    updates: int,
    envs: int,
    rollout_steps: int,
    episode_ticks: int,
    device: str,
    map_name: str,
    map_path: str,
    map_width: float,
    map_height: float,
    frame_ms: int = 2,
    physics_ms: int = 16,
    fast_forward: int = 1,
    lockstep: bool = True,
    lockstep_timeout_ms: int = 30000,
    vision_radius: float = 260.0,
    eval_every: int = 0,
    eval_episodes: int = 1,
    use_llm: bool = False,
    units_per_team: int | None = None,
    max_entities: int | None = None,
    target_slots: int | None = None,
    terrain: str | None = None,
    win_mode: str = "",
    unlock_ai: bool = False,
    ai_pilot: bool = False,
    record_ai_orders: bool = False,
    ai_difficulty: int | None = None,
    enemy_ai_difficulty: int | None = None,
    init_checkpoint: str | None = None,
) -> list[str]:
    """The ``ironzero.training.train_rwpp`` command line for one ``/ppo`` job.

    Pure (no filesystem, no subprocess) so the *exact* argv a web request produces
    can be asserted in a test. That matters more than it looks: three separate
    regressions in this file were "the payload said X, the command said Y":

    * the map pixel size was the constant ``2700x2500`` on every map;
    * a dropped request key sent no flag at all and still answered 200;
    * ``enemy_ai_difficulty`` was only forwarded when the caller happened to set
      it, so omitting it made the engine fall back to its global default of 0
      (the easiest real opponent) while the form showed a difficulty box.

    ``enemy_ai_difficulty`` is therefore written whenever it is not ``None`` --
    ``launch.py`` resolves "not supplied" to the intended default (1) *before*
    calling this, so the flag is always on the command line in practice.
    """

    command = [
        sys.executable,
        "-m",
        "ironzero.training.train_rwpp",
        "--updates",
        str(updates),
        "--envs",
        str(envs),
        "--rollout-steps",
        str(rollout_steps),
        "--episode-ticks",
        str(episode_ticks),
        "--device",
        device,
        "--map",
        map_path,
        "--map-name",
        map_name,
        "--map-width",
        str(map_width),
        "--map-height",
        str(map_height),
        "--frame-ms",
        str(max(0, int(frame_ms))),
        "--vision-radius",
        str(float(vision_radius)),
        "--out",
        str(job_dir),
        "--live",
        str(live_path),
        "--eval-every",
        str(int(eval_every) if int(eval_every) > 0 else int(updates)),
        "--eval-episodes",
        str(max(1, int(eval_episodes))),
    ]
    # Always pin the simulation delta explicitly: the trainer's own default has
    # already moved twice, and a web-launched run must not silently inherit it.
    # ``--physics-ms`` fixes what one episode simulates, so ``--frame-ms`` is free
    # to only pace the wall clock (0 ms = never sleep, measured 26.5x real time vs
    # 10.7x at 1 ms; both simulate exactly the same battle).
    command.extend(["--physics-ms", str(max(0, int(physics_ms)))])
    # Engine-native acceleration. Opt-in only, so the default command line stays
    # byte-identical; the trainer's own defaults already are ``--fast-forward 1``
    # and ``--lockstep`` on, but they are stated explicitly here so a web run
    # never silently inherits a changed default.
    command.extend(_accel_flags(fast_forward, lockstep, lockstep_timeout_ms))
    if use_llm:
        command.append("--use-llm")
    # Arena scenarios (e.g. the 25-vs-25 IronZero25 map) need the wider
    # observation/action contract, the terrain channel and the bridge's
    # annihilation win condition; legacy maps keep the old 8-unit defaults.
    if units_per_team:
        command.extend(["--units-per-team", str(int(units_per_team))])
    if max_entities:
        command.extend(["--max-entities", str(int(max_entities))])
    if target_slots:
        command.extend(["--target-slots", str(int(target_slots))])
    if terrain:
        command.extend(["--terrain", str(terrain)])
    if win_mode:
        command.extend(["--win-mode", str(win_mode)])
    if unlock_ai:
        command.append("--unlock-ai")
    # AI teacher / opponent. ``--ai-pilot`` hands the controlled team to the
    # built-in AI so it can be behaviour-cloned; ``--enemy-ai-difficulty`` pins
    # only the other side so the teacher faces real pressure (measured: two AIs
    # at the same level mirror each other and never fight).
    if ai_pilot or record_ai_orders:
        command.append("--ai-pilot")
    if record_ai_orders:
        command.append("--record-ai-orders")
    if ai_difficulty is not None:
        command.extend(["--ai-difficulty", str(int(ai_difficulty))])
    if enemy_ai_difficulty is not None:
        command.extend(["--enemy-ai-difficulty", str(int(enemy_ai_difficulty))])
    if init_checkpoint:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    return command


def run_ppo(
    job_id: str,
    updates: int,
    envs: int,
    rollout_steps: int,
    episode_ticks: int,
    device: str,
    map_name: str,
    map_path: str,
    map_width: float,
    map_height: float,
    # Defaults mirror ``launch.py::_start_ppo`` and ``RwppPpoRequest``: a direct
    # caller that omits the key gets the historical 2 ms, while the dashboard
    # sends 0 explicitly (unpaced, measured 26.5x with ``physics_ms=16``).
    frame_ms: int = 2,
    physics_ms: int = 16,
    fast_forward: int = 1,
    lockstep: bool = True,
    lockstep_timeout_ms: int = 30000,
    vision_radius: float = 260.0,
    eval_every: int = 0,
    eval_episodes: int = 1,
    use_llm: bool = False,
    units_per_team: int | None = None,
    max_entities: int | None = None,
    target_slots: int | None = None,
    terrain: str | None = None,
    win_mode: str = "",
    unlock_ai: bool = False,
    ai_pilot: bool = False,
    record_ai_orders: bool = False,
    ai_difficulty: int | None = None,
    enemy_ai_difficulty: int | None = None,
    init_checkpoint: str | None = None,
) -> None:
    job_dir = RUNS / f"rwpp-ppo-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    live_path = job_dir / "live.jsonl"
    log_path = job_dir / "train.log"
    summary_path = job_dir / "rwpp_summary.json"
    command = build_ppo_command(
        job_dir=job_dir,
        live_path=live_path,
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
    started = time.time()
    handle = log_path.open("w", encoding="utf-8", errors="replace")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        PPO_PROCESSES[job_id] = process
        _update(job_id, pid=process.pid, live_file=_relative(live_path), out_dir=_relative(job_dir))
        while process.poll() is None:
            time.sleep(2.0)
            events = _read_live_events(live_path)
            updates_done = sum(1 for event in events if event.get("kind") == "update")
            evaluations = [event["payload"] for event in events if event.get("kind") == "eval"]
            latest_update = next(
                (event["payload"] for event in reversed(events) if event.get("kind") == "update"), None
            )
            _update(
                job_id,
                updates_done=updates_done,
                latest_update=latest_update,
                evaluations=evaluations[-10:],
                elapsed=round(time.time() - started, 1),
                events=len(events),
            )
        return_code = process.wait()
        events = _read_live_events(live_path)
        updates_done = sum(1 for event in events if event.get("kind") == "update")
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = None
        log_tail = ""
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        except OSError:
            pass
        stopped = job_id in PPO_STOP
        status = "stopped" if stopped else ("done" if return_code == 0 else "failed")
        _update(
            job_id,
            status=status,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            seconds=round(time.time() - started, 1),
            elapsed=round(time.time() - started, 1),
            updates_done=updates_done,
            latest_update=next(
                (event["payload"] for event in reversed(events) if event.get("kind") == "update"), None
            ),
            evaluations=[event["payload"] for event in events if event.get("kind") == "eval"][-10:],
            summary=summary,
            last_log=log_tail,
            error=None if (return_code == 0 or stopped) else f"trainer exited with code {return_code}",
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="failed", error=str(exc), elapsed=round(time.time() - started, 1))
    finally:
        PPO_STOP.discard(job_id)
        handle.close()
        PPO_PROCESSES.pop(job_id, None)


def resolve_preset(payload: dict, map_name: str, map_path: str) -> dict:
    """把 payload 里的 arena 相关字段收敛成 trainer 可用的确定值。

    返回字典里的 ``arena_error`` 是**非空字符串**时表示 arena 地图准备失败，
    调用方必须把它变成 400 拒绝请求而不是照常开跑：训练器一旦拿不到 arena
    tmx，只会在子进程里报一个看不懂的错，或者干脆跑到一张不存在的地图上。

    这里必须捕获 ``BaseException``：``make_ironzero_map`` 的失败路径全是
    ``raise SystemExit(...)``（找不到源图、tileset 缺失、连通性检查不过），
    而 ``SystemExit`` 不是 ``Exception`` 子类。原来只写 ``except Exception``，
    于是 SystemExit 直接逃出 HTTP handler 线程 —— 客户端只看到连接被关掉，
    前端弹的却是「真机服务未启动」，真正的失败原因（runtime 路径不对）完全丢失。
    """

    units_per_team = payload.get("units_per_team")
    arena = (
        bool(payload.get("arena"))
        or "ironzero" in map_path.lower()
        or "ironzero" in map_name.lower()
    )
    max_entities = payload.get("max_entities")
    target_slots = payload.get("target_slots")
    terrain = str(payload.get("terrain") or "")
    win_mode = str(payload.get("win_mode") or "")
    unlock_ai = bool(payload.get("unlock_ai", False))
    arena_error = ""
    if arena:
        if units_per_team is not None:
            units_per_team = max(1, min(25, int(units_per_team)))
        else:
            digits = "".join(filter(str.isdigit, map_name))
            units_per_team = int(digits) if digits and 1 <= int(digits) <= 25 else 25
        max_entities = int(max_entities or units_per_team)
        target_slots = int(target_slots or (max_entities + 1))
        try:
            scripts_dir = str(REPO_ROOT / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from make_ironzero_map import ensure_arena_map

            _, rel_path, arena_map_name = ensure_arena_map(units_per_team)
            map_path = rel_path
            map_name = arena_map_name
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - SystemExit included on purpose
            # Keep the original map so the payload still describes what was asked
            # for, and let the caller refuse the request with this message.
            arena_error = (
                f"arena map preparation failed for units_per_team={units_per_team}: "
                f"{str(exc) or type(exc).__name__}"
            )
            print(f"[ppo] {arena_error}", file=sys.stderr, flush=True)
        terrain = terrain or str(REPO_ROOT / "configs" / "maps" / f"{map_name}.terrain.json")
        if not Path(terrain).is_file():
            terrain = str(REPO_ROOT / "configs" / "maps" / "IronZero25.terrain.json")
        win_mode = win_mode or "annihilation"
        unlock_ai = True
    else:
        if units_per_team is not None:
            units_per_team = max(1, min(25, int(units_per_team)))
        if not terrain:
            candidate_terrain = REPO_ROOT / "configs" / "maps" / f"{Path(map_name).stem}.terrain.json"
            if candidate_terrain.is_file():
                terrain = str(candidate_terrain)
            else:
                terrain = ""
        # Stock skirmish maps declare ``type=skirmish`` and carry no
        # ``winCondition``, so on the original maps the native win channel only
        # fires when the enemy is erased down to its last unit. Measured over
        # 7129 native-mode games: zero victories. The bridge's ``objective`` mode
        # scores the engine's own skirmish objectives instead (main buildings,
        # then "no factory and no builder"), which IS reachable on an economy map.
        # Default it for non-arena runs unless the caller pinned something.
        if not win_mode:
            win_mode = "objective"

    return {
        "arena": arena,
        "arena_error": arena_error,
        "units_per_team": units_per_team,
        "max_entities": max_entities,
        "target_slots": target_slots,
        "terrain": terrain,
        "win_mode": win_mode,
        "unlock_ai": unlock_ai,
        "map_name": map_name,
        "map_path": map_path,
    }


#: BC 预热权重可能落盘的父目录（都相对 ``REPO_ROOT``）。
#:
#: ``scripts/bc_warmstart_rwpp.py`` 的 ``--out`` 默认值是 ``runs/bc_ironzero25``，
#: 但仓库里实际发布的权重在 ``runs/experiments/bc_ironzero{5,10,15,20,25}``。
#: 原来只查 ``runs/bc_ironzero*``，结果 warmstart **永远找不到**权重、静默退回
#: ``None``（等于从零随机探索），训练表现和前几百轮胜率长期对不上预期。
#: 两处都查，顺序保持「先兼容旧默认位置」。
_BC_CHECKPOINT_PARENTS: tuple[str, ...] = ("runs", "runs/experiments")


def _robust_units(units_per_team) -> int | None:
    """把 ``units_per_team`` 收敛成正整数，非法值返回 ``None``。"""

    try:
        units = int(units_per_team)
    except (TypeError, ValueError):
        return None
    return units if units > 0 else None


def resolve_bc_checkpoint(units_per_team) -> Path | None:
    """按坦克数查找可用的 BC 预热权重，找不到返回 ``None``。

    只做**精确匹配**，不再回退到 25v25。回退看起来"总能找到权重"，但那份权重是
    在 25 辆坦克的布局上学的，给 2v2 地图用语义上是错的：迁移机制会把 768 维的
    trunk 逐列重映射到 511 维（实测 coverage 99%，所以不会崩），但策略学到的是
    "指挥 25 个单位"，不是"从 2 个单位起家搞经济"。用户看到"预热策略"会以为站在
    BC 肩膀上，实际拿到的是一个布局和玩法都不匹配的起点。

    找不到时返回 ``None``，由 :func:`resolve_init_state` 显式告诉用户"本次从零
    开始"，而不是静默换一个不相干的权重。

    每个候选名都在 :data:`_BC_CHECKPOINT_PARENTS` 的每个父目录下查找。
    """

    units = _robust_units(units_per_team)
    if units is None:
        return None
    name = f"bc_ironzero{units}"
    for parent in _BC_CHECKPOINT_PARENTS:
        candidate = REPO_ROOT / parent / name / "checkpoint.pt"
        if candidate.is_file():
            return candidate
    return None


def resolve_init_state(payload: dict, units_per_team) -> dict:
    """收敛 ``init_checkpoint``，并把「为什么没用上」也说出来。

    返回 ``{"checkpoint": str|None, "mode": str, "error": str}``。
    ``error`` 非空表示用户**显式**要求了某个 checkpoint 却无法使用（路径不存在、
    或是容器路径 ``/app/...`` 在宿主机上根本没有这个目录）——这种情况必须报错，
    因为静默退回 ``None`` 会让「加载已有权重继续训练」变成「从零重训」，界面上
    却完全看不出来。warmstart 找不到 BC 权重属于可选增强，只写日志不报错。
    """

    init_checkpoint = payload.get("init_checkpoint")
    init_mode = str(payload.get("init_mode") or "")
    error = ""
    warmstart_warning = ""
    explicit = bool(init_checkpoint)
    if explicit:
        # 平台后端把 checkpoint 列成容器路径（``/app/runs/...``）。宿主机上
        # 没有 ``/app``，所以要按仓库根还原；正常路径由 ``routers/rwpp.py``
        # 改写，这里兜底是为了别的调用方直接打宿主机服务时也能用。
        init_checkpoint = _strip_container_prefix(init_checkpoint)
    if (init_mode == "warmstart" or not init_checkpoint) and init_mode != "scratch":
        # 只按当前坦克数精确匹配预热权重。以前找不到会回退到 bc_ironzero25，
        # 那份权重是在 25 辆坦克的布局上学的：迁移机制能把它逐列搬进 2v2 的网络
        # （实测 coverage 99%，不会崩），但策略学到的是「指挥 25 个单位」，
        # 与「从 2 个单位起家搞经济」完全不是一回事。给了比没有更误导。
        best_bc = resolve_bc_checkpoint(units_per_team)
        if best_bc is not None:
            init_checkpoint = str(best_bc)
        elif init_mode == "warmstart":
            # 用户明确选了「预热策略」却拿不到匹配的权重：必须说出来，
            # 否则界面上「预热」和「从零」长得一模一样。
            print(
                f"[warmstart] 未找到匹配的 BC 预热权重 (units_per_team={units_per_team!r})，"
                f"已查找: "
                + ", ".join(
                    str(
                        REPO_ROOT
                        / parent
                        / f"bc_ironzero{_robust_units(units_per_team) or '?'}"
                        / "checkpoint.pt"
                    )
                    for parent in _BC_CHECKPOINT_PARENTS
                )
                + "；本次将从零开始训练（不再回退到其他坦克数的权重）。",
                file=sys.stderr,
                flush=True,
            )
            warmstart_warning = (
                f"没有 units_per_team={_robust_units(units_per_team)} 的 BC 预热权重，"
                "本次从零开始训练"
            )
    if warmstart_warning:
        # Surfaced through the job record so the dashboard can show the real start.
        print(f"[warmstart] {warmstart_warning}", file=sys.stderr, flush=True)
    if init_checkpoint:
        candidate = Path(str(init_checkpoint))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_file():
            init_checkpoint = str(candidate.resolve())
        else:
            if explicit:
                error = (
                    f"init_checkpoint not found on the host: {payload.get('init_checkpoint')}"
                    f" (resolved to {candidate})"
                )
            init_checkpoint = None
    return {
        "checkpoint": init_checkpoint,
        "mode": init_mode,
        "error": error,
        "warning": warmstart_warning,
    }


def resolve_init_checkpoint(payload: dict, units_per_team) -> str | None:
    """收敛 ``init_checkpoint``：显式路径优先，否则按坦克数回退到 BC 预热权重。"""

    return resolve_init_state(payload, units_per_team)["checkpoint"]
