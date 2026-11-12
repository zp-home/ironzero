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

from .paths import REPO_ROOT, RUNS, _relative, _update
from .state import PPO_PROCESSES, PPO_STOP

__all__ = [
    "run_ppo",
    "_read_live_events",
    "resolve_preset",
    "resolve_init_checkpoint",
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
    frame_ms: int = 2,
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
    init_checkpoint: str | None = None,
) -> None:
    job_dir = RUNS / f"rwpp-ppo-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    live_path = job_dir / "live.jsonl"
    log_path = job_dir / "train.log"
    summary_path = job_dir / "rwpp_summary.json"
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
    if init_checkpoint:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
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
    """把 payload 里的 arena 相关字段收敛成 trainer 可用的确定值。"""

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
        except Exception:
            pass
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

    return {
        "arena": arena,
        "units_per_team": units_per_team,
        "max_entities": max_entities,
        "target_slots": target_slots,
        "terrain": terrain,
        "win_mode": win_mode,
        "unlock_ai": unlock_ai,
        "map_name": map_name,
        "map_path": map_path,
    }


def resolve_init_checkpoint(payload: dict, units_per_team) -> str | None:
    """收敛 ``init_checkpoint``：显式路径优先，否则按坦克数回退到 BC 预热权重。"""

    init_checkpoint = payload.get("init_checkpoint")
    init_mode = payload.get("init_mode")
    if (init_mode == "warmstart" or not init_checkpoint) and init_mode != "scratch":
        # 根据当前坦克数量优先寻找最匹配的预热权重
        best_bc = REPO_ROOT / "runs" / f"bc_ironzero{units_per_team}" / "checkpoint.pt"
        if not best_bc.is_file():
            best_bc = REPO_ROOT / "runs" / "bc_ironzero25" / "checkpoint.pt"
        if best_bc.is_file():
            init_checkpoint = str(best_bc)
    if init_checkpoint:
        candidate = Path(str(init_checkpoint))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_file():
            init_checkpoint = str(candidate.resolve())
        else:
            init_checkpoint = None
    return init_checkpoint
