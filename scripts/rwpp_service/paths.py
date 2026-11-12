# SPDX-License-Identifier: Apache-2.0
"""仓库内路径常量与任务记录（JOBS）读写辅助。

**REPO_ROOT 语义**：原文件是 ``scripts/rwpp_real_service.py``，用
``Path(__file__).resolve().parents[1]`` 得到仓库根；本文件位于
``scripts/rwpp_service/paths.py``，多了一层目录，因此必须用 ``parents[2]``：

    parents[0] = scripts/rwpp_service
    parents[1] = scripts
    parents[2] = <仓库根>
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .state import JOBS, JOBS_LOCK

__all__ = [
    "REPO_ROOT",
    "RECORDER",
    "PLAYER",
    "RUNS",
    "JOB_STATE_DIR",
    "_relative",
    "_update",
    "_append_replay",
    "_persist_job",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDER = REPO_ROOT / "scripts" / "record_rwpp_replay.py"
PLAYER = REPO_ROOT / "scripts" / "play_rwpp_policy.py"
RUNS = REPO_ROOT / "runs" / "platform"
RUNS.mkdir(parents=True, exist_ok=True)
#: 任务表的落盘目录：重启后 ``recover.load_persisted_jobs`` 以此恢复 ``JOBS``。
#: 放在 ``runs/_rwpp_service`` 而不是 ``runs/platform``，避免混进"平台产物"，
#: 也避免被回放/模型扫描当成数据。
JOB_STATE_DIR = REPO_ROOT / "runs" / "_rwpp_service" / "jobs"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _persist_job(job_id: str, job: dict) -> None:
    """把一条任务记录写到磁盘，供服务重启后恢复。

    写失败必须**不能**影响正在跑的任务：任务表是内存态，落盘只是尽力而为的
    附属品，所以这里吞掉所有 OSError。用 ``pid`` 命名临时文件再 ``os.replace``，
    避免并发更新的两个线程写出半截 JSON。
    """

    try:
        JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)
        target = JOB_STATE_DIR / f"{job_id}.json"
        tmp = JOB_STATE_DIR / f".{job_id}.{os.getpid()}.tmp"
        tmp.write_text(
            json.dumps(job, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, target)
    except OSError:
        pass


def _update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        snapshot = dict(job)
    _persist_job(job_id, snapshot)


def _append_replay(job_id: str, replay: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["replays"].append(replay)
        snapshot = dict(job)
    _persist_job(job_id, snapshot)

