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

from pathlib import Path

from .state import JOBS, JOBS_LOCK

__all__ = [
    "REPO_ROOT",
    "RECORDER",
    "PLAYER",
    "RUNS",
    "_relative",
    "_update",
    "_append_replay",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDER = REPO_ROOT / "scripts" / "record_rwpp_replay.py"
PLAYER = REPO_ROOT / "scripts" / "play_rwpp_policy.py"
RUNS = REPO_ROOT / "runs" / "platform"
RUNS.mkdir(parents=True, exist_ok=True)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _append_replay(job_id: str, replay: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["replays"].append(replay)
