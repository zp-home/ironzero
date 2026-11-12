# SPDX-License-Identifier: Apache-2.0
"""服务重启后接管仍在运行的任务（``rwpp_service.recover``）。

背景：``run_ppo`` 的轮询线程活在服务进程里，trainer 是独立子进程。服务一重启，
``JOBS`` 表清空，但机器上还在训练 —— Web 大屏什么都看不到，用户既不能观察也不能
停止。这曾经真实发生过（第 107/500 轮训练从平台上消失）。

这些用例锁住三件事：

1. 任务记录会落盘，重启后能恢复；
2. 恢复出来的 ``running`` 记录必须按**产物证据**重新校验，不能把几天前就跑完的
   任务复活成"正在跑"；
3. 只有 ``live.jsonl`` 还在更新的目录才算可接管，并且 pid 能按 ``--out`` 正确归属。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "scripts", REPO / "web" / "backend"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture()
def service(tmp_path, monkeypatch):
    """Isolated ``JOBS`` + a temp runs tree, so tests never touch real runs.

    ``paths``/``recover`` bind ``JOBS`` at import time (``from .state import JOBS``),
    so the *same dict object* has to be shared with them -- rebinding
    ``state.JOBS`` alone would leave them writing to the real table. That is also
    the production contract (see ``state.py``: mutate in place, never reassign).
    """

    from rwpp_service import paths, recover, state

    state.JOBS.clear()
    state.PPO_STOP.clear()
    monkeypatch.setattr(paths, "JOBS", state.JOBS)
    monkeypatch.setattr(recover, "JOBS", state.JOBS)
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    monkeypatch.setattr(recover, "RUNS", tmp_path)
    state_dir = tmp_path / "_state"
    monkeypatch.setattr(paths, "JOB_STATE_DIR", state_dir)
    monkeypatch.setattr(recover, "JOB_STATE_DIR", state_dir)
    # ``_relative`` resolves against REPO_ROOT; point it at the temp tree instead.
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(recover, "REPO_ROOT", tmp_path)
    yield recover, paths, state, tmp_path
    state.JOBS.clear()
    state.PPO_STOP.clear()


def _write_live(job_dir: Path, *, updates: int, envs: int = 10, total: int = 500) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "seq": 1,
            "kind": "start",
            "payload": {
                "config": {
                    "total_updates": total,
                    "num_envs": envs,
                    "rollout_steps": 64,
                    "eval_every": 10,
                    "eval_episodes": 4,
                    "device": "cuda",
                },
                "created_at": "2026-11-12 14:52:58",
            },
        }
    ]
    for update in range(1, updates + 1):
        lines.append(
            {"seq": update + 1, "kind": "update", "payload": {"update": update, "elapsed_s": update * 6.0}}
        )
    (job_dir / "live.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    # The trainer's first stdout line: the only place the scenario is recorded.
    (job_dir / "train.log").write_text(
        json.dumps(
            {
                "mode": "real-game-ppo",
                "envs": envs,
                "episode_ticks": 5000,
                "units_per_team": 2,
                "map": "[p2]Lake (2p)",
                "frame_ms": 1,
                "vision_radius": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_job_state_is_persisted_and_restored(service) -> None:
    recover, paths, state, _tmp = service

    state.JOBS["ppo-abc"] = {
        "id": "ppo-abc",
        "kind": "ppo",
        "status": "running",
        "started_ts": time.time(),
        "out_dir": "runs/platform/rwpp-ppo-ppo-abc",
        "replays": [],
    }
    paths._update("ppo-abc", updates_done=7)

    on_disk = json.loads((paths.JOB_STATE_DIR / "ppo-abc.json").read_text(encoding="utf-8"))
    assert on_disk["updates_done"] == 7
    assert on_disk["status"] == "running"

    state.JOBS.clear()
    restored = recover.load_persisted_jobs()
    assert restored == 1
    assert state.JOBS["ppo-abc"]["updates_done"] == 7


def test_persist_failure_does_not_break_the_job(service) -> None:
    """Disk problems must never take down a running job (the table is the truth)."""

    recover, paths, state, tmp_path = service

    state.JOBS["ppo-abc"] = {"id": "ppo-abc", "kind": "ppo", "status": "running", "replays": []}
    # Make the state dir un-creatable by putting a *file* where the dir should go.
    (tmp_path / "_state").write_text("not a directory", encoding="utf-8")
    paths._update("ppo-abc", updates_done=3)  # must not raise

    assert state.JOBS["ppo-abc"]["updates_done"] == 3


def test_finished_run_is_restored_as_done_not_running(service) -> None:
    """A days-old job must not be resurrected as 'running' and hog the dashboard."""

    recover, paths, state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-old1"
    _write_live(job_dir, updates=3)
    (job_dir / "rwpp_summary.json").write_text('{"updates": 3}', encoding="utf-8")

    state.JOBS["ppo-old1"] = {
        "id": "ppo-old1",
        "kind": "ppo",
        "status": "running",
        "started_ts": time.time(),
        "out_dir": "rwpp-ppo-ppo-old1",
        "replays": [],
    }
    paths._update("ppo-old1")
    state.JOBS.clear()

    recover.load_persisted_jobs()
    assert state.JOBS["ppo-old1"]["status"] == "done"


def test_dead_run_without_summary_is_restored_as_stale(service) -> None:
    recover, paths, state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-dead"
    _write_live(job_dir, updates=3)
    old = time.time() - 10_000
    (job_dir / "live.jsonl").touch()
    import os

    os.utime(job_dir / "live.jsonl", (old, old))

    state.JOBS["ppo-dead"] = {
        "id": "ppo-dead",
        "kind": "ppo",
        "status": "running",
        "started_ts": old,
        "out_dir": "rwpp-ppo-ppo-dead",
        "replays": [],
    }
    paths._update("ppo-dead")
    state.JOBS.clear()

    recover.load_persisted_jobs()
    assert state.JOBS["ppo-dead"]["status"] == "stale"
    assert state.JOBS["ppo-dead"]["error"]


def test_adoptable_dirs_require_fresh_live_events(service) -> None:
    recover, _paths, _state, tmp_path = service

    fresh = tmp_path / "rwpp-ppo-ppo-fresh"
    _write_live(fresh, updates=2)

    stale = tmp_path / "rwpp-ppo-ppo-stale"
    _write_live(stale, updates=2)
    import os

    old = time.time() - (recover.STALE_SECONDS + 60)
    os.utime(stale / "live.jsonl", (old, old))

    finished = tmp_path / "rwpp-ppo-ppo-finished"
    _write_live(finished, updates=2)
    (finished / "rwpp_summary.json").write_text("{}", encoding="utf-8")

    adopted = {path.name for path in recover.adoptable_job_dirs()}
    assert adopted == {"rwpp-ppo-ppo-fresh"}


def test_job_is_rebuilt_from_live_events(service) -> None:
    recover, _paths, _state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-live"
    _write_live(job_dir, updates=4, envs=10, total=500)

    record = recover._job_from_live(job_dir)
    assert record is not None
    assert record["id"] == "ppo-live"
    assert record["kind"] == "ppo"
    assert record["updates_done"] == 4
    assert record["updates_total"] == 500
    assert record["envs"] == 10
    assert record["adopted"] is True
    assert record["latest_update"]["update"] == 4
    # Scenario fields come from the trainer's first log line, not live.jsonl.
    assert record["map_name"] == "[p2]Lake (2p)"
    assert record["episode_ticks"] == 5000
    assert record["frame_ms"] == 1


def test_scenario_comes_from_the_trainers_command_line(service) -> None:
    """The map name only survives in argv; the log summary has no map field."""

    recover, _paths, _state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-cmd"
    _write_live(job_dir, updates=1)
    # Drop the map key to mimic the real trainer summary, which lacks it.
    (job_dir / "train.log").write_text('{"mode": "real-game-ppo", "episode_ticks": 5000}\n', encoding="utf-8")

    cmdline = [
        "python",
        "-m",
        "ironzero.training.train_rwpp",
        "--out",
        str(job_dir),
        "--map",
        "maps/skirmish/[p2]Lake (2p).tmx",
        "--map-name",
        "[p2]Lake (2p)",
        "--frame-ms",
        "1",
        "--vision-radius",
        "0.0",
        "--units-per-team",
        "2",
    ]
    record = recover._job_from_live(job_dir, cmdline)
    assert record is not None
    assert record["map_name"] == "[p2]Lake (2p)"
    assert record["map_path"].endswith("Lake (2p).tmx")
    assert record["frame_ms"] == 1
    assert record["vision_radius"] == 0.0
    assert record["units_per_team"] == 2


def test_map_name_falls_back_to_the_map_file_stem(service) -> None:
    recover, _paths, _state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-stem"
    _write_live(job_dir, updates=1)
    record = recover._job_from_live(job_dir, ["--map", "maps/challenge/Artillery.tmx"])
    assert record is not None
    assert record["map_name"] == "Artillery"


def test_recover_does_not_duplicate_or_adopt_twice(service) -> None:
    recover, _paths, state, tmp_path = service

    job_dir = tmp_path / "rwpp-ppo-ppo-one"
    _write_live(job_dir, updates=2)

    first = recover.recover_jobs()
    second = recover.recover_jobs()
    assert first == 1
    assert second == 0, "an already-known run must not be adopted again"
    state.PPO_STOP.add("ppo-one")
