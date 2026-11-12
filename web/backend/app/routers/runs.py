# SPDX-License-Identifier: Apache-2.0
"""PPO training runs: launch, list, poll events, stop."""

from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..jobs import (
    _jobs,
    _jobs_lock,
    _job_status,
    _latest_from_events,
    _list_train_runs,
    _new_job_dir,
    _now,
    _python,
    _read_jsonl,
    _read_meta,
    _spawn,
    _tail,
    _write_meta,
)
from ..models import TrainRequest
from ..paths import RUNS_ROOT, _resolve_inside

router = APIRouter()


@router.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return _list_train_runs()


@router.post("/api/runs")
def start_run(request: TrainRequest) -> dict[str, Any]:
    job_id, job_dir = _new_job_dir("train")
    config = request.model_dump()
    config["name"] = request.name or f"PPO updates={request.updates} envs={request.envs}"
    command = [
        _python(),
        "-m",
        "ironzero.training.train",
        "--updates",
        str(request.updates),
        "--envs",
        str(request.envs),
        "--rollout-steps",
        str(request.rollout_steps),
        "--epochs",
        str(request.epochs),
        "--opponent",
        request.opponent,
        "--seed",
        str(request.seed),
        "--eval-every",
        str(request.eval_every),
        "--eval-episodes",
        str(request.eval_episodes),
        "--lr",
        str(request.lr),
        "--entropy",
        str(request.entropy),
        "--device",
        "cpu",
        "--out",
        str(job_dir),
        "--live",
        str(job_dir / "events.jsonl"),
    ]
    _spawn(job_id, job_dir, "train", config, command)
    return {"id": job_id, "status": "running", "dir": str(job_dir)}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, run_id)
    meta = _read_meta(job_dir)
    if not meta:
        raise HTTPException(status_code=404, detail="run not found")
    events = _read_jsonl(job_dir / "events.jsonl")
    summary = _latest_from_events(events)
    return {
        **meta,
        "status": _job_status(run_id, job_dir),
        "latest": summary["latest"],
        "evals": summary["evals"],
        "final_eval": summary["final"],
        "events": summary["events"],
        "log_tail": _tail(job_dir / "stdout.log"),
    }


@router.get("/api/runs/{run_id}/events")
def get_run_events(run_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, run_id)
    events = _read_jsonl(job_dir / "events.jsonl")
    pending = [event for event in events if int(event.get("seq", 0)) > after]
    next_seq = max((int(event.get("seq", 0)) for event in events), default=after)
    return {"next": next_seq, "events": pending, "status": _job_status(run_id, job_dir)}


@router.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, run_id)
    with _jobs_lock:
        entry = _jobs.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="run is not active in this service instance")
    process = entry["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    meta = _read_meta(job_dir)
    meta["status"] = "stopped"
    meta["finished_at"] = _now()
    _write_meta(job_dir, meta)
    with _jobs_lock:
        entry["meta"]["status"] = "stopped"
    return {"ok": True, "status": "stopped"}
