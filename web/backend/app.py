# SPDX-License-Identifier: Apache-2.0
"""IronZero training / inference platform backend.

A small FastAPI service that turns the CLI trainers into observable jobs:

* ``/api/runs``    — launch and monitor PPO training runs
* ``/api/infer``   — run inference episodes and stream live frames
* ``/api/replays`` — recorded episode library
* ``/``            — the dashboard (``platform/frontend``)

Everything runs locally: the service only listens where you bind it, jobs are
local subprocesses, and all artifacts live under ``IRONZERO_RUNS``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import maps as gamemaps

APP_VERSION = "0.1.0"
REPO_ROOT = Path(os.environ.get("IRONZERO_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = Path(os.environ.get("IRONZERO_RUNS", REPO_ROOT / "runs" / "platform")).resolve()
CHECKPOINT_ROOTS = [
    Path(item).resolve()
    for item in os.environ.get("IRONZERO_CHECKPOINT_ROOTS", str(REPO_ROOT / "runs")).split(os.pathsep)
    if item.strip()
]
FRONTEND_DIR = REPO_ROOT / "web" / "frontend"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IronZero Platform", version=APP_VERSION)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
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
    mode: str = Field(default="greedy", pattern="^(greedy|sample|random)$")
    max_decisions: int = Field(default=400, ge=10, le=2000)
    map_id: str = Field(default="", max_length=256)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _new_job_dir(kind: str) -> tuple[str, Path]:
    job_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    job_dir = RUNS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def _write_meta(job_dir: Path, meta: dict[str, Any]) -> None:
    (job_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_meta(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _tail(path: Path, lines: int = 120) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return [line.rstrip("\n") for line in content[-lines:]]


def _python() -> str:
    return sys.executable or "python"


def _job_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn(job_id: str, job_dir: Path, kind: str, config: dict[str, Any], command: list[str]) -> None:
    log_path = job_dir / "stdout.log"
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    meta = {
        "id": job_id,
        "name": config.get("name") or job_id,
        "kind": kind,
        "status": "running",
        "created_at": _now(),
        "started_at": _now(),
        "finished_at": None,
        "config": config,
        "command": command,
        "error": None,
    }
    _write_meta(job_dir, meta)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=_job_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    with _jobs_lock:
        _jobs[job_id] = {"process": process, "dir": job_dir, "meta": meta, "log": log_handle}

    def watch() -> None:
        return_code = process.wait()
        log_handle.close()
        with _jobs_lock:
            entry = _jobs.get(job_id)
            if entry is not None:
                entry["meta"]["status"] = "succeeded" if return_code == 0 else "failed"
                entry["meta"]["finished_at"] = _now()
                entry["meta"]["exit_code"] = return_code
                _write_meta(job_dir, entry["meta"])

    threading.Thread(target=watch, name=f"watch-{job_id}", daemon=True).start()


def _job_status(job_id: str, job_dir: Path) -> str:
    with _jobs_lock:
        entry = _jobs.get(job_id)
    if entry is not None:
        process = entry["process"]
        if process.poll() is None:
            return "running"
        return entry["meta"].get("status", "finished")
    meta = _read_meta(job_dir)
    return str(meta.get("status", "unknown"))


def _latest_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    evals: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload") or {}
        if kind == "update":
            latest = payload
        elif kind == "eval":
            evals.append(payload)
        elif kind == "done":
            final = payload
    return {"latest": latest, "evals": evals, "final": final, "events": len(events)}


def _list_train_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUNS_ROOT.is_dir():
        return runs
    for job_dir in sorted(RUNS_ROOT.iterdir(), key=lambda item: item.name, reverse=True):
        if not job_dir.is_dir():
            continue
        meta = _read_meta(job_dir)
        if meta.get("kind") != "train":
            continue
        events = _read_jsonl(job_dir / "events.jsonl")
        summary = _latest_from_events(events)
        runs.append(
            {
                **meta,
                "status": _job_status(meta.get("id", job_dir.name), job_dir),
                "latest": summary["latest"],
                "final_eval": summary["final"],
                "eval_count": len(summary["evals"]),
            }
        )
    return runs


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="path escapes the runs directory")
    return candidate


def _find_checkpoints(limit: int = 40) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in CHECKPOINT_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("checkpoint.pt"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                stat = resolved.stat()
            except OSError:
                continue
            eval_info = None
            summary_path = resolved.parent / "summary.json"
            if summary_path.is_file():
                try:
                    eval_info = json.loads(summary_path.read_text(encoding="utf-8")).get("final_eval")
                except json.JSONDecodeError:
                    eval_info = None
            found.append(
                {
                    "name": f"{resolved.parent.name}/checkpoint.pt",
                    "path": str(resolved),
                    "update": (eval_info or {}).get("update"),
                    "eval": eval_info,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                }
            )
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return found[:limit]


# ----------------------------------------------------------------------
# Health and dashboard
# ----------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "runs_root": str(RUNS_ROOT),
        "repo_root": str(REPO_ROOT),
        "jobs_running": sum(1 for job in _jobs.values() if job["process"].poll() is None),
    }


@app.get("/")
def index() -> FileResponse:
    page = FRONTEND_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=503, detail="dashboard frontend is not installed")
    return FileResponse(page)


# ----------------------------------------------------------------------
# Training runs
# ----------------------------------------------------------------------
@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return _list_train_runs()


@app.post("/api/runs")
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


@app.get("/api/runs/{run_id}")
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


@app.get("/api/runs/{run_id}/events")
def get_run_events(run_id: str, after: int = Query(default=0, ge=0)) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, run_id)
    events = _read_jsonl(job_dir / "events.jsonl")
    pending = [event for event in events if int(event.get("seq", 0)) > after]
    next_seq = max((int(event.get("seq", 0)) for event in events), default=after)
    return {"next": next_seq, "events": pending, "status": _job_status(run_id, job_dir)}


@app.post("/api/runs/{run_id}/stop")
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


# ----------------------------------------------------------------------
# Inference jobs (live sandbox)
# ----------------------------------------------------------------------
@app.post("/api/infer")
def start_infer(request: InferRequest) -> dict[str, Any]:
    checkpoint = request.checkpoint.strip()
    if checkpoint and checkpoint.lower() not in {"random", "none"}:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"checkpoint not found: {checkpoint}")
        checkpoint = str(path)
    else:
        checkpoint = "random"
    job_id, job_dir = _new_job_dir("infer")
    replays = job_dir / "replays"
    config = request.model_dump()
    config["checkpoint"] = checkpoint
    config["name"] = f"推理 {request.episodes} 局 × {request.opponent}"
    command = [
        _python(),
        "-m",
        "ironzero.training.record",
        "--checkpoint",
        checkpoint,
        "--episodes",
        str(request.episodes),
        "--opponent",
        request.opponent,
        "--seed",
        str(request.seed),
        "--mode",
        request.mode,
        "--max-decisions",
        str(request.max_decisions),
        "--device",
        "cpu",
        "--out-dir",
        str(replays),
        "--stream",
        str(job_dir / "frames.jsonl"),
        "--prefix",
        job_id,
    ]
    if request.map_id:
        command.extend(["--map-id", request.map_id])
    _spawn(job_id, job_dir, "infer", config, command)
    return {"id": job_id, "status": "running", "dir": str(job_dir)}


@app.get("/api/infer/{infer_id}")
def get_infer(infer_id: str) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, infer_id)
    meta = _read_meta(job_dir)
    if not meta:
        raise HTTPException(status_code=404, detail="inference job not found")
    stream = _read_jsonl(job_dir / "frames.jsonl")
    frames = [item for item in stream if item.get("type") == "frame"]
    episode_ends = [item for item in stream if item.get("type") == "episode_end"]
    done = any(item.get("type") == "done" for item in stream)
    return {
        **meta,
        "status": _job_status(infer_id, job_dir),
        "frames_total": len(frames),
        "latest_frame": frames[-1] if frames else None,
        "results": episode_ends,
        "episodes_done": len(episode_ends),
        "done": done,
    }


@app.get("/api/infer/{infer_id}/frames")
def get_infer_frames(
    infer_id: str, after: int = Query(default=0, ge=0)
) -> dict[str, Any]:
    job_dir = _resolve_inside(RUNS_ROOT, infer_id)
    stream = _read_jsonl(job_dir / "frames.jsonl")
    pending = stream[after:]
    return {
        "next": after + len(pending),
        "frames": pending,
        "status": _job_status(infer_id, job_dir),
    }


# ----------------------------------------------------------------------
# Replay library
# ----------------------------------------------------------------------
def _replay_index(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not root.is_dir():
        return items
    for path in root.rglob("replays/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = payload.get("meta") or {}
        relative = path.relative_to(root).as_posix()
        items.append(
            {
                "id": relative,
                "name": path.name,
                "checkpoint": meta.get("checkpoint"),
                "opponent": meta.get("opponent"),
                "seed": meta.get("seed"),
                "result": meta.get("result"),
                "return": meta.get("return"),
                "episodes": meta.get("episodes") or len(payload.get("episodes", [])),
                "frames": len(payload.get("frames", [])),
                "created_at": meta.get("created_at"),
                "mode": meta.get("mode"),
            }
        )
    items.sort(key=lambda item: (item.get("created_at") or "", item["id"]), reverse=True)
    return items


@app.get("/api/replays")
def list_replays() -> list[dict[str, Any]]:
    return _replay_index(RUNS_ROOT)


@app.get("/api/replays/{replay_id:path}")
def get_replay(replay_id: str) -> dict[str, Any]:
    path = _resolve_inside(RUNS_ROOT, replay_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="replay not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/replays/{replay_id:path}")
def delete_replay(replay_id: str) -> dict[str, Any]:
    path = _resolve_inside(RUNS_ROOT, replay_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="replay not found")
    path.unlink()
    return {"ok": True}


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
@app.get("/api/checkpoints")
def list_checkpoints() -> list[dict[str, Any]]:
    return _find_checkpoints()


# ----------------------------------------------------------------------
# Real game assets: maps, tilesets, unit sprites (local cache only)
# ----------------------------------------------------------------------
_map_list_cache: dict[str, Any] = {"at": 0.0, "payload": None}


@app.get("/api/maps")
def api_maps(force: bool = False) -> dict[str, Any]:
    now = time.time()
    if (
        force
        or _map_list_cache["payload"] is None
        or now - float(_map_list_cache["at"]) > 30.0
    ):
        available = gamemaps.assets_available()
        payload = {
            "available": available,
            "assets_root": str(gamemaps.ASSETS_ROOT),
            "maps": gamemaps.list_maps() if available else [],
            "units": gamemaps.unit_sprites() if available else {},
        }
        _map_list_cache["payload"] = payload
        _map_list_cache["at"] = now
    return _map_list_cache["payload"]


@app.get("/api/maps/{map_id}")
def api_map(map_id: str) -> JSONResponse:
    try:
        payload = gamemaps.load_map(map_id)
    except gamemaps.MapError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(payload)


@app.get("/api/assets/{asset_path:path}")
def api_asset(asset_path: str) -> FileResponse:
    try:
        path = gamemaps.resolve_asset(asset_path)
    except gamemaps.MapError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)



# Static frontend assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), check_dir=False), name="static")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    icon = FRONTEND_DIR / "favicon.svg"
    if icon.is_file():
        return FileResponse(icon)
    raise HTTPException(status_code=404)
