# SPDX-License-Identifier: Apache-2.0
"""Job storage helpers: job directories, meta files, subprocess supervision.

A "job" is a directory under ``RUNS_ROOT`` holding ``meta.json`` (status and
config), ``stdout.log`` (child output) plus whatever artifacts the worker
writes (``events.jsonl`` for trainers, ``frames.jsonl`` for inference).

Lifecycle: ``_new_job_dir`` allocates the id, ``_spawn`` starts the child and
flips status to ``running``, a watcher thread records ``succeeded``/``failed``
and the exit code when the child exits, and ``_job_status`` answers "is it
still alive?" for the API. Stopping a job goes through ``Popen.terminate()`` in
the router, which then rewrites the meta status to ``stopped``.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from .paths import BEIJING_TZ, CHECKPOINT_ROOTS, REPO_ROOT, RUNS_ROOT

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _new_job_dir(kind: str) -> tuple[str, Path]:
    job_id = f"{kind}-{datetime.datetime.now(BEIJING_TZ).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
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


def _find_checkpoints(limit: int = 60) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in CHECKPOINT_ROOTS:
        if not root.is_dir():
            continue
        for pattern in ("checkpoint.pt", "checkpoint_best.pt", "checkpoint_latest.pt"):
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    stat = resolved.stat()
                except OSError:
                    continue
                eval_info = None
                map_name = ""
                for summary_name in ("eval_report.json", "rwpp_summary.json", "summary.json", "metrics.json"):
                    summary_path = resolved.parent / summary_name
                    if summary_path.is_file():
                        try:
                            data = json.loads(summary_path.read_text(encoding="utf-8"))
                            eval_info = data.get("final_eval") or data.get("best_eval") or (data if "win_rate" in data else None)
                            if not map_name:
                                map_name = data.get("map") or (data.get("game") or {}).get("map") or ""
                            if eval_info:
                                break
                        except (json.JSONDecodeError, OSError):
                            continue
                if not map_name:
                    live_path = resolved.parent / "live.jsonl"
                    if live_path.is_file():
                        try:
                            with live_path.open("r", encoding="utf-8") as f:
                                first_line = f.readline()
                                if first_line:
                                    payload = json.loads(first_line).get("payload", {})
                                    map_name = (payload.get("config") or {}).get("map_name") or ""
                        except Exception:
                            pass
                if not map_name:
                    pname = resolved.parent.name.lower()
                    if "ironzero25" in pname:
                        map_name = "IronZero25"
                    elif "ironzero10" in pname:
                        map_name = "IronZero10"
                    elif "ironzero5" in pname:
                        map_name = "IronZero5"
                    elif "ironzero15" in pname:
                        map_name = "IronZero15"
                    elif "ironzero20" in pname:
                        map_name = "IronZero20"
                found.append(
                    {
                        "name": f"{resolved.parent.name}/{resolved.name}",
                        "map": map_name,
                        "path": str(resolved),
                        "update": (eval_info or {}).get("update"),
                        "eval": eval_info,
                        "mtime": datetime.datetime.fromtimestamp(stat.st_mtime, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                        "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    }
                )
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return found[:limit]
