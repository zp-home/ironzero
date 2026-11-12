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
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .paths import BEIJING_TZ, CHECKPOINT_ROOTS, REPO_ROOT, RUNS_ROOT, RWPP_SERVICE

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


CHECKPOINT_FILENAMES: tuple[str, ...] = ("checkpoint.pt", "checkpoint_best.pt", "checkpoint_latest.pt")

#: ``normcase``d version of :data:`CHECKPOINT_FILENAMES` — set membership on a
#: listing is how the scans recognise a weight without a per-name ``is_file()``.
_CHECKPOINT_NAMES: frozenset[str] = frozenset(os.path.normcase(n) for n in CHECKPOINT_FILENAMES)

#: Sidecar files that carry the winning eval (and the map name) next to a checkpoint.
_CHECKPOINT_SUMMARIES: tuple[str, ...] = ("eval_report.json", "rwpp_summary.json", "summary.json", "metrics.json")

#: Files that a trainer appends to continuously while it runs. Used as the *second*
#: liveness signal (see :func:`_dir_is_active`): the host service knows about jobs it
#: started in its current process, but a restarted service forgets them even though
#: the trainer it spawned keeps writing. These files are the witness that survives.
_ACTIVITY_FILENAMES: tuple[str, ...] = ("live.jsonl", "train.log", "events.jsonl", "frames.jsonl")

#: ``normcase``d :data:`_ACTIVITY_FILENAMES`, so :func:`_job_scan` can spot them with
#: a set lookup while it already has each entry's ``stat`` in hand.
_ACTIVITY_NAMES: frozenset[str] = frozenset(os.path.normcase(n) for n in _ACTIVITY_FILENAMES)

#: How recently an activity file must have been touched to count as "running". PPO
#: writes one event per update, and an update is tens of seconds even on CPU, so two
#: minutes is comfortably inside "still training" and far outside "this job finished
#: an hour ago". A finished job is not protected by this rule at all.
ACTIVE_JOB_SECONDS = 120.0

#: How long a completed scan stays valid. Walking the tree is the expensive part
#: (18-27s over a Docker Desktop Windows bind mount), so a short in-process cache
#: turns every repeat call into a slice of a list.
CHECKPOINT_TTL_SECONDS = 25.0

_checkpoints_cache: list[dict[str, Any]] | None = None
_checkpoints_cache_at = 0.0
_checkpoints_cache_roots: tuple[str, ...] = ()
_checkpoints_lock = threading.Lock()


def _walk_checkpoint_files(root: Path) -> list[Path]:
    """Collect the checkpoint files under ``root`` in a single traversal.

    One pass per root replaces the previous three ``rglob`` passes (one per
    filename), which tripled the syscall count on slow bind-mounted trees.

    ``os.scandir`` is used instead of ``os.walk`` because each directory is then
    opened exactly once and the type/name of every entry comes from the directory
    listing itself. On the Docker Desktop Windows bind mount over the full run tree
    that is ~2.3s versus ~3.1s for ``os.walk``, with an identical result set.
    ``follow_symlinks=False`` matches the ``rglob``/``os.walk`` default of not
    following directory symlinks.

    The traversal is deliberately unbounded: checkpoints are currently always at
    ``ROOT/<category>/<job>/``, but capping the depth would silently hide a
    checkpoint written anywhere deeper, and a missing model is worse than a slow one.
    """

    wanted = {os.path.normcase(name) for name in CHECKPOINT_FILENAMES}
    found: list[Path] = []
    if not root.is_dir():
        return found
    stack: list[str] = [str(root)]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif os.path.normcase(entry.name) in wanted:
                            found.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return found


def _describe_checkpoint(path: Path, info: os.stat_result) -> dict[str, Any]:
    """Build one API entry from a checkpoint's stat plus its sidecar files."""

    eval_info = None
    map_name = ""
    for summary_name in _CHECKPOINT_SUMMARIES:
        summary_path = path.parent / summary_name
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
        live_path = path.parent / "live.jsonl"
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
        pname = path.parent.name.lower()
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
    return {
        "name": f"{path.parent.name}/{path.name}",
        "map": map_name,
        "path": str(path),
        "update": (eval_info or {}).get("update"),
        "eval": eval_info,
        "mtime": datetime.datetime.fromtimestamp(info.st_mtime, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "size_mb": round(info.st_size / 1024 / 1024, 2),
    }


def _scan_checkpoints() -> list[dict[str, Any]]:
    """Collect every checkpoint under the configured roots, newest first.

    ``Path.resolve()`` is deliberately avoided here: on the Docker Desktop Windows
    bind mount it costs ~6ms per call because it stats every path component, and it
    was only ever used as a dedup key. ``normcase(abspath())`` gives the same key
    for free, which took the non-listing part of the scan from ~2.8s to ~1.3s.
    """

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned: list[Path] = []
    for root in CHECKPOINT_ROOTS:
        if not root.is_dir():
            continue
        if any(other == root or other in root.parents for other in scanned):
            continue  # nested root (runs/ + runs/platform): already covered above
        scanned.append(root)

        for path in _walk_checkpoint_files(root):
            # Dedup on a normalised *key*, but keep the original Path: it carries the
            # real on-disk casing, and ``Path.stat`` is the single stat entry point
            # (also what the unreadable-file test monkeypatches).
            key = os.path.normcase(os.path.abspath(str(path)))
            if key in seen:
                continue
            seen.add(key)
            try:
                info = path.stat()
            except OSError:
                continue
            found.append(_describe_checkpoint(path, info))
    found.sort(key=lambda item: item["mtime"], reverse=True)
    return found


def _clear_checkpoint_cache() -> None:
    """Drop the cached scan result (tests, and any future admin endpoint)."""

    global _checkpoints_cache, _checkpoints_cache_at, _checkpoints_cache_roots
    with _checkpoints_lock:
        _checkpoints_cache = None
        _checkpoints_cache_at = 0.0
        _checkpoints_cache_roots = ()


def _all_checkpoints(force: bool = False) -> list[dict[str, Any]]:
    """Every checkpoint under ``CHECKPOINT_ROOTS``, mtime desc, **uncapped**.

    This is the shared bottom layer: ``GET /api/checkpoints`` slices it to 60,
    ``GET /api/models`` (model management) needs the whole list plus job context.
    Keeping one cached scan means the model manager cannot double the walk cost.

    Results are cached in-process for ``CHECKPOINT_TTL_SECONDS``; ``force=True``
    (``GET /api/checkpoints?force=true`` / ``GET /api/models?force=1``) rescans
    immediately so the "refresh weights" button never waits on a stale cache.
    This is written against a slow bind mount: without the cache a rescan costs
    tens of seconds on Docker Desktop, which used to time out the dashboard's 8s
    request budget.
    """

    global _checkpoints_cache, _checkpoints_cache_at, _checkpoints_cache_roots

    roots = tuple(str(root) for root in CHECKPOINT_ROOTS)
    with _checkpoints_lock:
        fresh = (
            _checkpoints_cache is not None
            and _checkpoints_cache_roots == roots
            and (time.monotonic() - _checkpoints_cache_at) < CHECKPOINT_TTL_SECONDS
        )
        if force or not fresh:
            _checkpoints_cache = _scan_checkpoints()
            _checkpoints_cache_at = time.monotonic()
            _checkpoints_cache_roots = roots
        found = _checkpoints_cache
    # Copies keep callers from mutating the cached entries (``eval`` is shared).
    return [dict(item) for item in found]


def _find_checkpoints(limit: int = 60, force: bool = False) -> list[dict[str, Any]]:
    """List the newest checkpoints under ``CHECKPOINT_ROOTS`` (mtime desc).

    Thin ``limit`` wrapper over :func:`_all_checkpoints`; the default of 60 is
    pinned by ``tests/test_web_checkpoints.py``.
    """

    return _all_checkpoints(force=force)[:limit]


def _job_dirs(root: Path) -> list[Path]:
    """Return the ``root/<category>/<job>`` directories that hold job data.

    Every real job lives exactly two levels below the runs root (``runs``), e.g.
    ``runs/platform/rwpp-ppo-ppo-5a5fc499`` or
    ``runs/experiments/bc_ironzero25``; the platform's own ``RUNS_ROOT`` is
    ``runs/platform``, so its jobs are one level below it. Deeper trees are not
    job directories and are ignored. Unreadable entries are skipped instead of
    failing the request, like the checkpoint walk does.

    ``os.scandir`` is used rather than ``iterdir``/``is_dir()`` because it answers
    "is this a directory" from the listing itself: on the bind mount each extra
    ``stat`` costs ~3ms, which over ~800 entries was seconds per call.
    """

    dirs: list[Path] = []
    try:
        with os.scandir(root) as categories:
            category_paths = sorted(
                (entry.path for entry in categories if entry.is_dir(follow_symlinks=False)),
            )
    except OSError:
        return dirs
    for category_path in category_paths:
        try:
            with os.scandir(category_path) as children:
                child_paths = sorted(
                    (entry.path for entry in children if entry.is_dir(follow_symlinks=False)),
                )
        except OSError:
            continue
        dirs.extend(Path(path) for path in child_paths)
    return dirs


def _job_scan(root: Path, job_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """Describe every ``root/<category>/<job>`` directory in a single traversal.

    Each entry carries everything the model manager needs about one job directory:

    * ``relative`` (posix ``<category>/<job>``), ``category``, ``name``, ``path``;
    * ``names`` — the *top-level* entry names, normalised with :func:`os.path.normcase`.
      This is what turns the marker-file check into a set lookup instead of 9
      ``is_file()`` calls per directory;
    * ``checkpoints`` — the canonical :data:`CHECKPOINT_FILENAMES` present at top level;
    * ``size_bytes`` — total of every regular file below the directory;
    * ``mtime`` — newest of the directory's own mtime and its top-level checkpoints,
      matching what the panel sorts by;
    * ``activity_mtime`` — newest top-level write to any :data:`_ACTIVITY_FILENAMES`
      file (``0.0`` when there is none). Free here: the traversal already stats every
      entry, so this is a set lookup rather than the four extra ``stat`` calls a
      per-directory probe would cost (~5.1s over 400 dirs on the bind mount);
    * ``meta`` — the parsed ``meta.json``, or ``None`` when the job has none.

    Whether a *host-side* job is still running is not derivable from the directory
    at all (those jobs write no ``meta.json``); callers that need it consult
    :func:`_rwpp_job_dirs`, which asks the host service that owns that state.

    All of it comes out of one ``os.scandir`` pass: the marker probe, the recursive
    size and the mtime were previously three separate traversals (3.4s + 5.4s +
    4.3s over the 444 job dirs of the real tree on the Docker Desktop Windows bind
    mount), while the single pass costs ~4.3s. ``DirEntry.stat()`` is used because
    the directory listing already carries size/mtime, so it costs no extra syscall.

    ``job_dirs`` lets a caller reuse an enumeration it already has.
    """

    entries: list[dict[str, Any]] = []
    for job_dir in _job_dirs(root) if job_dirs is None else job_dirs:
        try:
            relative = job_dir.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - _job_dirs only yields children
            continue
        try:
            dir_mtime = job_dir.stat().st_mtime
        except OSError:
            dir_mtime = 0.0
        names: set[str] = set()
        checkpoint_mtimes: dict[str, float] = {}
        activity_mtime = 0.0
        size_bytes = 0
        stack: list[tuple[str, bool]] = [(str(job_dir), True)]
        while stack:
            directory, top = stack.pop()
            try:
                with os.scandir(directory) as children:
                    for child in children:
                        try:
                            is_dir = child.is_dir(follow_symlinks=False)
                            info = child.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if is_dir:
                            stack.append((child.path, False))
                            continue
                        key = os.path.normcase(child.name)
                        size_bytes += info.st_size
                        if top:
                            names.add(key)
                            if key in _CHECKPOINT_NAMES:
                                checkpoint_mtimes[key] = info.st_mtime
                            elif key in _ACTIVITY_NAMES and info.st_mtime > activity_mtime:
                                activity_mtime = info.st_mtime
            except OSError:
                continue
        mtime = dir_mtime
        if checkpoint_mtimes:
            mtime = max(mtime, *checkpoint_mtimes.values())
        parts = relative.split("/")
        entries.append(
            {
                "path": job_dir,
                "relative": relative,
                "category": parts[0] if len(parts) > 1 else "",
                "name": parts[-1],
                "names": names,
                "checkpoints": [
                    name for name in CHECKPOINT_FILENAMES if os.path.normcase(name) in names
                ],
                "size_bytes": size_bytes,
                "mtime": mtime,
                "activity_mtime": activity_mtime,
                "meta": _read_meta(job_dir) if os.path.normcase("meta.json") in names else None,
            }
        )
    return entries


def _job_meta_index(root: Path, job_dirs: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    """Map ``<category>/<job>`` (posix, relative to ``root``) to its meta.json.

    Only directories that actually carry a ``meta.json`` appear, so callers can
    treat a hit as "this is a supervised job directory". Reading every meta is a
    few hundred small reads; the checkpoint scan cache already spares the walk.

    ``job_dirs`` lets a caller that already listed the tree pass its own list, so
    the (expensive) directory enumeration is not repeated per caller.
    """

    index: dict[str, dict[str, Any]] = {}
    for job_dir in (job_dirs if job_dirs is not None else _job_dirs(root)):
        meta_path = os.path.join(str(job_dir), "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            relative = job_dir.relative_to(root).as_posix()
        except ValueError:
            continue
        index[relative] = _read_meta(job_dir)
    return index


def _activity_mtime_is_live(stamp: float) -> bool:
    """True when ``stamp`` is inside the :data:`ACTIVE_JOB_SECONDS` window."""

    return stamp > 0.0 and (time.time() - stamp) < ACTIVE_JOB_SECONDS


def _dir_is_active(job_dir: Path) -> bool:
    """True when a trainer is (or very recently was) writing into ``job_dir``.

    Two signals, because neither alone is sufficient:

    * the host service that owns real-game jobs reports them at
      ``GET {RWPP_SERVICE}/jobs`` — authoritative, but it forgets a job as soon as
      it restarts, while the trainer it spawned may still be running;
    * the trainer keeps appending to :data:`_ACTIVITY_FILENAMES` — survives service
      restarts, but only proves "recently active", hence the
      :data:`ACTIVE_JOB_SECONDS` window.

    For the *delete* path the conservative answer is wanted: something touched
    within the window is treated as running, because deleting the weights a live
    trainer is about to overwrite destroys an in-progress run.

    This single-directory variant costs four ``stat`` calls; callers that already
    ran :func:`_job_scan` should use its ``activity_mtime`` instead (free).
    """

    if _host_job_id(job_dir):
        return True
    newest = 0.0
    for name in _ACTIVITY_FILENAMES:
        try:
            info = os.stat(os.path.join(str(job_dir), name))
        except OSError:
            continue
        if info.st_mtime > newest:
            newest = info.st_mtime
    return _activity_mtime_is_live(newest)


def _host_job_id(job_dir: Path) -> str:
    """Id of the running host-side job that owns ``job_dir``; ``""`` when none."""

    running = _rwpp_job_dirs()
    if not running:
        return ""
    try:
        relative = job_dir.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return ""
    return running.get(relative.lower(), "")


def _rwpp_job_dirs(timeout: float = 3.0) -> dict[str, str]:
    """Repo-relative ``out_dir`` (lowercased) -> job id, for *running* host jobs.

    Read straight from the host real-game service rather than through
    ``routers/rwpp.py`` so this module keeps no dependency on the HTTP layer. Any
    transport/shape problem yields ``{}``: the caller treats that as "nothing known
    to be running".
    """

    url = RWPP_SERVICE.rstrip("/") + "/jobs"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - see docstring: fail open
        return {}
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        return {}
    running: dict[str, str] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if str(job.get("status") or "").lower() != "running":
            continue
        out_dir = str(job.get("out_dir") or "").replace("\\", "/").strip("/")
        if out_dir:
            running[out_dir.lower()] = str(job.get("id") or "")
    return running


def _dir_size_bytes(path: Path) -> int:
    """Total size of the regular files below ``path`` (``0`` when unreadable)."""

    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.stat(os.path.join(dirpath, filename)).st_size
            except OSError:
                continue
    return total
