# SPDX-License-Identifier: Apache-2.0
"""Model management: list checkpoints and delete expired weights / job dirs.

The inference and training pages let the user pick a policy checkpoint, but until
now nothing could clean the 70+ checkpoint files that accumulate under ``runs/``.
This router is the delete side of that panel, and it is deliberately paranoid
because the operation is irreversible:

* every path goes through :func:`paths._resolve_inside`, so ``..`` and absolute
  paths cannot escape the runs root;
* a single-file delete only accepts the three checkpoint filenames
  (:data:`jobs.CHECKPOINT_FILENAMES`) — never a metrics file, replay or log;
* a whole-job delete only accepts ``<runs root>/<category>/<job>`` (one or two
  levels below the root) and refuses the root itself plus any *category*
  directory (``platform``/``archive``/``experiments``/…), no matter how deep;
* running jobs and BC warmstart weights are protected (``409``) — see
  :func:`_protection_index` for the exact rules;
* the in-process scan cache is invalidated after every successful delete, so the
  list a UI re-fetches right after a delete is never stale.

Root selection (:func:`_models_root`): the platform's ``RUNS_ROOT`` is
``runs/platform`` in the shipped configuration, i.e. one of the *categories*, so
deleting relative to it would make the other categories unreachable. When
``RUNS_ROOT`` names a category, the manageable root is its parent (``runs``), so
the API's ``path`` is always ``<category>/<job>/checkpoint.pt``.
"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .. import jobs as job_store
from .. import paths

router = APIRouter()

#: Directory names that are *containers of jobs*, never a job themselves. Matched
#: against the single path segment right below the manageable root.
CATEGORY_DIR_NAMES: frozenset[str] = frozenset(
    {
        "platform",
        "archive",
        "experiments",
        "recordings",
        "_logs",
        "_scratch",
        "_rwpp_ppo",
        "_rwpp_perf",
        "logs",
    }
)

#: Job names that look like throwaway experiments ("smoke"/"bench"/"probe"/…).
#: Only used for the ``tmp_mb`` estimate the UI shows as "cleanable", never to
#: delete anything by itself.
#:
#: Two deliberate decisions:
#:
#: * ``econ`` is **not** in the list: it matches ``experiments/econ_train_full`` and
#:   ``econ_train_shaping*``, the economy training experiments this project cares
#:   about. Labelling those "temporary" invites the panel's bulk cleanup to delete
#:   them.
#: * the keyword may be preceded by nothing, a separator *or* a digit prefix, and
#:   may be followed by nothing, a separator *or* a digit suffix, so real names like
#:   ``_rwpp_ppo_smoke``, ``_rwpp_ppo_smoke2``, ``_terrain_test`` and
#:   ``rwpp_ppo_bench3`` are all recognised.
_TEMP_JOB_RE = re.compile(
    r"(?:^|[-_.])"
    r"(?:smoke|bench|probe|tmp|temp|scratch|verify|debug|trial)$|"
    r"(?:^|[-_.])(?:smoke|bench|probe|tmp|temp|scratch|verify|debug|trial|test)(?:[-_.]|\d|$)",
    re.IGNORECASE,
)

#: Files that mark a directory as a *job* (as opposed to an asset/cache dir that
#: happens to live under ``runs/``).
_JOB_MARKER_FILES: tuple[str, ...] = (
    "meta.json",
    "live.jsonl",
    "events.jsonl",
    "frames.jsonl",
    "train.log",
    "rwpp_summary.json",
    "eval_report.json",
    "metrics.json",
    "summary.json",
)


_models_cache: dict[str, Any] | None = None
_models_cache_at = 0.0
_models_cache_key: tuple[str, ...] = ()

#: Cache for "which host-side real-game jobs are running" (a small HTTP call to the
#: host service). Kept short so a just-started job becomes undeletable promptly.
_HOST_JOBS_TTL_SECONDS = 5.0
_host_jobs_cache: dict[str, str] | None = None
_host_jobs_cache_at = 0.0
_host_jobs_lock = threading.Lock()

#: How long a full model-management payload stays valid. Sizeable directories are
#: walked for their sizes, so this mirrors the checkpoint scan TTL.
MODELS_TTL_SECONDS = job_store.CHECKPOINT_TTL_SECONDS


# --------------------------------------------------------------------------- #
# roots & path safety
# --------------------------------------------------------------------------- #
def _models_root() -> Path:
    """The directory that ``path`` in this API is relative to.

    ``IRONZERO_MODELS_ROOT`` wins when set (handy for tests and unusual
    deployments). Otherwise ``RUNS_ROOT`` is used, except when it *is* a category
    directory (``runs/platform`` in the shipped config) — then its parent is the
    manageable root so the sibling categories stay addressable.
    """

    override = os.environ.get("IRONZERO_MODELS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    runs_root = Path(paths.RUNS_ROOT)
    if runs_root.name.lower() in CATEGORY_DIR_NAMES:
        parent = runs_root.parent
        if parent != runs_root:
            return parent
    return runs_root


def _mb(size_bytes: float) -> float:
    """Bytes -> megabytes, always a float so JSON clients can format uniformly.

    ``round()`` returns an ``int`` for whole numbers, which would make the
    frontend's ``toFixed`` calls fail intermittently; keeping every size a float
    makes the response shape predictable.
    """

    return round(float(size_bytes) / 1024 / 1024, 2)


def _managed_relative(root: Path, target: Path) -> str | None:
    """``target`` as a posix path relative to ``root``, or ``None`` if outside."""

    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _resolve_managed(relative: str) -> tuple[Path, str]:
    """Resolve an API ``path`` against the manageable root, rejecting escapes."""

    root = _models_root()
    cleaned = str(relative or "").replace("\\", "/").strip("/")
    if not cleaned:
        raise HTTPException(status_code=400, detail="path must not be empty")
    # ``_resolve_inside`` raises 400 for ``..``/absolute escapes.
    target = paths._resolve_inside(root, cleaned)
    if target == root.resolve():
        raise HTTPException(status_code=400, detail="refusing to touch the runs root itself")
    normalised = _managed_relative(root, target)
    if normalised is None:  # pragma: no cover - _resolve_inside already guards this
        raise HTTPException(status_code=400, detail="path escapes the runs directory")
    return target, normalised


def _is_checkpoint_filename(name: str) -> str | None:
    """Return the canonical whitelisted filename for ``name``, else ``None``."""

    wanted = {os.path.normcase(item): item for item in job_store.CHECKPOINT_FILENAMES}
    return wanted.get(os.path.normcase(name))


def _looks_like_category(directory: Path) -> bool:
    """True when ``directory`` contains job sub-directories rather than job data.

    Structural counterpart to :data:`CATEGORY_DIR_NAMES`: a directory holding
    child directories that themselves look like jobs is a container of jobs.
    Checked so a renamed/extra category cannot be deleted even though it is not
    in the hard-coded name list.
    """

    try:
        children = list(directory.iterdir())
    except OSError:
        return True  # unreadable: assume the worst and refuse
    for child in children:
        if not child.is_dir():
            continue
        if (child / "meta.json").is_file():
            return True
        for filename in job_store.CHECKPOINT_FILENAMES:
            if (child / filename).is_file():
                return True
    return False


def _is_job_dir(directory: Path) -> bool:
    """True when ``directory`` holds job data (marker files or a checkpoint).

    ``runs/`` is not exclusively job output — extracted archives and caches also
    land there — so a bare sub-directory is not enough to be listed as a job.
    Kept for single-directory callers; the list endpoints classify whole trees with
    :func:`jobs._job_scan`, which answers the same question from one ``scandir``.
    """

    for name in _JOB_MARKER_FILES:
        if (directory / name).is_file():
            return True
    for name in job_store.CHECKPOINT_FILENAMES:
        if (directory / name).is_file():
            return True
    return False


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def _job_index(
    root: Path,
    job_dirs: list[Path] | None = None,
    scan: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """``<category>/<job>`` (posix) -> meta.json payload for every job dir.

    ``scan`` (from :func:`jobs._job_scan`) already carries the parsed meta for every
    job directory, so a caller that has one pays nothing extra; ``job_dirs`` and the
    bare call fall back to a dedicated meta pass.
    """

    if scan is not None:
        return {entry["relative"]: entry["meta"] for entry in scan if entry["meta"] is not None}
    return job_store._job_meta_index(root, job_dirs)


def _live_job_dirs() -> set[str]:
    """Job dirs whose child process is still alive in *this* process.

    ``meta.json`` says ``running`` even after a container restart killed the
    child, and ``_jobs`` is the only source of truth for "actually alive now".
    """

    live: set[str] = set()
    with job_store._jobs_lock:
        entries = list(job_store._jobs.values())
    for entry in entries:
        process = entry.get("process")
        try:
            if process is not None and process.poll() is None:
                directory = entry.get("dir")
                if directory is not None:
                    live.add(Path(directory).name)
        except OSError:  # pragma: no cover - poll() on a reaped handle
            continue
    return live


def _host_running_jobs() -> dict[str, str]:
    """Repo-relative ``out_dir`` (lowercased) -> id of *running* host-side jobs.

    Cached for :data:`_HOST_JOBS_TTL_SECONDS` because the model listing and every
    protection check would otherwise hit the host service repeatedly. A failed lookup
    is cached too, so a down service cannot turn into a per-request timeout.

    Keys are **repo-relative** because that is how the host service reports
    ``out_dir`` (``runs/platform/rwpp-ppo-<id>``). Callers compare against
    :func:`_repo_relative_key` rather than against the models-root-relative path.
    """

    global _host_jobs_cache, _host_jobs_cache_at
    now = time.monotonic()
    with _host_jobs_lock:
        if _host_jobs_cache is not None and (now - _host_jobs_cache_at) < _HOST_JOBS_TTL_SECONDS:
            return dict(_host_jobs_cache)
    running = job_store._rwpp_job_dirs()
    with _host_jobs_lock:
        _host_jobs_cache = running
        _host_jobs_cache_at = time.monotonic()
        return dict(running)


def _clear_host_jobs_cache() -> None:
    """Forget the host-job lookup (called after a successful delete)."""

    global _host_jobs_cache, _host_jobs_cache_at
    with _host_jobs_lock:
        _host_jobs_cache = None
        _host_jobs_cache_at = 0.0


def _repo_relative_key(root_prefix: str | None, relative: str) -> str:
    """Repo-relative lowercase key for ``<models root>/`` + ``relative``.

    The host service reports ``out_dir`` relative to the repo root, while the models
    API speaks paths relative to the models root; in the shipped configuration those
    differ by one level (``runs/platform/x`` vs ``platform/x``). Both sides are
    normalised to the repo-relative form here. ``root_prefix`` is ``None`` when the
    models root sits outside the repo, in which case nothing can match a host job.
    """

    if root_prefix is None:
        return ""
    return f"{root_prefix}/{relative}".lower() if root_prefix else relative.lower()


def _repo_prefix(root: Path) -> str | None:
    """``root`` as a repo-relative posix prefix, or ``None`` when it is outside."""

    try:
        prefix = root.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return None
    return "" if prefix == "." else prefix


def _dir_is_active(job_dir: Path) -> bool:
    """Single-directory liveness check for the delete endpoints."""

    return job_store._dir_is_active(job_dir)


def _activity_in_dir(job_dir: Path) -> bool:
    """True when an activity file in ``job_dir`` was touched within the window.

    Four ``stat`` calls, so only use it for a single directory (a delete). Whole-tree
    callers read ``activity_mtime`` from :func:`jobs._job_scan` instead, which gets
    the same answer out of the traversal it is already doing.
    """

    newest = 0.0
    for name in job_store._ACTIVITY_FILENAMES:
        try:
            info = os.stat(os.path.join(str(job_dir), name))
        except OSError:
            continue
        if info.st_mtime > newest:
            newest = info.st_mtime
    return job_store._activity_mtime_is_live(newest)


def _normalise_reference(value: Any) -> str:
    """Normalise a checkpoint reference from a meta/config for comparison."""

    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = paths.REPO_ROOT / text
    try:
        return os.path.normcase(str(candidate.resolve()))
    except OSError:  # pragma: no cover - resolve() rarely fails
        return os.path.normcase(str(candidate))


def _referenced_checkpoints(job_index: dict[str, dict[str, Any]]) -> dict[str, str]:
    """``normalised checkpoint path`` -> ``"<category>/<job>"`` that references it.

    PPO jobs record ``init_checkpoint`` in their config; the host-side service
    only recently started passing it through, so this is a bonus signal on top of
    the BC-warmstart rule below.
    """

    referenced: dict[str, str] = {}
    for relative, meta in job_index.items():
        config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        candidates = [
            config.get("init_checkpoint"),
            meta.get("init_checkpoint"),
            config.get("checkpoint"),
            meta.get("checkpoint"),
        ]
        for value in candidates:
            key = _normalise_reference(value)
            if key:
                referenced.setdefault(key, relative)
    return referenced


def _is_bc_warmstart(relative_path: str) -> bool:
    """True for ``<category>/bc_ironzero<N>/checkpoint*.pt`` at any top level.

    ``resolve_bc_checkpoint()`` in ``scripts/rwpp_service/ppo.py`` loads these as
    the warmstart weights; deleting one would silently downgrade every future PPO
    run to "from scratch".
    """

    parts = [part for part in relative_path.split("/") if part]
    if not parts:
        return False
    return bool(re.match(r"^bc_ironzero\d+$", parts[-2] if len(parts) >= 2 else "", re.IGNORECASE))


def _job_is_bc_warmstart(job_relative: str) -> bool:
    parts = [part for part in job_relative.split("/") if part]
    return bool(parts) and bool(re.match(r"^bc_ironzero\d+$", parts[-1], re.IGNORECASE))


def _newest_checkpoint_job(root: Path) -> str | None:
    """``<category>/<job>`` holding the newest checkpoint, or ``None``.

    Mirrors the single-file protection rule: the newest weight is the dropdown
    default, so deleting its *directory* must be refused as well — otherwise the
    job-level route would quietly do what the file-level route forbids.

    The root is resolved once: :func:`_managed_relative` would resolve both sides per
    entry, i.e. ~2s of ``stat`` on the bind mount for the real tree.
    """

    try:
        root_resolved = root.resolve()
    except OSError:  # pragma: no cover - resolve() rarely fails
        root_resolved = root
    newest: str | None = None
    newest_mtime = ""
    for entry in job_store._all_checkpoints():
        try:
            relative = Path(os.path.abspath(str(entry["path"]))).relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue
        if "/" not in relative:
            continue
        mtime = str(entry.get("mtime") or "")
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = relative.rsplit("/", 1)[0]
    return newest


def _job_protection(root: Path, relative: str) -> str:
    """Why ``<category>/<job>`` may not be deleted; ``""`` when it may.

    Single-directory variant of :func:`_scan_job_dirs`'s protection rules: a
    delete should not have to stat every job directory just to answer "is this
    one protected?"
    """

    if _job_is_bc_warmstart(relative):
        return "BC 预热权重目录（PPO warmstart 依赖）"
    job_dir = root / relative
    meta = job_store._read_meta(job_dir)
    status = str(meta.get("status") or "")
    name = relative.rsplit("/", 1)[-1]
    if status == "running" or name in _live_job_dirs():
        return "作业正在运行（status=running）"
    if _dir_is_active(job_dir):
        host_id = _host_running_jobs().get(_repo_relative_key(_repo_prefix(root), relative), "")
        if host_id:
            return f"作业正在运行（host 真机任务 {host_id}）"
        if _activity_in_dir(job_dir):
            return "作业正在运行（训练日志仍在写入）"
        return "作业正在运行"
    newest = _newest_checkpoint_job(root)
    if newest and newest.lower() == relative.lower():
        return "含最新权重（下拉框默认首选，删目录等于删它）"
    return ""


def _scan_job_dirs(
    root: Path,
    newest_job: str | None = None,
    job_dirs: list[Path] | None = None,
    include_sizes: bool = True,
    scan: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Describe every job-level directory below ``root`` (sizes included).

    Directories without any :data:`_JOB_MARKER_FILES` entry and without a
    checkpoint are skipped: they are extracted assets/caches, not jobs, and
    offering them as "delete this job" would be a data-loss trap.

    ``newest_job`` is the ``<category>/<job>`` that holds the newest checkpoint
    (the dropdown's default). Its directory is protected for the same reason the
    single weight is: deleting the directory deletes that weight.

    ``scan`` (:func:`jobs._job_scan` output) is reused when the caller already has
    it — it carries the top-level names, sizes, mtimes and parsed metas of every job
    directory from one traversal, so this function only classifies and sorts.
    ``include_sizes=False`` blanks ``size_mb`` for callers that only need the
    protection flags (a delete), which keeps them off the recursive size walk.
    """

    if newest_job is None:
        newest_job = _newest_checkpoint_job(root)
    if scan is None:
        scan = job_store._job_scan(root, job_dirs)
    live = _live_job_dirs()
    host_running = _host_running_jobs()
    root_prefix = _repo_prefix(root)
    markers = {os.path.normcase(name) for name in _JOB_MARKER_FILES}
    described: list[dict[str, Any]] = []
    for entry in scan:
        names = entry["names"]
        if not entry["checkpoints"] and not (names & markers):
            continue
        relative = entry["relative"]
        name = entry["name"]
        meta = entry["meta"] or {}
        status = str(meta.get("status") or ("running" if name in live else "unknown"))
        host_id = host_running.get(_repo_relative_key(root_prefix, relative)) or None
        # A host trainer that the (possibly restarted) host service no longer reports
        # still shows up through its freshly-written activity file.
        trailing = host_id is None and job_store._activity_mtime_is_live(
            entry["activity_mtime"]
        )
        if name in live or host_id is not None or trailing:
            status = "running"
        reason = ""
        if _job_is_bc_warmstart(relative):
            reason = "BC 预热权重目录（PPO warmstart 依赖）"
        elif host_id is not None:
            reason = f"作业正在运行（host 真机任务 {host_id}）"
        elif trailing:
            reason = "作业正在运行（训练日志仍在写入）"
        elif status == "running":
            reason = "作业正在运行（status=running）"
        elif newest_job and relative.lower() == newest_job.lower():
            reason = "含最新权重（下拉框默认首选，删目录等于删它）"
        described.append(
            {
                "job_dir": relative,
                "category": entry["category"],
                "name": name,
                "size_mb": _mb(entry["size_bytes"]) if include_sizes else 0.0,
                "mtime": (
                    datetime.datetime.fromtimestamp(
                        entry["mtime"], paths.BEIJING_TZ
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    if entry["mtime"]
                    else ""
                ),
                "status": status,
                "protected": bool(reason),
                "protected_reason": reason,
                "tmp": bool(_TEMP_JOB_RE.search(name)),
                "checkpoints": list(entry["checkpoints"]),
            }
        )
    described.sort(key=lambda item: item["mtime"], reverse=True)
    return described


def _describe_models(
    root: Path,
    entries: list[dict[str, Any]],
    job_dirs: list[Path] | None = None,
    scan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join the checkpoint scan with job context and protection rules."""

    if scan is None:
        scan = job_store._job_scan(root, job_dirs)
    job_index = _job_index(root, scan=scan)
    referenced = _referenced_checkpoints(job_index)
    live = _live_job_dirs()
    host_running = _host_running_jobs()
    # Job dirs whose activity file is still being written. ``activity_mtime`` comes
    # from the traversal ``_job_scan`` already did, so this costs nothing per job.
    active_dirs = {
        entry["relative"]
        for entry in scan
        if job_store._activity_mtime_is_live(entry["activity_mtime"])
    }
    # ``_managed_relative`` resolves both sides, which stats every path component
    # (~6ms per call on the bind mount). The root never moves during a request, and
    # every checkpoint path was already built from a scan of that root, so resolving
    # it once here is enough; the per-entry mapping below is a plain string prefix.
    root_resolved = root.resolve()
    root_prefix = _repo_prefix(root)

    managed: list[dict[str, Any]] = []
    for entry in entries:
        absolute = Path(entry["path"])
        try:
            relative = Path(os.path.abspath(str(absolute))).relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue  # outside the manageable root: not deletable through this API
        job_relative = relative.rsplit("/", 1)[0] if "/" in relative else ""
        parts = job_relative.split("/") if job_relative else []
        job_name = parts[-1] if parts else ""
        category = parts[0] if len(parts) > 1 else ""
        meta = job_index.get(job_relative, {})
        status = str(meta.get("status") or ("running" if job_name in live else "unknown"))
        host_id = host_running.get(_repo_relative_key(root_prefix, job_relative)) or None
        trailing = host_id is None and job_relative in active_dirs
        if job_name in live or host_id is not None or trailing:
            status = "running"
        in_use = status == "running"
        eval_info = entry.get("eval") if isinstance(entry.get("eval"), dict) else {}
        win_rate = eval_info.get("win_rate")
        managed.append(
            {
                "path": relative,
                "job_dir": job_relative,
                "job_name": job_name,
                "category": category,
                "name": entry.get("name") or f"{job_name}/{absolute.name}",
                "file": absolute.name,
                "size_mb": entry.get("size_mb", 0.0),
                "mtime": entry.get("mtime", ""),
                "map": entry.get("map", ""),
                "update": entry.get("update"),
                "win_rate": win_rate if isinstance(win_rate, (int, float)) else None,
                "status": status,
                "in_use": in_use,
                "protected": False,
                "protected_reason": "",
                "tmp": bool(job_name and _TEMP_JOB_RE.search(job_name)),
                "_absolute": str(absolute),
                "_host_id": host_id,
                "_trailing": trailing,
                # Same key ``_normalise_reference`` would produce, but built from the
                # already-resolved root instead of resolving every entry again (~6ms
                # per ``Path.resolve()`` on the bind mount, 167 entries per payload).
                "_normalised": os.path.normcase(str(root_resolved / relative)),
            }
        )

    # The newest checkpoint is the dropdown default, so it is always protected:
    # deleting it would silently change what the next inference run loads.
    newest_key = ""
    if managed:
        newest_key = max(managed, key=lambda item: str(item.get("mtime") or ""))["_normalised"]

    for item in managed:
        reason = ""
        if _is_bc_warmstart(item["path"]):
            reason = "BC 预热权重（PPO warmstart 依赖，删除会让训练退回从零）"
        elif item["in_use"]:
            if item["_host_id"]:
                reason = f"作业正在运行（host 真机任务 {item['_host_id']}）"
            elif item["_trailing"]:
                reason = "作业正在运行（训练日志仍在写入）"
            else:
                reason = "作业正在运行（status=running）"
        elif item["_normalised"] in referenced:
            reason = f"被作业 {referenced[item['_normalised']]} 引用为初始权重"
        elif item["_normalised"] == newest_key:
            reason = "最新权重（下拉框默认首选）"
        item["protected"] = bool(reason)
        item["protected_reason"] = reason

    checkpoints_total_mb = _mb(sum(float(item.get("size_mb") or 0.0) for item in managed) * 1024 * 1024)
    return {
        "root": _display_root(root),
        "count": len(managed),
        "total_mb": checkpoints_total_mb,
        "items": managed,
    }


def _display_root(root: Path) -> str:
    """Human-readable root (repo-relative when possible) for the UI, posix style."""

    relative = _managed_relative(paths.REPO_ROOT, root)
    if relative is not None:
        return relative
    return Path(root).as_posix()


def _models_payload(force: bool = False) -> dict[str, Any]:
    """Full ``GET /api/models`` payload, cached for ``MODELS_TTL_SECONDS``."""

    global _models_cache, _models_cache_at, _models_cache_key

    root = _models_root()
    key = (
        str(root),
        *(str(item) for item in job_store.CHECKPOINT_ROOTS),
    )
    if (
        not force
        and _models_cache is not None
        and _models_cache_key == key
        and (time.monotonic() - _models_cache_at) < MODELS_TTL_SECONDS
    ):
        return copy.deepcopy(_models_cache)

    # Two independent full-tree walks. Each is latency-bound on the bind mount (~7s and
    # ~13s over the real runs tree, measured: the syscalls wait on the Windows host,
    # so the CPU is idle), and neither depends on the other, so they run concurrently
    # and the cold path costs about the slower one instead of their sum.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        entries_future = pool.submit(job_store._all_checkpoints, force)
        scan_future = pool.submit(job_store._job_scan, root)
        entries = entries_future.result()
        scan = scan_future.result()
    payload = _describe_models(root, entries, scan=scan)
    job_descriptions = _scan_job_dirs(root, scan=scan)
    payload["jobs"] = job_descriptions
    payload["jobs_count"] = len(job_descriptions)
    payload["jobs_total_mb"] = _mb(sum(item["size_mb"] for item in job_descriptions) * 1024 * 1024)
    payload["tmp_mb"] = _mb(
        sum(
            item["size_mb"]
            for item in job_descriptions
            if item["tmp"] and not item["protected"]
        )
        * 1024
        * 1024
    )
    payload["tmp_jobs_count"] = sum(
        1 for item in job_descriptions if item["tmp"] and not item["protected"]
    )
    for item in payload["items"]:
        item.pop("_absolute", None)
        item.pop("_normalised", None)
        item.pop("_host_id", None)
        item.pop("_trailing", None)

    _models_cache = payload
    _models_cache_at = time.monotonic()
    _models_cache_key = key
    return copy.deepcopy(payload)


def _clear_models_cache() -> None:
    """Drop both the job-level cache and the shared checkpoint scan cache."""

    global _models_cache, _models_cache_at, _models_cache_key
    _models_cache = None
    _models_cache_at = 0.0
    _models_cache_key = ()
    job_store._clear_checkpoint_cache()
    _clear_host_jobs_cache()


def _checkpoint_protection(root: Path, relative: str) -> str:
    """Why the weight at ``relative`` may not be deleted; ``""`` when it may.

    Single-file variant of :func:`_describe_models`'s protection rules. A delete
    must not pay for a full ``GET /api/models`` (which walks every job directory
    for its size): the newest-weight check only needs the cached checkpoint scan.
    """

    if _is_bc_warmstart(relative):
        return "BC 预热权重（PPO warmstart 依赖，删除会让训练退回从零）"
    job_relative = relative.rsplit("/", 1)[0] if "/" in relative else ""
    if job_relative:
        job_dir = root / job_relative
        meta = job_store._read_meta(job_dir)
        name = job_relative.rsplit("/", 1)[-1]
        if str(meta.get("status") or "") == "running" or name in _live_job_dirs():
            return "作业正在运行（status=running）"
        if _dir_is_active(job_dir):
            # A running PPO trainer keeps overwriting checkpoint.pt and
            # checkpoint_latest.pt on every update: deleting one mid-run both loses
            # the weights and can make its next save fail.
            host_id = _host_running_jobs().get(
                _repo_relative_key(_repo_prefix(root), job_relative), ""
            )
            if host_id:
                return f"作业正在运行（host 真机任务 {host_id}）"
            if _activity_in_dir(job_dir):
                return "作业正在运行（训练日志仍在写入）"
            return "作业正在运行"
    absolute = root / relative
    referenced = _referenced_checkpoints(_job_index(root))
    key = _normalise_reference(absolute)
    if key in referenced:
        return f"被作业 {referenced[key]} 引用为初始权重"
    newest = _newest_checkpoint_job(root)
    if newest and job_relative.lower() == newest.lower():
        return "最新权重（下拉框默认首选）"
    return ""


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@router.get("/api/models")
def list_models(force: bool = False) -> dict[str, Any]:
    """List manageable checkpoints + their job dirs, with protection status.

    ``force=true`` bypasses the scan cache (the panel's "刷新" button), matching
    ``GET /api/checkpoints?force=true``.
    """

    return _models_payload(force=force)


@router.delete("/api/models/job/{path:path}")
def delete_model_job(
    path: str,
    selected: str = Query(default="", description="当前下拉框选中的 checkpoint 相对路径"),
) -> dict[str, Any]:
    """Delete a whole job directory (weights + replays + logs).

    Only ``<runs root>/<category>/<job>`` is accepted; the runs root, any category
    directory and any protected job are refused.
    """

    root = _models_root()
    target, relative = _resolve_managed(path)
    parts = [part for part in relative.split("/") if part]
    # Legal shapes are "<job>" (root is already the runs root itself) and
    # "<category>/<job>" (root is the parent of the categories). Three or more
    # levels is a sub-directory *inside* a job — metrics/replays/logs live there.
    if len(parts) > 2:
        raise HTTPException(
            status_code=400,
            detail=f"只允许删除 <runs根>/<大类>/<作业> 这一层目录，拒绝: {relative}",
        )
    if len(parts) == 1:
        # A bare top-level name must not be a category directory (platform,
        # archive, experiments, recordings, …: those hold many jobs.
        if parts[0].lower() in CATEGORY_DIR_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"拒绝删除大类目录 {relative}；只能删大类下的单个作业目录",
            )
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"作业目录不存在: {relative}")
        if _looks_like_category(target):
            raise HTTPException(
                status_code=400,
                detail=f"拒绝删除包含多个作业的目录 {relative}",
            )
    else:
        category_dir = root / parts[0]
        if category_dir.resolve() != target.parent.resolve():
            # Defence in depth: the resolved target must sit directly inside a
            # top-level directory of the runs root.
            raise HTTPException(status_code=400, detail=f"路径层级异常: {relative}")
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"作业目录不存在: {relative}")
        if _looks_like_category(target):
            # "<category>/<job>" where the job turned out to be a container of
            # jobs (a nested runs tree) — refuse instead of deleting many jobs.
            raise HTTPException(
                status_code=400,
                detail=f"拒绝删除嵌套的大类目录 {relative}",
            )
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {relative}")
    selected_relative = _selected_relative(root, selected)
    if selected_relative and selected_relative.startswith(relative + "/"):
        raise HTTPException(
            status_code=409,
            detail=f"作业 {relative} 里的 {selected_relative} 是下拉框当前选中的模型，不能删除",
        )

    # Only this one directory's protection matters, and none of the rules needs the
    # per-job recursive size walk, so a delete never pays for a full job scan.
    reason = _job_protection(root, relative)
    if reason:
        raise HTTPException(status_code=409, detail=f"{relative} 受保护：{reason}")

    freed_mb = _mb(job_store._dir_size_bytes(target))
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除 {relative} 失败: {exc}") from exc
    _clear_models_cache()
    return {"ok": True, "deleted": relative, "freed_mb": freed_mb}


@router.delete("/api/models/{path:path}")
def delete_model(
    path: str,
    selected: str = Query(default="", description="当前下拉框选中的 checkpoint 相对路径"),
) -> dict[str, Any]:
    """Delete a single checkpoint weight file (whitelisted filenames only)."""

    root = _models_root()
    target, relative = _resolve_managed(path)
    canonical = _is_checkpoint_filename(target.name)
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "只允许删除权重文件 "
                + " / ".join(job_store.CHECKPOINT_FILENAMES)
                + f"，拒绝: {relative}"
            ),
        )
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"checkpoint 不存在: {relative}")

    selected_relative = _selected_relative(root, selected)
    if selected_relative and selected_relative.lower() == relative.lower():
        raise HTTPException(
            status_code=409,
            detail=f"{relative} 是下拉框当前选中的模型，不能删除",
        )

    reason = _checkpoint_protection(root, relative)
    if reason:
        raise HTTPException(status_code=409, detail=f"{relative} 受保护：{reason}")

    try:
        size_mb = _mb(target.stat().st_size)
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除 {relative} 失败: {exc}") from exc
    _clear_models_cache()
    return {"ok": True, "deleted": relative, "freed_mb": size_mb}


def _selected_relative(root: Path, selected: str) -> str:
    """Normalise the frontend's "currently selected" checkpoint path.

    The dropdown values are whatever ``GET /api/checkpoints`` produced, i.e. an
    *absolute host path*, while this API speaks root-relative posix paths. Accept
    both (plus a repo-relative path) and return the root-relative form, or the
    raw text when it cannot be mapped — the caller only uses it for a
    ``startswith``/equality check, so an unmapped value simply never matches.
    """

    text = str(selected or "").replace("\\", "/").strip().strip("/")
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        mapped = _managed_relative(root, candidate)
        if mapped:
            return mapped
    root_relative = _managed_relative(paths.REPO_ROOT, root) or ""
    if root_relative and text.lower().startswith(root_relative.lower() + "/"):
        return text[len(root_relative) + 1 :]
    return text
