# SPDX-License-Identifier: Apache-2.0
"""Reclaim space under ``runs/``: JVM logs, caches, redundant and stale weights.

``runs/`` is a runtime artefact area (``docs/产物目录规范.md``) and is never
committed, but it grows without bound: every game spawn leaves a
``game-*.log``, every PPO job writes three ``*.pt`` files of which
``checkpoint_latest.pt`` duplicates ``checkpoint.pt``, and a superseded
experiment keeps its weights forever.

This script is the reproducible version of the ad-hoc cleanups that used to be
run by hand. It is **dry-run by default**: pass ``--apply`` to delete.

Protection rules (nothing below is ever deleted):

* anything modified in the last ``--grace-minutes`` (default 15) — a host
  trainer may be writing right now;
* job dirs whose ``meta.json`` says ``status: running``;
* ``bc_ironzero*`` weights — ``scripts/rwpp_service/ppo.py`` warm-starts PPO
  from them;
* everything under ``runs/experiments/`` — the kept experiments;
* the ``--keep-newest`` newest job dirs (default 3) — they hold the dropdown
  default in the web UI;
* every job id mentioned by name in ``README.md`` / ``audit/*.md`` /
  ``docs/*.md`` — those are cited as evidence in the project's reports;
* ``runs/_scratch/probe_obs_type_blind.py`` — referenced by
  ``src/ironzero/micro_config.py`` and ``tests/test_observation_identity.py``.

Categories
----------
``logs``      ``runs/**/game-*.log`` + top-level ``runs/*.log`` (JVM stderr of
              one spawn; regenerated on every run, never read back)
``caches``    ``__pycache__/`` and ``.pytest_cache/``
``junk-dirs`` ``runs/archive``, ``runs/_speedprobe``, ``runs/_logs`` and every
              file in ``runs/_scratch`` except the protected probe
``stale-runs`` top-level one-off run dirs (``p1_*``, ``econ_fix_*``): weights
              and ``live.jsonl`` go, the small ``*report.json`` / ``summary``
              sidecars stay so the audit citations keep resolving
``dup-latest````checkpoint_latest.pt`` when it is not newer than the sibling
              ``checkpoint.pt`` (same ``update`` and same weights, or an older
              ``update``)
``stale-weights`` remaining ``*.pt`` under ``runs/platform`` whose job is not
              protected

Usage::

    python scripts/cleanup_runs.py                  # 只看，不删
    python scripts/cleanup_runs.py --apply           # 真删
    python scripts/cleanup_runs.py --apply --grace-minutes 0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"

#: Top-level files under ``runs/`` that are one spawn's JVM output.
_LOG_GLOBS = ("game-*.log",)

#: One-off top-level run dirs that predate the ``docs/产物目录规范.md`` layout.
_STALE_RUN_RE = re.compile(r"^(?:p1_|econ_fix_)\w+$", re.IGNORECASE)

#: Small sidecars kept when a stale run dir is gutted.
_KEEP_IN_STALE_RUN = frozenset(
    {"eval_report.json", "eval_summary.json", "rwpp_summary.json", "summary.json", "metrics.json"}
)

#: Delete outright — but note ``_logs`` is handled with an exception below: it
#: holds the *live* host service's stdout, which must survive.
_JUNK_DIRS = ("archive", "_speedprobe", "_logs")

#: Inside ``runs/_logs``, this is the running host service's own stdout stream.
_LOG_DIR_KEEP = frozenset({"rwpp_service.stdout.log"})

#: ``runs/_scratch`` is throwaway except this reference implementation.
_SCRATCH_KEEP = frozenset({"probe_obs_type_blind.py"})

#: ``rwpp-ppo-ppo-<8 hex>`` — how the platform names a PPO job dir.
_JOB_ID_RE = re.compile(r"rwpp-ppo-ppo-([0-9a-f]{8})", re.IGNORECASE)

#: A backticked or bare path ending in one of :data:`WEIGHT_NAMES`, e.g.
#: ``runs/platform/x/checkpoint_best.pt`` or ``econ_fix_long/checkpoint_best.pt``.
_CITED_WEIGHT_RE = re.compile(
    r"[A-Za-z0-9_.\-/\\]*[\\/](?:checkpoint(?:_best|_latest)?\.pt)\b",
    re.IGNORECASE,
)

WEIGHT_NAMES = frozenset({"checkpoint.pt", "checkpoint_best.pt", "checkpoint_latest.pt"})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mb(size: float) -> float:
    return round(float(size) / 1024 / 1024, 2)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _referenced_job_ids(root: Path | None = None) -> set[str]:
    """PPO job ids cited by name in the project's prose (lowercase hex)."""

    base = root or REPO_ROOT
    ids: set[str] = set()
    for path in _prose_sources(base):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids.update(match.lower() for match in _JOB_ID_RE.findall(text))
    return ids


def _prose_sources(base: Path) -> list[Path]:
    """``README.md`` plus every markdown file under ``audit/`` and ``docs/``."""

    sources = [base / "README.md"]
    for sub in ("audit", "docs"):
        directory = base / sub
        if directory.is_dir():
            sources.extend(sorted(directory.glob("*.md")))
    return sources


def _cited_weight_paths(root: Path | None = None) -> set[str]:
    """Repo-relative weight paths spelled out in the project's prose.

    Two citation styles matter, and both are literal evidence: a full path
    (``runs/platform/rwpp-ppo-ppo-2588dc84/checkpoint_best.pt``) and a job-relative
    one (``econ_fix_long/checkpoint_best.pt``). The latter is how
    ``audit/闭环复核_20261114_B_训练评估.md`` pins the checkpoint it verified
    ``load_policy`` against for a missing ``schema_version``; deleting it would
    silently invalidate that claim.
    """

    found: set[str] = set()
    for path in _prose_sources(root or REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _CITED_WEIGHT_RE.finditer(text):
            literal = match.group(0).replace("\\", "/").strip("/")
            found.add(os.path.normcase(literal))
            # A citation without a leading ``runs/`` still names a real file
            # under ``runs/``; record both spellings so either lookup hits.
            if not literal.lower().startswith("runs/"):
                found.add(os.path.normcase(f"runs/{literal}"))
    return found


def _job_status(job_dir: Path) -> str:
    """``running`` when the job's meta or a freshly written activity file says so."""

    meta = job_dir / "meta.json"
    if meta.is_file():
        try:
            import json

            data = json.loads(meta.read_text(encoding="utf-8"))
            if str(data.get("status") or "").lower() == "running":
                return "running(meta)"
        except (OSError, ValueError):
            pass
    return ""


def _recent(path: Path, cutoff: float) -> bool:
    try:
        return path.stat().st_mtime > cutoff
    except OSError:
        return False


def _load_weights(path: Path) -> dict:
    """``torch.load`` a checkpoint into a plain payload dict.

    Kept as a separate function so tests can substitute a stub: the decision logic
    in :func:`_weights_equal` is what needs pinning, not torch's deserialiser.
    """

    import torch

    return torch.load(str(path), map_location="cpu", weights_only=False)


def _tensors_equal(left, right) -> bool:
    """Compare two state-dict tensors (or any objects with ``torch.equal``)."""

    import torch

    return bool(torch.equal(left, right))


def _weights_equal(main: Path, other: Path) -> tuple[bool, str]:
    """``(delete_ok, why)`` for a ``checkpoint_latest.pt`` next to ``checkpoint.pt``.

    Deletable when the two hold the same update with identical weights (the file
    only adds a ``metrics`` key), or when the ``_latest`` copy is *older* than
    the final one — in both cases nothing is lost by removing it.
    """

    try:
        left = _load_weights(main)
        right = _load_weights(other)
    except ImportError:  # pragma: no cover - torch is a hard project dependency
        return False, "torch 不可用"
    except Exception as exc:  # noqa: BLE001 - a corrupt weight is worth reporting
        return False, f"读取失败: {exc.__class__.__name__}"
    left_update = left.get("update")
    right_update = right.get("update")
    if right_update != left_update:
        if isinstance(left_update, int) and isinstance(right_update, int) and right_update < left_update:
            return True, f"latest 落后 (upd {right_update} < {left_update})"
        return False, f"update 不同 ({right_update} vs {left_update})"
    left_model = left.get("model") or {}
    right_model = right.get("model") or {}
    if set(left_model) != set(right_model):
        return False, "参数集合不同"
    for key, tensor in left_model.items():
        try:
            if not _tensors_equal(tensor, right_model[key]):
                return False, f"权重不同 ({key})"
        except Exception:  # noqa: BLE001
            return False, "权重不可比较"
    return True, f"与 checkpoint.pt 同权重 (upd {left_update})"


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
class Plan:
    """Collects ``(category, path, bytes, note)`` rows before anything is deleted."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, Path, int, str]] = []
        self._seen: set[str] = set()

    def add(self, category: str, path: Path, size: int | None = None, note: str = "") -> None:
        key = os.path.normcase(str(path))
        if key in self._seen:
            return
        # A parent directory already queued for deletion swallows this entry:
        # listing both would make the manifest over-count the freed bytes and
        # hand ``--apply`` a path that no longer exists.
        for parent in path.parents:
            if os.path.normcase(str(parent)) in self._seen:
                return
        self._seen.add(key)
        if size is None:
            try:
                size = path.stat().st_size if path.is_file() else _dir_size(path)
            except OSError:
                size = 0
        self.rows.append((category, path, size, note))

    def summary(self) -> list[tuple[str, int, float]]:
        buckets: dict[str, list[int]] = {}
        for category, _path, size, _note in self.rows:
            bucket = buckets.setdefault(category, [0, 0])
            bucket[0] += 1
            bucket[1] += size
        return [(name, count, _mb(total)) for name, (count, total) in sorted(buckets.items())]


def _logs_are_being_written(cutoff: float, repo_root: Path | None = None) -> str:
    """Repo-relative name of a live-service artifact touched after ``cutoff``.

    ``runs/_logs/rwpp_service.stdout.log`` is appended to while the host service
    runs, and every real-game job writes ``runs/_rwpp_service/jobs/*.json``. If
    either moved inside the grace window, an experiment is in flight, so weight
    deletion is withheld (see :data:`WEIGHT_CATEGORIES`) rather than risking a run
    in progress.
    """

    base = (repo_root or REPO_ROOT).resolve()
    runs = base / "runs"
    candidates = [
        runs / "_logs" / "rwpp_service.stdout.log",
        runs / "_rwpp_service" / "jobs",
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            newest = path.stat().st_mtime
            if path.is_dir():
                for child in path.iterdir():
                    newest = max(newest, child.stat().st_mtime)
            if newest > cutoff:
                return str(path.relative_to(base).as_posix())
        except OSError:
            continue
    return ""


def _newest_in_dir(directory: Path) -> float:
    """Newest mtime of any entry directly inside ``directory`` (0.0 when none).

    Appendix-only writes (``live.jsonl``, ``train.log``) do not bump the
    directory's own mtime, so the directory stat alone would call a live training
    job "old". One ``scandir`` per job is cheap next to the risk of deleting a
    weight out from under a running trainer.
    """

    newest = 0.0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
    except OSError:
        return 0.0
    return newest


def _job_is_recent(directory: Path, cutoff: float) -> bool:
    """True when the job dir itself or anything in it moved after ``cutoff``."""

    try:
        if directory.stat().st_mtime > cutoff:
            return True
    except OSError:
        return True  # unreadable: assume live
    return _newest_in_dir(directory) > cutoff


#: Categories that destroy model weights. While the host service is actively
#: writing ``runs/`` these are withheld even if every per-job gate passes: the
#: cost of waiting is nil, the cost of deleting the weight a trainer is about to
#: reload is not.
WEIGHT_CATEGORIES = frozenset({"dup-latest", "stale-runs", "stale-weights"})


def build_plan(grace_minutes: float, keep_newest: int, min_age_hours: float = 24.0) -> Plan:
    cutoff = time.time() - grace_minutes * 60.0
    #: Weights are only "stale" once the whole job has been idle this long; a job
    #: from this session is someone's current result, not litter.
    stale_cutoff = time.time() - min_age_hours * 3600.0
def build_plan(
    grace_minutes: float,
    keep_newest: int,
    min_age_hours: float = 24.0,
    repo_root: Path | None = None,
) -> Plan:
    """Classify everything under ``<repo_root>/runs`` into deletable buckets.

    ``repo_root`` exists so the protection rules can be tested against a throwaway
    tree instead of the real repository (deleting real evidence to prove a rule
    works would be self-defeating).
    """

    base = (repo_root or REPO_ROOT).resolve()
    runs = base / "runs"
    cutoff = time.time() - grace_minutes * 60.0
    #: Weights are only "stale" once the whole job has been idle this long; a job
    #: from this session is someone's current result, not litter.
    stale_cutoff = time.time() - min_age_hours * 3600.0
    plan = Plan()
    if not runs.is_dir():
        return plan

    # ---- protecting sets -------------------------------------------------- #
    referenced = _referenced_job_ids(base)
    cited = _cited_weight_paths(base)
    newest_jobs = {
        str(path.parent)
        for path in sorted(
            (p for p in runs.rglob("*.pt") if p.name in WEIGHT_NAMES),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )[: keep_newest * 3]
    }

    def job_protected(path: Path) -> str:
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            return "不在仓库内"
        parent = path.parent
        if "runs/experiments/" in rel + "/":
            return "runs/experiments 保留"
        if "bc_ironzero" in rel.lower():
            return "BC 预热权重"
        # Checked before the job-id rule: a prose citation may name the weight
        # file directly while never spelling out the job's hex id.
        if os.path.normcase(rel) in cited:
            return "文档/审计按文件名引用"
        match = _JOB_ID_RE.search(rel)
        if match and match.group(1).lower() in referenced:
            return "文档/审计引用"
        if str(parent) in newest_jobs:
            return "最新权重"
        if _job_status(parent):
            return "作业运行中"
        if _job_is_recent(parent, cutoff):
            return f"作业在最近 {grace_minutes:g} 分钟内仍有写入"
        return ""

    # ---- 1. logs ---------------------------------------------------------- #
    for pattern in _LOG_GLOBS:
        for path in runs.rglob(pattern):
            if not path.is_file() or _recent(path, cutoff):
                continue
            plan.add("logs", path, note="JVM spawn log")
    for path in runs.glob("*.log"):
        if path.is_file() and not _recent(path, cutoff):
            plan.add("logs", path, note="runs 顶层散落日志")

    # ---- 2. caches -------------------------------------------------------- #
    for cache in base.rglob("__pycache__"):
        if ".git" in cache.parts or not cache.is_dir():
            continue
        plan.add("caches", cache, note="__pycache__")
    pytest_cache = base / ".pytest_cache"
    if pytest_cache.is_dir():
        plan.add("caches", pytest_cache, note="pytest 缓存")

    # ---- 3. junk dirs ----------------------------------------------------- #
    for name in _JUNK_DIRS:
        target = runs / name
        if not target.is_dir():
            continue
        if name == "_logs":
            # The host service appends to its own stdout here; gut the directory
            # instead of removing it so the open handle keeps working.
            for path in target.rglob("*"):
                if path.is_file() and path.name not in _LOG_DIR_KEEP and not _recent(path, cutoff):
                    plan.add("junk-dirs", path, note="一次性服务/探针日志")
            continue
        plan.add("junk-dirs", target, note="一次性冒烟/探针目录")
    scratch = runs / "_scratch"
    if scratch.is_dir():
        for path in scratch.rglob("*"):
            if path.is_file() and path.name not in _SCRATCH_KEEP and not _recent(path, cutoff):
                plan.add("junk-dirs", path, note="一次性调试脚本/截图")

    # ---- 4. stale top-level run dirs ------------------------------------- #
    for directory in runs.iterdir():
        if not directory.is_dir() or not _STALE_RUN_RE.match(directory.name):
            continue
        if _job_status(directory):
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.name in _KEEP_IN_STALE_RUN:
                continue
            if _recent(path, cutoff):
                continue
            # A weight this audit's prose cites by filename is evidence, not
            # litter: keep the file even though the rest of the run is gutted.
            try:
                rel = path.relative_to(base).as_posix()
            except ValueError:
                rel = ""
            if rel and os.path.normcase(rel) in cited:
                continue
            note = "一次性 run 的权重" if path.suffix == ".pt" else "一次性 run 的事件流"
            plan.add("stale-runs", path, note=note)

    # ---- 5. duplicated checkpoint_latest.pt ------------------------------ #
    for main in sorted(runs.rglob("checkpoint.pt")):
        if not main.is_file():
            continue
        latest = main.parent / "checkpoint_latest.pt"
        if not latest.is_file() or _recent(latest, cutoff) or _recent(main, cutoff):
            continue
        # Age gate on the whole job: a run from this session is someone's current
        # result, not litter.
        if _job_is_recent(main.parent, stale_cutoff):
            continue
        # Never touch a job holding one of the newest weights: the web UI's
        # dropdown default may *be* the ``_latest`` copy.
        if job_protected(main):
            continue
        ok, why = _weights_equal(main, latest)
        if ok:
            plan.add("dup-latest", latest, note=why)

    # ---- 6. stale weights under runs/platform ---------------------------- #
    platform = runs / "platform"
    if platform.is_dir():
        for path in platform.rglob("*.pt"):
            if path.name not in WEIGHT_NAMES or not path.is_file():
                continue
            if _recent(path, cutoff):
                continue
            # The job has to have been idle for ``--min-age-hours`` before its
            # weights count as stale. Without this gate a job that finished ten
            # minutes ago (its latest eval wrote nothing for the grace window)
            # would be gutted while the user is still looking at it.
            if _job_is_recent(path.parent, stale_cutoff):
                continue
            if job_protected(path):
                continue
            plan.add("stale-weights", path, note=f"作业 {path.parent.name} 未被引用")
    return plan


# --------------------------------------------------------------------------- #
# reporting & execution
# --------------------------------------------------------------------------- #
def _write_manifest(plan: Plan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "kind", "relative_path", "size_bytes", "note"])
        for category, target, size, note in plan.rows:
            kind = "dir" if target.is_dir() else "file"
            try:
                rel = target.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel = str(target)
            writer.writerow([category, kind, rel, size, note])


def _apply(plan: Plan) -> tuple[int, int]:
    deleted = 0
    failed = 0
    for category, target, _size, _note in plan.rows:
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failed += 1
            print(f"  ! {category}: {target} -> {exc}", file=sys.stderr)
            continue
        deleted += 1
    return deleted, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只列出）")
    parser.add_argument(
        "--grace-minutes",
        type=float,
        default=15.0,
        help="最近 N 分钟内改动过的文件一律跳过（默认 15，防抢跑正在训练的作业）",
    )
    parser.add_argument("--keep-newest", type=int, default=3, help="保留最近 N 个作业的权重（默认 3）")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=24.0,
        help="作业要闲置满 N 小时，其权重才算陈旧（默认 24，防删掉刚跑完的结果）",
    )
    parser.add_argument("--manifest", default="audit/cleanup_runs_manifest.csv", help="清单输出路径")
    parser.add_argument(
        "--force",
        action="store_true",
        help="无视“宿主服务/实验仍在写 runs/” 的忙碌检测（通常不该用）",
    )
    args = parser.parse_args(argv)

    if not RUNS.is_dir():
        print(f"找不到 {RUNS}", file=sys.stderr)
        return 2

    busy = _logs_are_being_written(time.time() - args.grace_minutes * 60.0)
    if busy and args.apply and not args.force:
        print(
            f"注意：{busy} 在最近 {args.grace_minutes:g} 分钟内仍在写入，"
            "说明宿主服务/实验正在跑 —— 本次只清理日志与缓存，权重类全部跳过。",
            file=sys.stderr,
        )

    started = time.time()
    plan = build_plan(args.grace_minutes, args.keep_newest, args.min_age_hours)
    if busy and not args.force:
        # Log/cache deletion is safe while anything runs (the JVM log of a live
        # spawn is inside the grace window anyway); weight deletion is not.
        withheld = [row for row in plan.rows if row[0] in WEIGHT_CATEGORIES]
        plan.rows = [row for row in plan.rows if row[0] not in WEIGHT_CATEGORIES]
        if withheld:
            print(
                f"已跳过 {len(withheld)} 项权重类清理（{_mb(sum(r[2] for r in withheld)):.2f} MB）："
                "宿主服务仍在写 runs/。确认安全后加 --force 强制删。\n",
                file=sys.stderr,
            )
    elif busy and args.force:
        print(f"注意：{busy} 仍在写入，但 --force 已指定，权重类照删。", file=sys.stderr)
    total_bytes = sum(size for _c, _p, size, _n in plan.rows)

    print("=" * 72)
    print(f"runs/ 清理计划  ({'执行' if args.apply else '演练，不删除'})")
    print(f"仓库: {REPO_ROOT}")
    print(f"保护窗口: 最近 {args.grace_minutes:g} 分钟 | 作业闲置 {args.min_age_hours:g} 小时才算陈旧 | 保留最新 {args.keep_newest} 个作业")
    if busy:
        print(f"注意: {busy} 在保护窗口内仍在写入（宿主服务可能在跑）")
    print("=" * 72)
    for category, count, megabytes in plan.summary():
        print(f"  {category:<14} {count:>6} 项  {megabytes:>9.2f} MB")
    print("-" * 72)
    print(f"  {'合计':<14} {len(plan.rows):>6} 项  {_mb(total_bytes):>9.2f} MB")
    print("=" * 72)

    by_category: dict[str, list[tuple[str, Path, int, str]]] = {}
    for row in plan.rows:
        by_category.setdefault(row[0], []).append(row)
    for category, rows in by_category.items():
        if category in {"logs", "caches"}:
            print(f"[{category}] {len(rows)} 项，示例 3 条：")
            for _c, path, _s, note in rows[:3]:
                print(f"    {path.relative_to(REPO_ROOT).as_posix()}  ({note})")
        else:
            print(f"[{category}] {len(rows)} 项：")
            for _c, path, size, note in rows[:25]:
                print(f"    {path.relative_to(REPO_ROOT).as_posix()}  {_mb(size):.2f} MB  {note}")
            if len(rows) > 25:
                print(f"    ... 其余 {len(rows) - 25} 项见清单")

    manifest = REPO_ROOT / args.manifest
    _write_manifest(plan, manifest)
    print(f"\n清单已写入 {manifest.relative_to(REPO_ROOT).as_posix()}")

    if not args.apply:
        print("\n演练结束：未删除任何文件。加 --apply 执行。")
        return 0

    print("\n开始删除 ...")
    deleted, failed = _apply(plan)
    print(f"完成：删除 {deleted} 项，失败 {failed} 项，用时 {time.time() - started:.1f}s")
    print(f"预计释放 {_mb(total_bytes)} MB")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
