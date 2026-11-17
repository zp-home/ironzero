# SPDX-License-Identifier: Apache-2.0
"""One-off: delete the two weight classes the user picked.

Kept out of ``scripts/cleanup_runs.py`` on purpose — that script encodes the
*default* policy (cited weights are evidence, experiments are kept). This is the
explicit override the user chose:

1. platform weights whose job id is cited in ``README.md``/``audit``/``docs``
   (91 files, ~75 MB) — the numbers in those reports stop being re-runnable;
2. weights under ``runs/experiments`` except ``bc_ironzero*`` (27 files, ~21 MB)
   — ``bc_ironzero*`` is the PPO warm start and is never touched.

Left alone: BC warmstart weights, and every job that wrote in the last 24 hours
(those belong to the experiment currently running).

Usage::

    python scripts/cleanup_selected_weights.py            # 演练
    python scripts/cleanup_selected_weights.py --apply
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cleanup_runs as base  # noqa: E402


def _victims(repo_root: Path | None = None) -> list[tuple[str, Path, int]]:
    root = (repo_root or base.REPO_ROOT).resolve()
    runs = root / "runs"
    cutoff = time.time() - 24 * 3600.0
    cited = base._cited_weight_paths(root)
    cited_ids = base._referenced_job_ids(root)

    picked: list[tuple[str, Path, int]] = []
    for path in sorted(runs.rglob("*.pt")):
        if path.name not in base.WEIGHT_NAMES or not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        lowered = rel.lower()
        # Never: BC warm start, or anything the running experiment touched today.
        if "bc_ironzero" in lowered:
            continue
        if base._job_is_recent(path.parent, cutoff):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue

        in_experiments = lowered.startswith("runs/experiments/")
        match = base._JOB_ID_RE.search(rel)
        job_cited = bool(match and match.group(1).lower() in cited_ids)
        file_cited = os.path.normcase(rel) in cited

        if in_experiments:
            picked.append(("experiments", path, size))
        elif job_cited or file_cited:
            picked.append(("cited-platform", path, size))
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="真正删除（默认只列出）")
    parser.add_argument("--manifest", default="audit/cleanup_selected_weights.csv")
    args = parser.parse_args(argv)

    rows = _victims()
    total = sum(size for _c, _p, size in rows)
    by_class: dict[str, list[tuple[str, Path, int]]] = {}
    for row in rows:
        by_class.setdefault(row[0], []).append(row)

    print("=" * 66)
    print(f"按用户指定范围清理权重  ({'执行' if args.apply else '演练，不删除'})")
    print("=" * 66)
    for name, group in by_class.items():
        print(f"  {name:<16} {len(group):>4} 项  {base._mb(sum(r[2] for r in group)):>8.2f} MB")
    print("-" * 66)
    print(f"  {'合计':<16} {len(rows):>4} 项  {base._mb(total):>8.2f} MB")
    print(f"\n  不删：BC 预热权重、近 24 小时内被写入的作业（正在跑的实验）")

    manifest = base.REPO_ROOT / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "relative_path", "size_bytes"])
        for klass, path, size in rows:
            writer.writerow([klass, path.relative_to(base.REPO_ROOT).as_posix(), size])
    print(f"\n清单: {manifest.relative_to(base.REPO_ROOT).as_posix()}")

    if not args.apply:
        print("\n演练结束，未删除。加 --apply 执行。")
        return 0

    deleted = failed = freed = 0
    for _klass, path, size in rows:
        try:
            path.unlink()
            deleted += 1
            freed += size
        except FileNotFoundError:
            continue
        except OSError as exc:
            failed += 1
            print(f"  ! {path} -> {exc}", file=sys.stderr)
    print(f"\n完成：删除 {deleted} 个权重，失败 {failed}，释放 {base._mb(freed)} MB")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
