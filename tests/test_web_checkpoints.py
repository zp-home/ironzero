# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_find_checkpoints`` / ``GET /api/checkpoints`` (scan + cache).

Why this exists: the endpoint returned ``200`` but took 18-27s inside the
container, because ``_find_checkpoints`` ran three ``rglob`` passes (one per
checkpoint filename) over a Docker Desktop Windows bind mount with 690
directories. The dashboard aborts at 8s, so the model list was always empty.

The contract this file pins down:

* one directory walk per root discovers all three checkpoint filenames;
* the returned entries keep their exact shape/order/``limit`` semantics;
* a repeated call inside the TTL does no filesystem work at all, ``force=True``
  always rescans, and a changed root set is never served from a stale cache;
* a truncated/corrupt sidecar ``summary.json`` degrades to "no eval" instead of
  failing the request.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from web.backend.app import jobs

EXPECTED_KEYS = {"name", "map", "path", "update", "eval", "mtime", "size_mb"}


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class CheckpointScanTests(unittest.TestCase):
    """Each test builds a throwaway ``runs`` tree and points jobs at it."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-ckpt-"))
        self.root = self._tmp / "runs"
        self.root.mkdir()
        self._real_roots = jobs.CHECKPOINT_ROOTS
        self._real_walk = jobs._walk_checkpoint_files
        self._walk_calls: list[Path] = []

        def counting_walk(root: Path) -> list[Path]:
            self._walk_calls.append(root)
            return self._real_walk(root)

        jobs.CHECKPOINT_ROOTS = [self.root]
        jobs._walk_checkpoint_files = counting_walk  # type: ignore[assignment]
        jobs._clear_checkpoint_cache()

    def tearDown(self) -> None:
        jobs.CHECKPOINT_ROOTS = self._real_roots
        jobs._walk_checkpoint_files = self._real_walk  # type: ignore[assignment]
        jobs._clear_checkpoint_cache()
        shutil.rmtree(self._tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    def _make_checkpoint(self, relative_dir: str, filename: str, *, age_seconds: float = 0.0,
                         summary: dict | None = None, summary_text: str | None = None) -> Path:
        directory = self.root / relative_dir
        path = _write(directory / filename, "weights")
        if summary is not None:
            _write(directory / "summary.json", json.dumps(summary, ensure_ascii=False))
        if summary_text is not None:
            _write(directory / "summary.json", summary_text)
        if age_seconds:
            stamp = time.time() - age_seconds
            os.utime(path, (stamp, stamp))
        return path.resolve()

    # -- discovery ---------------------------------------------------------
    def test_one_walk_per_root_finds_all_three_filenames(self) -> None:
        self._make_checkpoint("job-a/ironzero25", "checkpoint_best.pt")
        self._make_checkpoint("job-b/ironzero10", "checkpoint_latest.pt")
        self._make_checkpoint("job-c/ironzero5/nested/deeper", "checkpoint.pt")

        found = jobs._find_checkpoints(force=True)

        self.assertEqual(len(found), 3)
        self.assertEqual({Path(item["path"]).name for item in found},
                         {"checkpoint_best.pt", "checkpoint_latest.pt", "checkpoint.pt"})
        # The whole point of the rewrite: exactly one walk, not three.
        self.assertEqual(len(self._walk_calls), 1)

    def test_entry_shape_and_map_resolution(self) -> None:
        self._make_checkpoint(
            "job-summary/ironzero25", "checkpoint_best.pt",
            summary={"map": "IronZero25", "best_eval": {"win_rate": 0.75, "update": 42}},
        )
        self._make_checkpoint("job-fallback/ironzero10", "checkpoint_latest.pt")

        found = {Path(item["path"]).name: item for item in jobs._find_checkpoints(force=True)}

        entry = found["checkpoint_best.pt"]
        self.assertEqual(set(entry), EXPECTED_KEYS)
        self.assertEqual(entry["name"], "ironzero25/checkpoint_best.pt")
        self.assertEqual(entry["map"], "IronZero25")
        self.assertEqual(entry["update"], 42)
        self.assertEqual(entry["eval"], {"win_rate": 0.75, "update": 42})
        self.assertEqual(entry["size_mb"], round(len("weights") / 1024 / 1024, 2))
        self.assertRegex(entry["mtime"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

        # No sidecar at all: the directory name still names the map.
        self.assertEqual(found["checkpoint_latest.pt"]["map"], "IronZero10")
        self.assertIsNone(found["checkpoint_latest.pt"]["eval"])
        self.assertIsNone(found["checkpoint_latest.pt"]["update"])

    def test_sorted_by_mtime_descending(self) -> None:
        self._make_checkpoint("job-old/ironzero5", "checkpoint.pt", age_seconds=3 * 86400)
        self._make_checkpoint("job-new/ironzero5", "checkpoint_best.pt", age_seconds=0)
        self._make_checkpoint("job-mid/ironzero5", "checkpoint_latest.pt", age_seconds=86400)

        found = jobs._find_checkpoints(force=True)

        # ``name`` is "<parent dir>/<file>", so key on the job directory instead.
        jobs_order = [Path(item["path"]).parent.parent.name for item in found]
        self.assertEqual(jobs_order, ["job-new", "job-mid", "job-old"])
        mtimes = [item["mtime"] for item in found]
        self.assertEqual(mtimes, sorted(mtimes, reverse=True))

    def test_limit_truncates_the_newest_entries(self) -> None:
        for index in range(5):
            self._make_checkpoint(f"job-{index}/ironzero5", "checkpoint.pt",
                                  age_seconds=(5 - index) * 3600)

        found = jobs._find_checkpoints(limit=2, force=True)

        self.assertEqual([Path(item["path"]).parent.parent.name for item in found],
                         ["job-4", "job-3"])

    def test_default_limit_is_still_sixty(self) -> None:
        for index in range(63):
            self._make_checkpoint(f"job-{index:02d}/ironzero5", "checkpoint.pt",
                                  age_seconds=(63 - index) * 60)

        self.assertEqual(len(jobs._find_checkpoints()), 60)
        self.assertEqual(len(jobs._find_checkpoints(force=True)), 60)
        self.assertEqual(len(jobs._find_checkpoints(limit=100, force=True)), 63)

    def test_nested_roots_are_not_scanned_twice(self) -> None:
        self._make_checkpoint("job-a/ironzero5", "checkpoint.pt")
        jobs.CHECKPOINT_ROOTS = [self.root, self.root]
        jobs._clear_checkpoint_cache()
        self.assertEqual(len(jobs._find_checkpoints(force=True)), 1)

        jobs.CHECKPOINT_ROOTS = [self.root, self.root / "job-a"]
        jobs._clear_checkpoint_cache()
        walks = len(self._walk_calls)
        self.assertEqual(len(jobs._find_checkpoints(force=True)), 1)
        self.assertEqual(len(self._walk_calls) - walks, 1)

    def test_filename_match_matches_rglob_case_rules(self) -> None:
        """``rglob`` is case-insensitive on Windows; the single walk must agree."""

        self._make_checkpoint("job-a/ironzero5", "Checkpoint_Best.PT")
        self._make_checkpoint("job-b/ironzero5", "checkpoint.pt")

        expected = 2 if os.name == "nt" else 1
        self.assertEqual(len(jobs._find_checkpoints(force=True)), expected)

    # -- cache -------------------------------------------------------------
    def test_second_call_within_ttl_does_not_walk(self) -> None:
        self._make_checkpoint("job-a/ironzero5", "checkpoint.pt")

        first = jobs._find_checkpoints(force=True)
        second = jobs._find_checkpoints()

        self.assertEqual(first, second)
        self.assertEqual(len(self._walk_calls), 1)

        # A checkpoint written now is invisible until TTL expiry or force=True.
        self._make_checkpoint("job-b/ironzero5", "checkpoint_best.pt")
        self.assertEqual(len(jobs._find_checkpoints()), 1)
        self.assertEqual(len(self._walk_calls), 1)

        forced = jobs._find_checkpoints(force=True)
        self.assertEqual(len(forced), 2)
        self.assertEqual(len(self._walk_calls), 2)

    def test_cache_expires_after_ttl(self) -> None:
        self._make_checkpoint("job-a/ironzero5", "checkpoint.pt")
        jobs._find_checkpoints(force=True)
        self.assertEqual(len(self._walk_calls), 1)

        jobs._checkpoints_cache_at -= jobs.CHECKPOINT_TTL_SECONDS + 1.0
        jobs._find_checkpoints()
        self.assertEqual(len(self._walk_calls), 2)

    def test_cache_is_not_served_after_roots_change(self) -> None:
        self._make_checkpoint("job-a/ironzero5", "checkpoint.pt")
        other = self._tmp / "other-runs"
        other.mkdir()
        _write(other / "job-z" / "ironzero10" / "checkpoint_best.pt", "weights")

        self.assertEqual(len(jobs._find_checkpoints(force=True)), 1)
        jobs.CHECKPOINT_ROOTS = [other]
        self.assertEqual([Path(item["path"]).parent.parent.name for item in jobs._find_checkpoints()],
                         ["job-z"])

    def test_returned_entries_do_not_alias_the_cache(self) -> None:
        self._make_checkpoint("job-a/ironzero5", "checkpoint.pt",
                              summary={"best_eval": {"update": 7}})

        jobs._find_checkpoints(force=True)
        returned = jobs._find_checkpoints()
        returned[0]["name"] = "mutated"
        returned.clear()

        fresh = jobs._find_checkpoints()
        self.assertEqual(fresh[0]["name"], "ironzero5/checkpoint.pt")
        self.assertEqual(fresh[0]["eval"], {"update": 7})

    # -- robustness --------------------------------------------------------
    def test_corrupt_summary_json_does_not_raise(self) -> None:
        self._make_checkpoint("job-broken/ironzero25", "checkpoint.pt",
                              summary_text='{"best_eval": {"win_rate": 0.5,')
        self._make_checkpoint("job-good/ironzero10", "checkpoint_latest.pt",
                              summary={"best_eval": {"win_rate": 0.25, "update": 3}})

        found = jobs._find_checkpoints(force=True)

        self.assertEqual(len(found), 2)
        broken = next(item for item in found if "job-broken" in item["path"])
        self.assertIsNone(broken["eval"])
        self.assertEqual(broken["map"], "IronZero25")  # falls back to the dir name
        good = next(item for item in found if "job-good" in item["path"])
        self.assertEqual(good["eval"]["win_rate"], 0.25)

    def test_missing_roots_yield_empty_list(self) -> None:
        jobs.CHECKPOINT_ROOTS = [self._tmp / "does-not-exist"]
        jobs._clear_checkpoint_cache()
        self.assertEqual(jobs._find_checkpoints(force=True), [])

    def test_unreadable_stat_is_skipped(self) -> None:
        checkpoint = self._make_checkpoint("job-a/ironzero5", "checkpoint.pt")
        real_stat = Path.stat

        def flaky_stat(self, *args, **kwargs):  # noqa: ANN001, ANN202
            if self.name == checkpoint.name:
                raise OSError("gone")
            return real_stat(self, *args, **kwargs)

        Path.stat = flaky_stat  # type: ignore[assignment]
        try:
            self.assertEqual(jobs._find_checkpoints(force=True), [])
        finally:
            Path.stat = real_stat  # type: ignore[assignment]


class CheckpointRouteTests(unittest.TestCase):
    """The route must pass ``force`` through so "refresh weights" really rescans."""

    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except ImportError:  # pragma: no cover - fastapi is a test dependency
            self.skipTest("fastapi is not installed")
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-ckpt-api-"))
        self.root = self._tmp / "runs"
        _write(self.root / "job-a" / "ironzero25" / "checkpoint_best.pt", "weights")
        self._real_roots = jobs.CHECKPOINT_ROOTS
        jobs.CHECKPOINT_ROOTS = [self.root]
        jobs._clear_checkpoint_cache()

    def tearDown(self) -> None:
        jobs.CHECKPOINT_ROOTS = self._real_roots
        jobs._clear_checkpoint_cache()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_route_defaults_and_force_query(self) -> None:
        from fastapi.testclient import TestClient

        from web.backend.app import create_app

        with TestClient(create_app()) as client:
            response = client.get("/api/checkpoints")
            self.assertEqual(response.status_code, 200)
            self.assertEqual([item["name"] for item in response.json()],
                             ["ironzero25/checkpoint_best.pt"])

            # Warm cache: a checkpoint written now needs force=true to show up.
            _write(self.root / "job-b" / "ironzero10" / "checkpoint_latest.pt", "weights")
            cached = client.get("/api/checkpoints")
            self.assertEqual(len(cached.json()), 1)

            forced = client.get("/api/checkpoints", params={"force": "true"})
            self.assertEqual(forced.status_code, 200)
            self.assertEqual(sorted(item["name"] for item in forced.json()),
                             ["ironzero10/checkpoint_latest.pt",
                              "ironzero25/checkpoint_best.pt"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
