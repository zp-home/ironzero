# SPDX-License-Identifier: Apache-2.0
"""Tests for model management: ``GET /api/models`` + the two ``DELETE`` routes.

Why this exists: checkpoint deletion is irreversible, so the endpoint surface is
as much a safety contract as a feature. This file pins down:

* the listing shape (``path``/``job_dir``/``size_mb``/``protected``/...) and the
  aggregate sizes (``total_mb``, ``tmp_mb``);
* the single-file whitelist: only ``checkpoint*.pt`` may go, everything else
  (metrics, logs, replays) is a ``400``;
* directory-traversal attempts (``..``, absolute paths, drive letters) are
  rejected and never touch a file outside the runs root;
* protected objects answer ``409``: running jobs, BC warmstart weights, the
  newest weight and the model the UI currently has selected;
* a whole-job delete refuses the runs root and every category directory;
* ``404`` for something that is not there;
* the scan cache is invalidated on delete, so a re-list never shows a deleted row.

Every test builds a throwaway ``runs`` tree in ``tempfile``; the real ``runs/``
directory is never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from web.backend.app import jobs, paths

MB = 1024 * 1024

EXPECTED_ITEM_KEYS = {
    "path",
    "job_dir",
    "job_name",
    "category",
    "name",
    "file",
    "size_mb",
    "mtime",
    "map",
    "update",
    "win_rate",
    "status",
    "in_use",
    "protected",
    "protected_reason",
    "tmp",
}


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ModelsTestCase(unittest.TestCase):
    """Shared fixture: a temp ``runs`` tree plus the routes pointed at it."""

    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except ImportError:  # pragma: no cover - fastapi is a test dependency
            self.skipTest("fastapi is not installed")

        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-models-")).resolve()
        self.root = self._tmp / "runs"
        self.root.mkdir()

        self._real_checkpoint_roots = jobs.CHECKPOINT_ROOTS
        self._real_models_root = os.environ.get("IRONZERO_MODELS_ROOT")

        # Point every scan at the fixture tree: ``IRONZERO_MODELS_ROOT`` steers the
        # router, ``jobs.CHECKPOINT_ROOTS`` steers the (shared, cached) scan.
        os.environ["IRONZERO_MODELS_ROOT"] = str(self.root)
        jobs.CHECKPOINT_ROOTS = [self.root]
        jobs._clear_checkpoint_cache()

        from web.backend.app import create_app
        from web.backend.app.routers import models as models_router

        self.models = models_router
        models_router._clear_models_cache()

        from fastapi.testclient import TestClient

        self.app = create_app()
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        jobs.CHECKPOINT_ROOTS = self._real_checkpoint_roots
        jobs._clear_checkpoint_cache()
        self.models._clear_models_cache()
        if self._real_models_root is None:
            os.environ.pop("IRONZERO_MODELS_ROOT", None)
        else:
            os.environ["IRONZERO_MODELS_ROOT"] = self._real_models_root
        shutil.rmtree(self._tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    def _make_job(
        self,
        category: str,
        job: str,
        *,
        filename: str = "checkpoint.pt",
        meta: dict[str, Any] | None = None,
        extra: dict[str, str] | None = None,
        size: int = 2 * MB,
    ) -> Path:
        """Create a finished job directory and return its checkpoint path.

        Activity files (``live.jsonl``/``train.log``/…) are back-dated: a fixture job
        is a *completed* job unless the test says otherwise, and a freshly written
        activity file legitimately means "a trainer is still writing" — the liveness
        rule under test. Tests that want a live job call :meth:`_touch_activity`.
        """

        job_dir = self.root / category / job
        job_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = job_dir / filename
        checkpoint.write_bytes(b"w" * size)
        if meta is not None:
            _write(job_dir / "meta.json", json.dumps(meta, ensure_ascii=False))
        for name, text in (extra or {}).items():
            path = _write(job_dir / name, text)
            if name in jobs._ACTIVITY_FILENAMES:
                self._age(path, 86_400)
        return checkpoint

    def _listing(self) -> dict[str, Any]:
        response = self.client.get("/api/models", params={"force": "true"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _item(self, listing: dict[str, Any], path: str) -> dict[str, Any]:
        for item in listing["items"]:
            if item["path"].lower() == path.lower():
                return item
        self.fail(f"{path} not in listing: {[i['path'] for i in listing['items']]}")

    def _age(self, path: Path, seconds_ago: float) -> None:
        stamp = 1_600_000_000.0 - seconds_ago
        os.utime(path, (stamp, stamp))

    def _patch_host_jobs(self, running: dict[str, str]) -> None:
        """Pretend the host real-game service reports these jobs as running.

        ``running`` maps the repo-relative ``out_dir`` to the host job id. The fixture
        tree lives in a temp dir, so the *models root* is treated as the repo root for
        the duration of the test: ``paths.REPO_ROOT`` is patched to the temp dir and
        the keys are given relative to it, which is exactly how the host service
        reports ``out_dir`` in production.
        """

        patcher = mock.patch.object(jobs, "_rwpp_job_dirs", return_value=dict(running))
        patcher.start()
        self.addCleanup(patcher.stop)
        repo_patcher = mock.patch.object(paths, "REPO_ROOT", self._tmp)
        repo_patcher.start()
        self.addCleanup(repo_patcher.stop)
        self.models._clear_host_jobs_cache()
        self.addCleanup(self.models._clear_host_jobs_cache)

    def _touch_activity(self, job_dir: Path, name: str = "live.jsonl") -> Path:
        """Write an activity file now, so the "still writing" signal is on."""

        path = job_dir / name
        _write(path, "{}\n")
        stamp = time.time()
        os.utime(path, (stamp, stamp))
        return path


class ModelsListingTests(ModelsTestCase):
    def test_listing_shape_and_totals(self) -> None:
        older = self._make_job("platform", "job-a", size=2 * MB,
                               meta={"status": "succeeded"})
        newest = self._make_job("archive", "job-b", filename="checkpoint_best.pt",
                                size=1 * MB)
        self._age(older, 3600)
        self._age(newest, 0)

        listing = self._listing()

        self.assertEqual(listing["count"], 2)
        self.assertEqual(listing["root"], Path(self.root).as_posix())
        self.assertEqual(set(listing["items"][0]), EXPECTED_ITEM_KEYS)
        self.assertAlmostEqual(listing["total_mb"], 3.0)
        item = self._item(listing, "platform/job-a/checkpoint.pt")
        self.assertEqual(item["job_dir"], "platform/job-a")
        self.assertEqual(item["category"], "platform")
        self.assertEqual(item["job_name"], "job-a")
        self.assertEqual(item["file"], "checkpoint.pt")
        self.assertEqual(item["name"], "job-a/checkpoint.pt")
        self.assertEqual(item["size_mb"], 2.0)
        self.assertEqual(item["status"], "succeeded")
        self.assertFalse(item["protected"])
        self.assertEqual(item["protected_reason"], "")
        self.assertRegex(item["mtime"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        # Job-level inventory: only directories that actually hold job data.
        self.assertEqual({job["job_dir"] for job in listing["jobs"]},
                         {"platform/job-a", "archive/job-b"})
        self.assertIsInstance(listing["tmp_mb"], float)

    def test_paths_are_relative_posix(self) -> None:
        self._make_job("platform", "job-a")

        item = self._listing()["items"][0]

        self.assertNotIn("\\", item["path"])
        self.assertFalse(Path(item["path"]).is_absolute())
        self.assertTrue(item["path"].startswith("platform/job-a/"))

    def test_entries_outside_the_root_are_not_deletable(self) -> None:
        """A checkpoint root outside the manageable root yields no delete entry."""

        outside = self._tmp / "elsewhere"
        _write(outside / "job-x" / "checkpoint.pt", "weights")
        self._make_job("platform", "job-a")
        jobs.CHECKPOINT_ROOTS = [self.root, outside]
        jobs._clear_checkpoint_cache()

        listing = self._listing()

        self.assertEqual([item["path"] for item in listing["items"]],
                         ["platform/job-a/checkpoint.pt"])

    def test_tmp_jobs_are_flagged_and_summed(self) -> None:
        smoke = self._make_job("platform", "rwpp-ppo-ppo-smoke-123", size=4 * MB,
                               extra={"train.log": "boom"})
        keeper = self._make_job("platform", "rwpp-ppo-ppo-5a5fc499", size=4 * MB,
                                extra={"train.log": "ok"})
        newest = self._make_job("platform", "rwpp-ppo-ppo-newest", size=4 * MB,
                                extra={"train.log": "ok"})
        for index, path in enumerate((smoke, keeper)):
            self._age(path, 7200 * (index + 1))
        self._age(newest, 0)

        listing = self._listing()

        smoke_item = self._item(listing, "platform/rwpp-ppo-ppo-smoke-123/checkpoint.pt")
        self.assertTrue(smoke_item["tmp"])
        self.assertFalse(smoke_item["protected"])
        self.assertFalse(
            self._item(listing, "platform/rwpp-ppo-ppo-5a5fc499/checkpoint.pt")["tmp"]
        )
        self.assertEqual(listing["tmp_jobs_count"], 1)
        self.assertGreater(listing["tmp_mb"], 0)

    def test_directories_without_job_data_are_not_listed_as_jobs(self) -> None:
        self._make_job("platform", "job-a")
        # An extracted asset tree, not a job.
        _write(self.root / "_pkg" / "META-INF" / "MANIFEST.MF", "x")

        listing = self._listing()

        self.assertEqual([job["job_dir"] for job in listing["jobs"]], ["platform/job-a"])

    def test_cache_is_reused_unless_forced(self) -> None:
        self._make_job("platform", "job-a")
        walk_calls: list[Path] = []
        real_walk = jobs._walk_checkpoint_files

        def counting_walk(root: Path) -> list[Path]:
            walk_calls.append(root)
            return real_walk(root)

        jobs._walk_checkpoint_files = counting_walk  # type: ignore[assignment]
        try:
            self.assertEqual(self.client.get("/api/models").status_code, 200)
            self.assertEqual(len(walk_calls), 1)

            self._make_job("platform", "job-b")
            # Cached: the new job stays invisible and nothing is walked.
            self.assertEqual(self.client.get("/api/models").json()["count"], 1)
            self.assertEqual(len(walk_calls), 1)

            self.assertEqual(self._listing()["count"], 2)
            self.assertEqual(len(walk_calls), 2)
        finally:
            jobs._walk_checkpoint_files = real_walk  # type: ignore[assignment]


class ModelsProtectionTests(ModelsTestCase):
    def test_running_job_checkpoint_is_protected(self) -> None:
        self._make_job("platform", "job-running", meta={"status": "running", "kind": "train"})
        self._make_job("platform", "job-old", meta={"status": "succeeded"})

        listing = self._listing()

        running = self._item(listing, "platform/job-running/checkpoint.pt")
        self.assertTrue(running["protected"])
        self.assertTrue(running["in_use"])
        self.assertIn("running", running["protected_reason"])
        self.assertFalse(self._item(listing, "platform/job-old/checkpoint.pt")["protected"])

    def test_bc_warmstart_checkpoints_are_protected(self) -> None:
        for units in (5, 10, 25):
            self._make_job("experiments", f"bc_ironzero{units}")
        # The legacy location must be recognised too.
        self._make_job("runs", "bc_ironzero25")
        self._make_job("platform", "job-a")

        listing = self._listing()

        protected = {item["path"] for item in listing["items"] if item["protected"]}
        self.assertIn("experiments/bc_ironzero5/checkpoint.pt", protected)
        self.assertIn("experiments/bc_ironzero10/checkpoint.pt", protected)
        self.assertIn("experiments/bc_ironzero25/checkpoint.pt", protected)
        self.assertIn("runs/bc_ironzero25/checkpoint.pt", protected)
        self.assertNotIn("platform/job-a/checkpoint.pt", protected)
        bc = self._item(listing, "experiments/bc_ironzero10/checkpoint.pt")
        self.assertIn("warmstart", bc["protected_reason"])

    def test_newest_checkpoint_is_protected(self) -> None:
        old = self._make_job("platform", "job-old")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)

        listing = self._listing()

        self.assertTrue(self._item(listing, "platform/job-new/checkpoint.pt")["protected"])
        self.assertFalse(self._item(listing, "platform/job-old/checkpoint.pt")["protected"])

    def test_reference_in_meta_config_is_protected(self) -> None:
        referenced = self._make_job("experiments", "custom_policy")
        self._make_job(
            "platform",
            "job-finetune",
            meta={"status": "succeeded", "config": {"init_checkpoint": str(referenced)}},
        )
        self._age(referenced, 7200)

        listing = self._listing()

        item = self._item(listing, "experiments/custom_policy/checkpoint.pt")
        self.assertTrue(item["protected"])
        self.assertIn("job-finetune", item["protected_reason"])

    def test_host_running_job_without_meta_is_protected(self) -> None:
        """Host-side real-game jobs write no ``meta.json``; they must still be safe.

        ``scripts/rwpp_service/ppo.py`` creates ``runs/platform/rwpp-ppo-<id>/`` with
        only ``live.jsonl``/``train.log``/checkpoints. Before this rule such a job
        read as ``status=unknown`` and a running training run was deletable.
        """

        running = self._make_job("platform", "rwpp-ppo-ppo-live")
        idle = self._make_job("platform", "rwpp-ppo-ppo-idle")
        self._age(running, 0)
        self._age(idle, 7200)
        # No meta.json anywhere in either job.
        self.assertFalse((self.root / "platform" / "rwpp-ppo-ppo-live" / "meta.json").exists())

        self._patch_host_jobs({"runs/platform/rwpp-ppo-ppo-live": "ppo-abc123"})
        listing = self._listing()

        live_item = self._item(listing, "platform/rwpp-ppo-ppo-live/checkpoint.pt")
        self.assertTrue(live_item["protected"])
        self.assertTrue(live_item["in_use"])
        self.assertIn("ppo-abc123", live_item["protected_reason"])
        self.assertFalse(self._item(listing, "platform/rwpp-ppo-ppo-idle/checkpoint.pt")["protected"])

        job = next(j for j in listing["jobs"] if j["job_dir"] == "platform/rwpp-ppo-ppo-live")
        self.assertTrue(job["protected"])
        self.assertIn("ppo-abc123", job["protected_reason"])

        # Both delete routes must refuse it.
        self.assertEqual(
            self.client.delete("/api/models/platform/rwpp-ppo-ppo-live/checkpoint.pt").status_code,
            409,
        )
        self.assertEqual(
            self.client.delete("/api/models/job/platform/rwpp-ppo-ppo-live").status_code, 409
        )
        self.assertTrue((self.root / "platform" / "rwpp-ppo-ppo-live" / "checkpoint.pt").is_file())

    def test_host_service_unreachable_fails_open(self) -> None:
        """A down host service must not make every job look protected (nor block deletes)."""

        old = self._make_job("platform", "job-a")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)
        with mock.patch.object(jobs, "_rwpp_job_dirs", return_value={}):
            self.models._clear_host_jobs_cache()
            listing = self._listing()

        self.assertFalse(self._item(listing, "platform/job-a/checkpoint.pt")["protected"])
        self.assertEqual(
            self.client.delete("/api/models/platform/job-a/checkpoint.pt").status_code, 200
        )

    def test_recently_written_activity_file_is_protected(self) -> None:
        """A trainer still writing is protected even when the host service forgot it.

        ``scripts/rwpp_service`` restarts lose their in-memory job list while the
        trainer subprocess keeps appending to ``live.jsonl``. The activity file is the
        witness that survives, and it is what stops a delete of a live run's weights.
        """

        running = self._make_job("platform", "rwpp-ppo-ppo-live")
        finished = self._make_job("platform", "rwpp-ppo-ppo-done")
        self._age(running, 0)
        self._age(finished, 7200)
        self._touch_activity(running.parent)
        self._touch_activity(finished.parent)
        self._age(finished.parent / "live.jsonl", 7200)

        # The host service knows about neither job.
        self._patch_host_jobs({})
        listing = self._listing()

        live_item = self._item(listing, "platform/rwpp-ppo-ppo-live/checkpoint.pt")
        self.assertTrue(live_item["protected"])
        self.assertTrue(live_item["in_use"])
        self.assertIn("写入", live_item["protected_reason"])
        self.assertFalse(self._item(listing, "platform/rwpp-ppo-ppo-done/checkpoint.pt")["protected"])

        job = next(j for j in listing["jobs"] if j["job_dir"] == "platform/rwpp-ppo-ppo-live")
        self.assertTrue(job["protected"])

        # Both delete routes refuse the live one and allow the finished one.
        self.assertEqual(
            self.client.delete("/api/models/platform/rwpp-ppo-ppo-live/checkpoint.pt").status_code,
            409,
        )
        self.assertEqual(
            self.client.delete("/api/models/job/platform/rwpp-ppo-ppo-live").status_code, 409
        )
        self.assertTrue((running.parent / "checkpoint.pt").is_file())


class DeleteOneCheckpointTests(ModelsTestCase):
    def test_delete_removes_only_the_weight_file(self) -> None:
        self._make_job("platform", "job-a",
                       extra={"metrics.json": "{}", "train.log": "log"})
        self._make_job("platform", "job-b")
        job_dir = self.root / "platform" / "job-a"
        self._age(job_dir / "checkpoint.pt", 7200)

        response = self.client.delete("/api/models/platform/job-a/checkpoint.pt")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["deleted"], "platform/job-a/checkpoint.pt")
        self.assertGreater(body["freed_mb"], 0)
        self.assertFalse((job_dir / "checkpoint.pt").exists())
        # Everything else in the job directory survives.
        self.assertTrue((job_dir / "metrics.json").is_file())
        self.assertTrue((job_dir / "train.log").is_file())
        self.assertTrue(job_dir.is_dir())

    def test_listing_shrinks_after_delete(self) -> None:
        paths = [self._make_job("platform", name) for name in ("job-a", "job-b", "job-c")]
        # job-a is the oldest, job-c the newest (so job-a is the deletable one).
        for index, path in enumerate(paths):
            self._age(path, 3600 * (3 - index))
        before = self._listing()["count"]

        self.assertEqual(
            self.client.delete("/api/models/platform/job-a/checkpoint.pt").status_code, 200
        )
        after = self._listing()

        self.assertEqual((before, after["count"]), (3, 2))
        self.assertNotIn("platform/job-a/checkpoint.pt",
                         [item["path"] for item in after["items"]])

    def test_non_whitelisted_filenames_are_rejected(self) -> None:
        self._make_job("platform", "job-a", extra={"metrics.json": "{}", "train.log": "x"})
        for relative in (
            "platform/job-a/metrics.json",
            "platform/job-a/train.log",
            "platform/job-a/meta.json",
            "platform/job-a",  # the directory itself is not a checkpoint file
        ):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/{relative}")
                self.assertEqual(response.status_code, 400, response.text)
        self.assertTrue((self.root / "platform" / "job-a" / "metrics.json").is_file())
        self.assertTrue((self.root / "platform" / "job-a" / "train.log").is_file())

    def test_path_traversal_is_rejected(self) -> None:
        self._make_job("platform", "job-a")
        outside = self._tmp / "outside"
        outside.mkdir()
        target = outside / "checkpoint.pt"
        target.write_bytes(b"important")
        encoded_abs = str(target).replace("\\", "%5C").replace(":", "%3A")

        # ``..`` must be percent-encoded: a raw one is collapsed by the HTTP layer
        # before it ever reaches the route, which would test nothing at all.
        for relative in (
            "..%2Foutside%2Fcheckpoint.pt",
            "platform%2F..%2F..%2Foutside%2Fcheckpoint.pt",
            "..%5Coutside%5Ccheckpoint.pt",
            "C%3A%2FWindows%2Fcheckpoint.pt",
            encoded_abs,
            "%2Fetc%2Fpasswd",
        ):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/{relative}")
                self.assertIn(response.status_code, (400, 404, 422), response.text)
        self.assertTrue(target.is_file())

    def test_missing_file_is_404(self) -> None:
        self._make_job("platform", "job-a")
        response = self.client.delete("/api/models/platform/job-ghost/checkpoint.pt")
        self.assertEqual(response.status_code, 404, response.text)

    def test_protected_checkpoint_is_409(self) -> None:
        self._make_job("experiments", "bc_ironzero25")
        self._make_job("platform", "job-a")

        response = self.client.delete("/api/models/experiments/bc_ironzero25/checkpoint.pt")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("BC", response.json()["detail"])
        self.assertTrue((self.root / "experiments" / "bc_ironzero25" / "checkpoint.pt").is_file())

    def test_running_job_checkpoint_is_409(self) -> None:
        self._make_job("platform", "job-running", meta={"status": "running"})
        self._make_job("platform", "job-old")

        response = self.client.delete("/api/models/platform/job-running/checkpoint.pt")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue((self.root / "platform" / "job-running" / "checkpoint.pt").is_file())

    def test_newest_checkpoint_is_409(self) -> None:
        old = self._make_job("platform", "job-old")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)

        response = self.client.delete("/api/models/platform/job-new/checkpoint.pt")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(newest.is_file())

    def test_selected_checkpoint_is_409(self) -> None:
        old = self._make_job("platform", "job-old")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)

        response = self.client.delete(
            "/api/models/platform/job-old/checkpoint.pt",
            params={"selected": "platform/job-old/checkpoint.pt"},
        )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("platform/job-old/checkpoint.pt", response.json()["detail"])
        self.assertTrue(old.is_file())

    def test_delete_drops_the_scan_cache(self) -> None:
        old = self._make_job("platform", "job-old")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)
        # Prime both caches.
        self.assertEqual(self._listing()["count"], 2)

        self.assertEqual(
            self.client.delete("/api/models/platform/job-old/checkpoint.pt").status_code, 200
        )
        # No ``force``: the delete itself must have invalidated the cache.
        cached = self.client.get("/api/models").json()
        self.assertEqual(cached["count"], 1)
        self.assertEqual(cached["items"][0]["path"], "platform/job-new/checkpoint.pt")

    def test_delete_supports_alternate_checkpoint_names(self) -> None:
        older = self._make_job("platform", "job-best", filename="checkpoint_best.pt")
        latest = self._make_job("platform", "job-latest", filename="checkpoint_latest.pt")
        newest = self._make_job("platform", "job-new")
        for index, path in enumerate((older, latest)):
            self._age(path, 7200 * (index + 1))
        self._age(newest, 0)

        for relative in (
            "platform/job-best/checkpoint_best.pt",
            "platform/job-latest/checkpoint_latest.pt",
        ):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/{relative}")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse((self.root / relative).exists())


class DeleteJobDirTests(ModelsTestCase):
    def test_delete_job_removes_everything(self) -> None:
        job_a = self._make_job(
            "platform",
            "job-a",
            extra={"metrics.json": "{}", "train.log": "log", "replays/r1.json": "{}"},
        )
        job_b = self._make_job("platform", "job-b")
        # job-a must not be the newest, or it would be protected.
        self._age(job_a, 7200)
        self._age(job_b, 0)
        job_dir = self.root / "platform" / "job-a"

        response = self.client.delete("/api/models/job/platform/job-a")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["deleted"], "platform/job-a")
        self.assertGreater(response.json()["freed_mb"], 0)
        self.assertFalse(job_dir.exists())
        self.assertTrue((self.root / "platform" / "job-b").is_dir())

    def test_deleting_root_is_rejected(self) -> None:
        self._make_job("platform", "job-a")
        for relative in ("", "%2E", "%2F", "platform%2F.."):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/job/{relative}")
                self.assertIn(response.status_code, (400, 404, 422), response.text)
        self.assertTrue(self.root.is_dir())
        self.assertTrue((self.root / "platform" / "job-a").is_dir())

    def test_deleting_category_dir_is_rejected(self) -> None:
        self._make_job("platform", "job-a")
        self._make_job("archive", "job-b")
        self._make_job("experiments", "bc_ironzero25")
        _write(self.root / "recordings" / "run-1" / "replay.json", "{}")
        _write(self.root / "_logs" / "x.log", "log")
        _write(self.root / "_scratch" / "junk.txt", "junk")

        for category in ("platform", "archive", "experiments", "recordings", "_logs", "_scratch"):
            with self.subTest(category=category):
                response = self.client.delete(f"/api/models/job/{category}")
                self.assertEqual(response.status_code, 400, response.text)
                self.assertTrue((self.root / category).is_dir())
        # Nothing below the categories was touched either.
        self.assertTrue((self.root / "platform" / "job-a").is_dir())
        self.assertTrue((self.root / "experiments" / "bc_ironzero25").is_dir())

    def test_deleting_a_nested_job_is_rejected(self) -> None:
        self._make_job("platform", "job-a")
        _write(self.root / "platform" / "job-a" / "replays" / "ep1.json", "{}")

        response = self.client.delete("/api/models/job/platform/job-a/replays")

        self.assertEqual(response.status_code, 400, response.text)
        self.assertTrue((self.root / "platform" / "job-a" / "replays").is_dir())

    def test_path_traversal_is_rejected_for_jobs(self) -> None:
        self._make_job("platform", "job-a")
        outside = self._tmp / "outside-job"
        _write(outside / "checkpoint.pt", "x")

        for relative in (
            "..%2Foutside-job",
            "platform%2F..%2F..%2Foutside-job",
            str(outside).replace("\\", "%5C").replace(":", "%3A"),
        ):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/job/{relative}")
                self.assertIn(response.status_code, (400, 404, 422), response.text)
        self.assertTrue(outside.is_dir())

    def test_missing_job_is_404(self) -> None:
        self._make_job("platform", "job-a")
        response = self.client.delete("/api/models/job/platform/job-ghost")
        self.assertEqual(response.status_code, 404, response.text)

    def test_protected_jobs_are_409(self) -> None:
        self._make_job("experiments", "bc_ironzero10")
        self._make_job("platform", "job-running", meta={"status": "running"})

        for relative in ("experiments/bc_ironzero10", "platform/job-running"):
            with self.subTest(relative=relative):
                response = self.client.delete(f"/api/models/job/{relative}")
                self.assertEqual(response.status_code, 409, response.text)
                self.assertTrue((self.root / relative).is_dir())

    def test_newest_job_dir_is_409(self) -> None:
        """Deleting the directory must not be a loophole around the newest-weight rule."""

        old = self._make_job("platform", "job-old", extra={"train.log": "x"})
        newest = self._make_job("platform", "job-new", extra={"train.log": "x"})
        self._age(old, 7200)
        self._age(newest, 0)
        listing = self._listing()
        self.assertTrue(self._item(listing, "platform/job-new/checkpoint.pt")["protected"])
        job = next(j for j in listing["jobs"] if j["job_dir"] == "platform/job-new")
        self.assertTrue(job["protected"])
        self.assertIn("最新", job["protected_reason"])

        response = self.client.delete("/api/models/job/platform/job-new")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue((self.root / "platform" / "job-new" / "checkpoint.pt").is_file())
        # The older job directory stays deletable.
        self.assertEqual(
            self.client.delete("/api/models/job/platform/job-old").status_code, 200
        )

    def test_selected_model_blocks_job_delete(self) -> None:
        old = self._make_job("platform", "job-a")
        newest = self._make_job("platform", "job-new")
        self._age(old, 7200)
        self._age(newest, 0)
        response = self.client.delete(
            "/api/models/job/platform/job-a",
            params={"selected": "platform/job-a/checkpoint.pt"},
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue((self.root / "platform" / "job-a").is_dir())
        # Without the selection it is deletable, so the 409 really came from it.
        self.assertEqual(
            self.client.delete("/api/models/job/platform/job-a").status_code, 200
        )

    def test_rmtree_failure_is_500_with_reason(self) -> None:
        older = self._make_job("platform", "job-a")
        newest = self._make_job("platform", "job-new")
        self._age(older, 7200)
        self._age(newest, 0)
        real_rmtree = shutil.rmtree

        def failing_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
            if str(path).endswith("job-a"):
                raise OSError("device busy")
            return real_rmtree(path, *args, **kwargs)

        shutil.rmtree = failing_rmtree  # type: ignore[assignment]
        try:
            response = self.client.delete("/api/models/job/platform/job-a")
        finally:
            shutil.rmtree = real_rmtree  # type: ignore[assignment]

        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("device busy", response.json()["detail"])


class ModelsEndToEndTests(ModelsTestCase):
    """The exact flow the panel performs: list -> delete -> smaller list -> 409."""

    def test_full_flow(self) -> None:
        old = self._make_job("platform", "rwpp-ppo-ppo-old", meta={"status": "succeeded"},
                             extra={"train.log": "done"})
        smoke = self._make_job("platform", "rwpp-ppo-ppo-smoke", meta={"status": "succeeded"},
                               extra={"train.log": "done"})
        newest = self._make_job("platform", "rwpp-ppo-ppo-new", meta={"status": "succeeded"})
        base = self._make_job("experiments", "bc_ironzero25")
        for path in (old, smoke, base):
            self._age(path, 10_000)
        self._age(newest, 0)

        listing = self._listing()
        self.assertEqual(listing["count"], 4)
        before_mb = listing["total_mb"]
        self.assertFalse(
            self._item(listing, "platform/rwpp-ppo-ppo-old/checkpoint.pt")["protected"]
        )

        # 1) single-file delete of an unprotected weight
        deleted = self.client.delete("/api/models/platform/rwpp-ppo-ppo-old/checkpoint.pt")
        self.assertEqual(deleted.status_code, 200, deleted.text)

        # 2) whole-job delete of a temp job
        job_deleted = self.client.delete("/api/models/job/platform/rwpp-ppo-ppo-smoke")
        self.assertEqual(job_deleted.status_code, 200, job_deleted.text)

        after = self._listing()
        self.assertEqual(after["count"], 2)
        self.assertLess(after["total_mb"], before_mb)
        self.assertEqual({item["path"] for item in after["items"]},
                         {"platform/rwpp-ppo-ppo-new/checkpoint.pt",
                          "experiments/bc_ironzero25/checkpoint.pt"})

        # 3) protected objects still refuse
        self.assertEqual(
            self.client.delete(
                "/api/models/experiments/bc_ironzero25/checkpoint.pt"
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.delete("/api/models/job/experiments/bc_ironzero25").status_code, 409
        )
        self.assertEqual(
            self.client.delete(
                "/api/models/platform/rwpp-ppo-ppo-new/checkpoint.pt"
            ).status_code,
            409,
        )
        self.assertTrue((self.root / "experiments" / "bc_ironzero25" / "checkpoint.pt").is_file())
        self.assertTrue(
            (self.root / "platform" / "rwpp-ppo-ppo-new" / "checkpoint.pt").is_file()
        )


class ModelsRootTests(ModelsTestCase):
    def test_runs_root_that_is_a_category_unlocks_its_parent(self) -> None:
        """``IRONZERO_RUNS=runs/platform`` must still reach sibling categories."""

        self._make_job("platform", "job-a")
        self._make_job("experiments", "bc_ironzero25")
        os.environ.pop("IRONZERO_MODELS_ROOT", None)
        real_runs_root = self.models.paths.RUNS_ROOT
        self.models.paths.RUNS_ROOT = self.root / "platform"
        self.models._clear_models_cache()
        try:
            response = self.client.get("/api/models", params={"force": "true"})
            self.assertEqual(response.status_code, 200, response.text)
            paths = {item["path"] for item in response.json()["items"]}
            self.assertIn("platform/job-a/checkpoint.pt", paths)
            self.assertIn("experiments/bc_ironzero25/checkpoint.pt", paths)
            self.assertEqual(response.json()["root"], Path(self.root).as_posix())
        finally:
            self.models.paths.RUNS_ROOT = real_runs_root
            self.models._clear_models_cache()


class WarmstartResolverTests(unittest.TestCase):
    """The BC path bug: warmstart used to look only at ``runs/bc_ironzero*``."""

    def setUp(self) -> None:
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scripts.rwpp_service import ppo

        self.ppo = ppo
        self._real_repo_root = ppo.REPO_ROOT
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-bc-")).resolve()
        ppo.REPO_ROOT = self._tmp

    def tearDown(self) -> None:
        self.ppo.REPO_ROOT = self._real_repo_root
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _bc(self, parent: str, units: int) -> Path:
        return _write(self._tmp / parent / f"bc_ironzero{units}" / "checkpoint.pt", "w")

    def test_finds_weights_under_runs_experiments(self) -> None:
        expected = self._bc("runs/experiments", 15)
        self.assertEqual(self.ppo.resolve_bc_checkpoint(15), expected)
        self.assertEqual(
            self.ppo.resolve_init_checkpoint({"init_mode": "warmstart"}, 15),
            str(expected.resolve()),
        )

    def test_still_supports_the_legacy_default_location(self) -> None:
        expected = self._bc("runs", 10)
        self.assertEqual(self.ppo.resolve_bc_checkpoint(10), expected)

    def test_prefers_the_exact_unit_count_then_falls_back_to_25(self) -> None:
        exact = self._bc("runs/experiments", 20)
        fallback = self._bc("runs/experiments", 25)
        self.assertEqual(self.ppo.resolve_bc_checkpoint(20), exact)
        # No 7v7 weights: fall back to the 25v25 ones rather than nothing.
        self.assertEqual(self.ppo.resolve_bc_checkpoint(7), fallback)
        self.assertEqual(self.ppo.resolve_bc_checkpoint(None), fallback)

    def test_scratch_and_explicit_paths_are_unchanged(self) -> None:
        self._bc("runs/experiments", 25)
        self.assertIsNone(self.ppo.resolve_init_checkpoint({"init_mode": "scratch"}, 25))
        explicit = self._bc("runs/experiments", 5)
        self.assertEqual(
            self.ppo.resolve_init_checkpoint(
                {"init_mode": "checkpoint", "init_checkpoint": str(explicit)}, 25
            ),
            str(explicit.resolve()),
        )
        self.assertIsNone(
            self.ppo.resolve_init_checkpoint(
                {"init_mode": "checkpoint", "init_checkpoint": "runs/missing.pt"}, 25
            )
        )

    def test_missing_weights_return_none(self) -> None:
        self.assertIsNone(self.ppo.resolve_bc_checkpoint(25))
        self.assertIsNone(self.ppo.resolve_init_checkpoint({"init_mode": "warmstart"}, 25))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
