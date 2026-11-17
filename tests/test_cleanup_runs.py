# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/cleanup_runs.py`` — the only script that deletes weights.

The script is dry-run by default, but ``--apply`` removes files without a recycle
bin, so every protection rule is pinned here against a throwaway tree. The rules
exist because each one was a real way to lose data:

* a cited weight (``econ_fix_long/checkpoint_best.pt``) is evidence in
  ``audit/`` — deleting it invalidates a published verification claim;
* ``bc_ironzero*`` weights are the PPO warm start;
* ``runs/experiments/`` holds the kept experiments;
* the newest weights are the web dropdown's default;
* a job that wrote anything inside the grace window may still be training;
* a job younger than ``--min-age-hours`` is someone's current result, not litter;
* ``runs/_logs/rwpp_service.stdout.log`` belongs to the *running* host service.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cleanup_runs.py"


def _load_module():
    """Import the script by path (``scripts/`` is not a package)."""

    spec = importlib.util.spec_from_file_location("cleanup_runs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cleanup_runs"] = module
    spec.loader.exec_module(module)
    return module


cleanup = _load_module()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str = "x", age_hours: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if age_hours:
        stamp = time.time() - age_hours * 3600.0
        os.utime(path, (stamp, stamp))
    return path


def _age_dir(directory: Path, age_hours: float) -> None:
    """Backdate a directory and everything in it."""

    stamp = time.time() - age_hours * 3600.0
    for path in [directory, *sorted(directory.rglob("*"))]:
        try:
            os.utime(path, (stamp, stamp))
        except OSError:
            continue


class _Tensor:
    """Stand-in for a torch tensor: ``_weights_equal`` only needs equality."""

    def __init__(self, value: float) -> None:
        self.value = value


# --------------------------------------------------------------------------- #
# planning: what gets deleted, what is protected
# --------------------------------------------------------------------------- #
class CleanupPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-cleanup-"))
        # ``build_plan`` resolves its root; resolve here too so the paths it plans
        # compare equal to the ones these tests construct.
        self.base = (self._tmp / "repo").resolve()
        self.runs = self.base / "runs"
        self.runs.mkdir(parents=True)
        self._real_load = cleanup._load_weights
        self._real_equal = cleanup._tensors_equal

    def tearDown(self) -> None:
        cleanup._load_weights = self._real_load
        cleanup._tensors_equal = self._real_equal
        shutil.rmtree(self._tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    def _stub_weights(self, payloads: dict[str, dict]) -> None:
        """Substitute the checkpoint loader (no torch, no multi-MB reads)."""

        cleanup._load_weights = lambda path: payloads[str(path)]  # type: ignore[assignment]
        cleanup._tensors_equal = lambda a, b: a.value == b.value  # type: ignore[assignment]

    def _plan(self, *, grace_minutes: float = 15.0, keep_newest: int = 0,
              min_age_hours: float = 24.0) -> tuple[Path, ...]:
        return tuple(path for _c, path, _s, _n in self._rows(
            grace_minutes=grace_minutes, keep_newest=keep_newest, min_age_hours=min_age_hours))

    def _rows(self, *, grace_minutes: float = 15.0, keep_newest: int = 0,
              min_age_hours: float = 24.0) -> tuple[tuple[str, Path, int, str], ...]:
        plan = cleanup.build_plan(
            grace_minutes=grace_minutes,
            keep_newest=keep_newest,
            min_age_hours=min_age_hours,
            repo_root=self.base,
        )
        return tuple(plan.rows)

    def _in_category(self, category: str, **kwargs) -> tuple[Path, ...]:
        return tuple(path for cat, path, _s, _n in self._rows(**kwargs) if cat == category)

    def _job(self, name: str, *, age_hours: float = 48.0) -> Path:
        directory = self.runs / "platform" / f"rwpp-ppo-ppo-{name}"
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("checkpoint.pt", "checkpoint_best.pt", "checkpoint_latest.pt"):
            _write(directory / filename, f"weights-{filename}")
        _write(directory / "live.jsonl", "{}\n")
        _age_dir(directory, age_hours)
        return directory

    # -- deletion targets --------------------------------------------------
    def test_jvm_spawn_logs_are_deleted(self) -> None:
        log = _write(self.runs / "_rwpp_ppo" / "game-1234-5678.log", "jvm", age_hours=48)
        self.assertIn(log, self._plan(keep_newest=0))

    def test_fresh_log_inside_grace_window_is_kept(self) -> None:
        # A running game is appending to its own log right now.
        log = _write(self.runs / "_rwpp_ppo" / "game-live-9999.log", "append")
        self.assertNotIn(log, self._plan(keep_newest=0))

    def test_pycache_is_deleted(self) -> None:
        cache = self.base / "src" / "__pycache__"
        _write(cache / "mod.cpython-311.pyc", "pyc")
        self.assertIn(cache, self._plan(keep_newest=0))

    def test_junk_dirs_are_deleted(self) -> None:
        for name in ("archive", "_speedprobe"):
            directory = self.runs / name
            _write(directory / "checkpoint.pt", "old", age_hours=48)
            _age_dir(directory, 48)
            self.assertIn(directory, self._plan(keep_newest=0))

    def test_scratch_is_gutted_but_reference_probe_survives(self) -> None:
        junk = _write(self.runs / "_scratch" / "dbg.py", "print()", age_hours=48)
        probe = _write(self.runs / "_scratch" / "probe_obs_type_blind.py", "# cited", age_hours=48)
        planned = self._plan(keep_newest=0)
        self.assertIn(junk, planned)
        self.assertNotIn(probe, planned)

    def test_live_service_stdout_log_survives_but_stale_rotations_go(self) -> None:
        live = _write(self.runs / "_logs" / "rwpp_service.stdout.log", "live", age_hours=48)
        rotated = _write(self.runs / "_logs" / "rwpp_service.stdout.20261113.log", "old", age_hours=48)
        planned = self._plan(keep_newest=0)
        self.assertNotIn(live, planned)
        self.assertIn(rotated, planned)

    # -- protections -------------------------------------------------------
    def test_weights_of_an_unreferenced_stale_job_are_deleted(self) -> None:
        directory = self._job("0123abcd")
        planned = self._plan(keep_newest=0)
        self.assertIn(directory / "checkpoint.pt", planned)
        self.assertIn(directory / "checkpoint_best.pt", planned)

    def test_bc_warmstart_weights_are_never_deleted(self) -> None:
        directory = self.runs / "experiments" / "bc_ironzero25"
        _write(directory / "checkpoint.pt", "bc")
        _age_dir(directory, 48)
        self.assertNotIn(directory / "checkpoint.pt", self._plan(keep_newest=0))

    def test_experiments_weights_are_never_deleted(self) -> None:
        directory = self.runs / "experiments" / "econ_train_full"
        _write(directory / "checkpoint.pt", "exp")
        _age_dir(directory, 48)
        self.assertNotIn(directory / "checkpoint.pt", self._plan(keep_newest=0))

    def test_job_id_cited_in_prose_is_protected(self) -> None:
        _write(self.base / "audit" / "report.md", "实测 `rwpp-ppo-ppo-deadbeef/checkpoint_best.pt`")
        directory = self._job("deadbeef")
        planned = self._plan(keep_newest=0)
        self.assertNotIn(directory / "checkpoint.pt", planned)
        self.assertNotIn(directory / "checkpoint_best.pt", planned)

    def test_weight_cited_by_filename_is_protected_even_when_the_run_is_gutted(self) -> None:
        # This is the real ``audit/闭环复核_20261114_B_训练评估.md:393`` citation:
        # the run itself is a one-off ``econ_fix_*`` dir, but that single file is
        # the checkpoint ``load_policy`` compatibility was verified against.
        _write(
            self.base / "audit" / "B.md",
            "`load_policy` 对 `schema_version` 缺失的 `econ_fix_long/checkpoint_best.pt` 仍能加载",
        )
        directory = self.runs / "econ_fix_long"
        kept = _write(directory / "checkpoint_best.pt", "cited", age_hours=48)
        doomed = _write(directory / "checkpoint.pt", "other", age_hours=48)
        _write(directory / "live.jsonl", "{}\n", age_hours=48)
        _age_dir(directory, 48)
        planned = self._plan(keep_newest=0)
        self.assertNotIn(kept, planned)
        self.assertIn(doomed, planned)

    def test_newest_weight_jobs_are_protected(self) -> None:
        old = self.runs / "platform" / "old-job"
        _write(old / "checkpoint.pt", "old", age_hours=100)
        _age_dir(old, 100)
        newest_dir = self.runs / "platform" / "new-job"
        newest = _write(newest_dir / "checkpoint.pt", "new", age_hours=48)
        _age_dir(newest_dir, 48)
        self.assertNotIn(newest, self._plan(keep_newest=1))

    def test_job_written_inside_grace_window_is_protected(self) -> None:
        directory = self._job("aaaabbbb", age_hours=48)
        # A heartbeat after the age gate has passed still marks it live.
        _write(directory / "live.jsonl", "{}\n", age_hours=0)
        self.assertNotIn(directory / "checkpoint.pt", self._plan(keep_newest=0))

    def test_job_younger_than_min_age_is_not_litter(self) -> None:
        # Finished two hours ago: outside the grace window, but someone may still
        # be looking at the result. Only ``--min-age-hours`` expiring makes its
        # weights stale.
        directory = self._job("ccccdddd", age_hours=2)
        stale = "stale-weights"
        self.assertNotIn(directory / "checkpoint.pt", self._in_category(stale, min_age_hours=24))
        self.assertIn(directory / "checkpoint.pt", self._in_category(stale, min_age_hours=1))

    def test_running_status_in_meta_is_protected(self) -> None:
        directory = self._job("eeeeffff", age_hours=48)
        _write(directory / "meta.json", '{"status": "running"}', age_hours=48)
        _age_dir(directory, 48)
        self.assertNotIn(directory / "checkpoint.pt", self._plan(keep_newest=0))

    # -- duplicate latest --------------------------------------------------
    def test_identical_latest_checkpoint_is_dropped(self) -> None:
        directory = self._job("11112222")
        main, latest = directory / "checkpoint.pt", directory / "checkpoint_latest.pt"
        self._stub_weights({
            str(main): {"update": 40, "model": {"w": _Tensor(1.0)}},
            str(latest): {"update": 40, "model": {"w": _Tensor(1.0)}},
        })
        self.assertIn(latest, self._plan(keep_newest=0))

    def test_diverged_latest_checkpoint_is_kept(self) -> None:
        directory = self._job("33334444")
        main, latest = directory / "checkpoint.pt", directory / "checkpoint_latest.pt"
        self._stub_weights({
            str(main): {"update": 40, "model": {"w": _Tensor(1.0)}},
            str(latest): {"update": 40, "model": {"w": _Tensor(2.0)}},
        })
        # The job is still stale enough for its *spare* weights to go, but the
        # duplicated-latest rule specifically must not fire on differing content.
        self.assertNotIn(latest, self._in_category("dup-latest"))
        self.assertNotIn(latest, self._in_category("dup-latest", keep_newest=3))

    def test_latest_older_than_final_is_dropped(self) -> None:
        directory = self._job("55556666")
        main, latest = directory / "checkpoint.pt", directory / "checkpoint_latest.pt"
        self._stub_weights({
            str(main): {"update": 40, "model": {"w": _Tensor(1.0)}},
            str(latest): {"update": 20, "model": {"w": _Tensor(0.5)}},
        })
        self.assertIn(latest, self._plan(keep_newest=0))


# --------------------------------------------------------------------------- #
# busy detection
# --------------------------------------------------------------------------- #
class BusyDetectionTests(unittest.TestCase):
    """``--apply`` must not delete weights while the host service is writing."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-busy-"))
        self.base = self._tmp / "repo"
        (self.base / "runs").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_idle_service_reports_not_busy(self) -> None:
        _write(self.base / "runs" / "_logs" / "rwpp_service.stdout.log", "x", age_hours=48)
        self.assertEqual(
            cleanup._logs_are_being_written(time.time() - 900, self.base), ""
        )

    def test_fresh_stdout_write_reports_busy(self) -> None:
        _write(self.base / "runs" / "_logs" / "rwpp_service.stdout.log", "x")
        self.assertTrue(cleanup._logs_are_being_written(time.time() - 900, self.base))

    def test_fresh_job_file_reports_busy(self) -> None:
        _write(self.base / "runs" / "_rwpp_service" / "jobs" / "rwpp-abc.json", "{}")
        self.assertTrue(cleanup._logs_are_being_written(time.time() - 900, self.base))


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
class ApplyTests(unittest.TestCase):
    """``_apply`` deletes exactly the planned rows and tolerates races."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-apply-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_apply_removes_only_planned_paths(self) -> None:
        doomed = _write(self._tmp / "junk" / "game-1.log", "log")
        kept = _write(self._tmp / "keep" / "checkpoint.pt", "w")
        plan = cleanup.Plan()
        plan.add("logs", doomed)
        deleted, failed = cleanup._apply(plan)
        self.assertEqual((deleted, failed), (1, 0))
        self.assertFalse(doomed.exists())
        self.assertTrue(kept.exists())

    def test_missing_target_is_not_a_failure(self) -> None:
        # The host service can rotate a log away between planning and deleting.
        gone = self._tmp / "junk" / "game-2.log"
        plan = cleanup.Plan()
        plan.add("logs", gone, size=0)
        deleted, failed = cleanup._apply(plan)
        self.assertEqual((deleted, failed), (0, 0))

    def test_plan_does_not_double_count_a_parent_and_its_child(self) -> None:
        directory = self._tmp / "archive"
        child = _write(directory / "checkpoint.pt", "w")
        plan = cleanup.Plan()
        plan.add("junk-dirs", directory, size=1)
        plan.add("junk-dirs", child, size=1)
        self.assertEqual(len(plan.rows), 1)
        self.assertEqual(plan.rows[0][1], directory)


if __name__ == "__main__":
    unittest.main()
