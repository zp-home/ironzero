# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the "web dashboard could not actually train" cluster.

Every case here was reproduced on the real stack before being fixed; each one
used to fail *silently*, which is why none of them had a test yet:

* ``/ppo`` died with a bare connection reset because ``SystemExit`` escaped an
  ``except Exception`` (``make_ironzero_map`` fails via ``SystemExit``);
* the arena runtime was resolved two different ways, so the map generator looked
  for ``Artillery.tmx`` in a directory that does not exist;
* ``RwppPpoRequest`` dropped 13 request keys (AI teacher, physics_ms, arena
  preset) and still answered 200;
* ``/ppo`` forwarded container paths ``/app/runs/...`` unchanged, so a requested
  checkpoint was "not found" on the host and training restarted from scratch;
* ``load_policy`` rebuilt the network from the *checkpoint's* layout constants,
  so any checkpoint saved before ``global_dim`` grew 12 -> 15 crashed the encoder;
* ``_run_managed_proc`` decoded child output with the locale codec, turning any
  decoding error into an empty log with ``error=None``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _load_models_module():
    spec = importlib.util.spec_from_file_location(
        "ironzero_request_models_regression", REPO / "web" / "backend" / "app" / "models.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArenaRuntimeResolutionTests(unittest.TestCase):
    """The map generator and the trainer must agree on the runtime directory."""

    def setUp(self) -> None:
        scripts = REPO / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import make_ironzero_map

        self.module = make_ironzero_map

    def test_same_default_as_the_training_env(self) -> None:
        from ironzero.rwpp.config import DEFAULT_RUNTIME as ENV_RUNTIME

        # Nothing pinned in the environment: both sides fall back to the same build.
        if os.environ.get("IRONZERO_RWPP_RUNTIME"):
            pytest.skip("IRONZERO_RWPP_RUNTIME is set; the fallback is not exercised")
        self.assertEqual(self.module.DEFAULT_RUNTIME, ENV_RUNTIME)

    def test_env_override_wins(self) -> None:
        target = tempfile.mkdtemp(prefix="ironzero-runtime-")
        previous = os.environ.get("IRONZERO_RWPP_RUNTIME")
        os.environ["IRONZERO_RWPP_RUNTIME"] = target
        try:
            self.assertEqual(self.module.resolve_repo_runtime(), Path(target))
        finally:
            if previous is None:
                os.environ.pop("IRONZERO_RWPP_RUNTIME", None)
            else:
                os.environ["IRONZERO_RWPP_RUNTIME"] = previous

    def test_module_level_constant_is_the_resolved_value(self) -> None:
        self.assertEqual(self.module.DEFAULT_RUNTIME, self.module.resolve_repo_runtime())


class ResolvePresetSystemExitTests(unittest.TestCase):
    """``ensure_arena_map`` fails with ``SystemExit``; it must not kill the thread."""

    def setUp(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service import ppo

        self.ppo = ppo

    def test_system_exit_is_reported_not_propagated(self) -> None:
        import make_ironzero_map

        real = make_ironzero_map.ensure_arena_map

        def boom(*_args, **_kwargs):
            raise SystemExit("source map not found: /nonexistent/Artillery.tmx")

        make_ironzero_map.ensure_arena_map = boom
        try:
            preset = self.ppo.resolve_preset(
                {"units_per_team": 25, "arena": True}, "IronZero25", "maps/ironzero/IronZero25.tmx"
            )
        finally:
            make_ironzero_map.ensure_arena_map = real
        self.assertTrue(preset["arena_error"], "the failure must be surfaced on the preset")
        self.assertIn("source map not found", preset["arena_error"])
        # The request still describes what was asked for.
        self.assertEqual(preset["map_name"], "IronZero25")

    def test_successful_resolution_has_no_error(self) -> None:
        import make_ironzero_map

        real = make_ironzero_map.ensure_arena_map
        make_ironzero_map.ensure_arena_map = lambda units: (
            Path("configs/maps/IronZero25.tmx"),
            f"maps/ironzero/IronZero{units}.tmx",
            f"IronZero{units}",
        )
        try:
            preset = self.ppo.resolve_preset(
                {"units_per_team": 10, "arena": True}, "IronZero10", "maps/ironzero/IronZero10.tmx"
            )
        finally:
            make_ironzero_map.ensure_arena_map = real
        self.assertEqual(preset["arena_error"], "")
        self.assertEqual(preset["map_path"], "maps/ironzero/IronZero10.tmx")
        self.assertTrue(preset["unlock_ai"])
        self.assertEqual(preset["win_mode"], "annihilation")


class ArenaFailureIsRejectedTests(unittest.TestCase):
    """A request whose arena map could not be prepared must be a 400, not a silent run."""

    def setUp(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service.launch import StartMixin
        from rwpp_service import state

        state.MAX_PARALLEL = 10
        import threading

        state.PORT_SLOTS.queue.clear()

        class Handler(StartMixin):
            def __init__(self) -> None:
                self.sent: list[tuple[dict, int]] = []

            def _send(self, payload, status=200):  # noqa: ANN001
                self.sent.append((payload, status))

        self.Handler = Handler
        self.state = state
        self._thread = threading

    def test_arena_error_refuses_the_job(self) -> None:
        import rwpp_service.ppo as ppo_module

        real_preset = ppo_module.resolve_preset

        def failing_preset(payload, map_name, map_path):  # noqa: ANN001
            result = real_preset(payload, "Artillery", "maps/challenge/Artillery.tmx")
            result["arena"] = True
            result["arena_error"] = "arena map preparation failed: source map not found"
            return result

        import rwpp_service.launch as launch_module

        launch_module.resolve_preset = failing_preset
        try:
            handler = self.Handler()
            handler._start_ppo({"updates": 1, "envs": 1, "map_name": "IronZero25"})
        finally:
            launch_module.resolve_preset = real_preset
        self.assertEqual(len(handler.sent), 1)
        payload, status = handler.sent[0]
        self.assertEqual(status, 400)
        self.assertIn("arena map preparation failed", payload["error"])

    def test_missing_explicit_checkpoint_is_refused(self) -> None:
        # No running job exists (fresh state), so this only depends on the
        # init-checkpoint check.
        handler = self.Handler()
        handler._start_ppo(
            {
                "updates": 1,
                "envs": 1,
                "map_name": "Artillery",
                "map_path": "maps/challenge/Artillery.tmx",
                "units_per_team": 22,
                "init_mode": "checkpoint",
                "init_checkpoint": "/app/runs/does-not-exist/checkpoint.pt",
            }
        )
        self.assertEqual(len(handler.sent), 1)
        payload, status = handler.sent[0]
        self.assertEqual(status, 400, payload)
        self.assertIn("init_checkpoint not found", payload["error"])


class ResolveInitStateTests(unittest.TestCase):
    """``/app/...`` must resolve, and an explicit miss must be an error."""

    def setUp(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service import ppo

        self.ppo = ppo
        self._real_root = ppo.REPO_ROOT
        self._tmp = Path(tempfile.mkdtemp(prefix="ironzero-initstate-")).resolve()
        ppo.REPO_ROOT = self._tmp

    def tearDown(self) -> None:
        self.ppo.REPO_ROOT = self._real_root
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_container_prefix_is_stripped(self) -> None:
        weight = self._tmp / "runs" / "platform" / "job" / "checkpoint.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"x")
        state = self.ppo.resolve_init_state(
            {"init_mode": "checkpoint", "init_checkpoint": "/app/runs/platform/job/checkpoint.pt"}, 25
        )
        self.assertEqual(state["error"], "")
        self.assertEqual(Path(state["checkpoint"]), weight.resolve())

    def test_missing_explicit_path_is_an_error(self) -> None:
        state = self.ppo.resolve_init_state(
            {"init_mode": "checkpoint", "init_checkpoint": "/app/runs/missing/checkpoint.pt"}, 25
        )
        self.assertIsNone(state["checkpoint"])
        self.assertIn("init_checkpoint not found", state["error"])

    def test_warmstart_falls_back_without_an_error(self) -> None:
        weight = self._tmp / "runs" / "experiments" / "bc_ironzero10" / "checkpoint.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"x")
        state = self.ppo.resolve_init_state({"init_mode": "warmstart"}, 10)
        self.assertEqual(state["error"], "")
        self.assertEqual(Path(state["checkpoint"]), weight.resolve())

    def test_scratch_ignores_everything(self) -> None:
        state = self.ppo.resolve_init_state({"init_mode": "scratch"}, 25)
        self.assertIsNone(state["checkpoint"])
        self.assertEqual(state["error"], "")


class PpoRequestContractTests(unittest.TestCase):
    """Every key the host service reads must survive the backend model."""

    #: Keys ``scripts/rwpp_service/launch.py::_start_ppo`` reads from the payload.
    HOST_KEYS = (
        "updates", "envs", "rollout_steps", "episode_ticks", "device", "map_name",
        "map_path", "map_width", "map_height", "frame_ms", "physics_ms",
        "vision_radius", "eval_every", "eval_episodes", "use_llm", "ai_pilot",
        "record_ai_orders", "ai_difficulty", "enemy_ai_difficulty", "units_per_team",
        "max_entities", "target_slots", "terrain", "win_mode", "unlock_ai",
        "init_checkpoint", "init_mode", "arena", "timeout_penalty", "num_teams",
        "fast_forward", "lockstep", "lockstep_timeout_ms",
    )

    def setUp(self) -> None:
        self.models = _load_models_module()

    def test_no_host_key_is_silently_dropped(self) -> None:
        dumped = self.models.RwppPpoRequest().model_dump()
        missing = [key for key in self.HOST_KEYS if key not in dumped]
        self.assertEqual(missing, [], f"backend drops host-service keys: {missing}")

    def test_unknown_key_is_rejected_instead_of_ignored(self) -> None:
        with pytest.raises(Exception):
            self.models.RwppPpoRequest(not_a_real_switch=True)

    def test_ai_teacher_switch_survives(self) -> None:
        model = self.models.RwppPpoRequest(
            ai_pilot=True, record_ai_orders=True, ai_difficulty=2, enemy_ai_difficulty=1
        )
        dumped = model.model_dump()
        self.assertTrue(dumped["ai_pilot"])
        self.assertTrue(dumped["record_ai_orders"])
        self.assertEqual(dumped["ai_difficulty"], 2)
        self.assertEqual(dumped["enemy_ai_difficulty"], 1)

    def test_physics_ms_and_arena_preset_survive(self) -> None:
        dumped = self.models.RwppPpoRequest(
            physics_ms=16, arena=True, max_entities=25, target_slots=26,
            terrain="configs/maps/IronZero25.terrain.json", win_mode="annihilation",
            unlock_ai=True,
        ).model_dump()
        self.assertEqual(dumped["physics_ms"], 16)
        self.assertTrue(dumped["arena"])
        self.assertEqual(dumped["max_entities"], 25)
        self.assertEqual(dumped["target_slots"], 26)
        self.assertEqual(dumped["terrain"], "configs/maps/IronZero25.terrain.json")
        self.assertEqual(dumped["win_mode"], "annihilation")
        self.assertTrue(dumped["unlock_ai"])


class AccelerationContractTests(unittest.TestCase):
    """``fast_forward`` / ``lockstep`` must survive every layer of the request path.

    Same failure mode as the 13 dropped keys before them: pydantic's default
    ``ignore`` policy or a missing payload read would turn "ask for 2x game
    speed" into "run at 1x, answer 200". The contract (audit/加速方案_契约.md §3)
    is: 1..32, 1 = off, out-of-range is a **400** whose text is exactly
    ``fast_forward must be 1..32``.
    """

    def setUp(self) -> None:
        self.models = _load_models_module()

    def test_defaults_are_off_and_host_decides_lockstep(self) -> None:
        dumped = self.models.RwppPpoRequest().model_dump()
        self.assertEqual(dumped["fast_forward"], 1, "1 = off must be the default")
        self.assertIsNone(
            dumped["lockstep"],
            "an omitted lockstep must stay NULL so the host picks the runner default",
        )
        self.assertEqual(dumped["lockstep_timeout_ms"], 30000)

    def test_request_and_ppo_models_both_carry_the_two_switches(self) -> None:
        record = self.models.RwppRecordRequest(fast_forward=4, lockstep=False).model_dump()
        self.assertEqual(record["fast_forward"], 4)
        self.assertFalse(record["lockstep"])
        ppo = self.models.RwppPpoRequest(fast_forward=2, lockstep=True).model_dump()
        self.assertEqual(ppo["fast_forward"], 2)
        self.assertTrue(ppo["lockstep"])

    def test_backend_rejects_out_of_range_with_the_contract_message(self) -> None:
        """A pydantic ``ge``/``le`` pair would answer 422 with pydantic's wording."""

        from fastapi.testclient import TestClient

        import web.backend.app.routers.rwpp as rwpp_router
        from web.backend.app import app

        client = TestClient(app)
        real_call = rwpp_router._rwpp_call
        rwpp_router._rwpp_call = lambda *a, **k: {"job_id": "x"}  # type: ignore[assignment]
        try:
            for path in ("/api/rwpp/infer", "/api/rwpp/record", "/api/rwpp/train", "/api/rwpp/ppo"):
                for bad in (0, 33, -1):
                    with self.subTest(path=path, fast_forward=bad):
                        response = client.post(path, json={"fast_forward": bad})
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertIn("fast_forward must be 1..32", response.text)
                # A non-integer is a *type* error, so pydantic refuses it (422)
                # before the range check runs. That is still a refusal -- the
                # one thing that must never happen is a 200 that silently ran
                # the default speed instead of what was asked for.
                with self.subTest(path=path, fast_forward="abc"):
                    response = client.post(path, json={"fast_forward": "abc"})
                    self.assertIn(response.status_code, (400, 422), response.text)
                    if response.status_code == 400:
                        self.assertIn("fast_forward must be 1..32", response.text)
            # The bounds themselves are accepted.
            for good in (1, 2, 32):
                with self.subTest(fast_forward=good):
                    self.assertEqual(
                        client.post("/api/rwpp/infer", json={"fast_forward": good}).status_code,
                        200,
                    )
        finally:
            rwpp_router._rwpp_call = real_call  # type: ignore[assignment]

    def test_host_service_rejects_out_of_range_and_accepts_the_bounds(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service.launch import acceleration_error, resolve_acceleration

        self.assertIsNone(acceleration_error(None))
        self.assertIsNone(acceleration_error(1))
        self.assertIsNone(acceleration_error(32))
        self.assertEqual(acceleration_error(0), "fast_forward must be 1..32")
        self.assertEqual(acceleration_error(33), "fast_forward must be 1..32")
        self.assertEqual(acceleration_error("abc"), "fast_forward must be 1..32")
        # Missing keys must fall back to the defaults instead of raising.
        defaults = resolve_acceleration({}, lockstep_default=True)
        self.assertEqual(defaults["fast_forward"], 1)
        self.assertTrue(defaults["lockstep"])
        self.assertFalse(resolve_acceleration({}, lockstep_default=False)["lockstep"])
        # An explicit "false" string must not read as True.
        self.assertFalse(
            resolve_acceleration({"lockstep": "false"}, lockstep_default=True)["lockstep"]
        )

    def test_lockstep_defaults_per_runner(self) -> None:
        """On for policy-driven work, off for the recorder.

        ``record_rwpp_replay.py`` only sends an order every N observations, so
        blocking after *every* observation would stall the engine on the frames it
        has nothing to say.
        """

        from rwpp_service.proc import _accel_flags

        self.assertEqual(_accel_flags(), [], "a default request must add no flags")
        self.assertEqual(_accel_flags(1, False), [])
        self.assertEqual(_accel_flags(2, True), ["--fast-forward", "2", "--lockstep"])
        self.assertEqual(
            _accel_flags(4, True, 45000),
            ["--fast-forward", "4", "--lockstep", "--lockstep-timeout-ms", "45000"],
        )
        # Out-of-range values are clamped, never echoed into the child command.
        self.assertEqual(_accel_flags(99)[:2], ["--fast-forward", "32"])
        self.assertEqual(_accel_flags(0), [])

    def test_episode_timeout_is_divided_by_fast_forward(self) -> None:
        from rwpp_service.proc import _episode_timeout

        plain = _episode_timeout(20000, 16)
        sped = _episode_timeout(20000, 16, 2)
        self.assertLess(sped, plain, "a 2x match needs about half the wall clock")
        # The 120 s floor survives: boot + map load + replay write do not speed up.
        self.assertEqual(_episode_timeout(120, 0, 32), 120.0)


class AccelerationJobRecordTests(unittest.TestCase):
    """The job record and the runner call must carry the resolved acceleration.

    ``/infer`` used to be pinned to ``frame_ms=2``: the payload key was read, but
    the dashboard had no control and ``feInferRun.js`` fell back to a literal 2,
    so "0 = unpaced" could never be selected from the UI. That is why these
    tests assert the *record* and the *runner call*, not just the payload read.
    """

    def setUp(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        import threading

        from rwpp_service import launch as launch_module
        from rwpp_service import state

        class Handler(launch_module.StartMixin):
            def __init__(self) -> None:
                self.sent: list[tuple[dict, int]] = []

            def _send(self, payload, status=200):  # noqa: ANN001
                self.sent.append((payload, status))

            def _ppo_running(self) -> bool:
                return False

        self.launch = launch_module
        self.state = state
        self.Handler = Handler
        self._threading = threading
        state.PORT_SLOTS.queue.clear()
        for slot in range(10):
            state.PORT_SLOTS.put(5130 + slot)
        state.JOBS.clear()
        self.calls: list[tuple[str, dict]] = []
        self._real = {
            "infer": launch_module.run_infer,
            "train": launch_module.run_train,
            "ppo": launch_module.run_ppo,
        }

        def make_spy(kind):
            def spy(job_id, *args, **kwargs):  # noqa: ANN001
                self.calls.append((kind, kwargs))

            return spy

        launch_module.run_infer = make_spy("infer")
        launch_module.run_train = make_spy("train")
        launch_module.run_ppo = make_spy("ppo")

    def tearDown(self) -> None:
        self.launch.run_infer = self._real["infer"]
        self.launch.run_train = self._real["train"]
        self.launch.run_ppo = self._real["ppo"]
        self.state.JOBS.clear()
        self.state.PORT_SLOTS.queue.clear()
        for slot in range(10):
            self.state.PORT_SLOTS.put(5130 + slot)

    def _start(self, kind: str, payload: dict) -> tuple[dict, int]:
        handler = self.Handler()
        handler._start(kind=kind, payload=payload)
        # The runner runs on a daemon thread; let it record its call.
        for thread in self._threading.enumerate():
            if thread.daemon and thread is not self._threading.current_thread():
                thread.join(timeout=5.0)
        self.assertEqual(len(handler.sent), 1, handler.sent)
        return handler.sent[0]

    def test_out_of_range_fast_forward_is_a_400_with_the_contract_message(self) -> None:
        for bad in (0, 33, -1, "abc", 100):
            with self.subTest(fast_forward=bad):
                payload, status = self._start("infer", {"ticks": 600, "fast_forward": bad})
                self.assertEqual(status, 400, payload)
                self.assertEqual(payload["error"], "fast_forward must be 1..32")
        self.assertEqual(self.calls, [], "a refused request must not start a game")

    def test_infer_records_and_forwards_the_acceleration(self) -> None:
        payload, status = self._start(
            "infer", {"ticks": 600, "frame_ms": 0, "fast_forward": 4, "lockstep": False}
        )
        self.assertEqual(status, 200, payload)
        job_id = payload["job_id"]
        record = self.state.JOBS[job_id]
        self.assertEqual(record["frame_ms"], 0)
        self.assertEqual(record["fast_forward"], 4)
        self.assertFalse(record["lockstep"])
        kind, kwargs = self.calls[0]
        self.assertEqual(kind, "infer")
        self.assertEqual(kwargs["fast_forward"], 4)
        self.assertEqual(kwargs["frame_ms"], 0)
        self.assertIs(kwargs["lockstep"], False)

    def test_infer_lockstep_follows_the_runner_when_omitted(self) -> None:
        """"Absent" must stay "the runner decides", not a frozen ``False``.

        A scripted recorder only sends an order every N observations; a model
        player answers every one. Freezing either choice onto both would either
        stall the engine or drop decisions.
        """

        _payload, status = self._start("infer", {"ticks": 600, "fast_forward": 2})
        self.assertEqual(status, 200)
        _kind, kwargs = self.calls[-1]
        self.assertIsNone(
            kwargs["lockstep"],
            "an omitted lockstep must reach run_infer as None (runner default)",
        )

    def test_defaults_are_the_historical_ones(self) -> None:
        payload, status = self._start("infer", {"ticks": 600})
        self.assertEqual(status, 200, payload)
        record = self.state.JOBS[payload["job_id"]]
        self.assertEqual(record["fast_forward"], 1)
        self.assertEqual(record["frame_ms"], 2, "the /infer default stays 2 ms")
        self.assertFalse(record["lockstep"])
        self.assertEqual(record["lockstep_timeout_ms"], 30000)

    def test_train_forwards_the_acceleration_to_every_episode(self) -> None:
        payload, status = self._start(
            "train",
            {
                "ticks": 600,
                "policy": "scripted",
                "episodes": 2,
                "concurrency": 1,
                "frame_ms": 0,
                "fast_forward": 2,
                "lockstep": True,
            },
        )
        self.assertEqual(status, 200, payload)
        _kind, kwargs = self.calls[-1]
        self.assertEqual(kwargs["fast_forward"], 2)
        self.assertTrue(kwargs["lockstep"])
        self.assertEqual(kwargs["map_width"] >= 0, True)
        record = self.state.JOBS[payload["job_id"]]
        self.assertEqual(record["fast_forward"], 2)
        self.assertTrue(record["lockstep"])


class PpoRoutePathRewriteTests(unittest.TestCase):
    """``/ppo`` must translate container paths just like ``/infer`` already did."""

    def setUp(self) -> None:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from fastapi.testclient import TestClient

        import web.backend.app.routers.rwpp as rwpp_router
        from web.backend.app import app

        self.router = rwpp_router
        self.client = TestClient(app)
        self.captured: dict[str, dict] = {}
        self._real_call = rwpp_router._rwpp_call

        def fake_call(path, payload=None, timeout=10.0):  # noqa: ANN001
            self.captured[path] = payload
            return {"job_id": "ppo-test", "status": "running", "kind": "ppo"}

        rwpp_router._rwpp_call = fake_call

    def tearDown(self) -> None:
        self.router._rwpp_call = self._real_call

    def test_ppo_rewrites_the_container_prefix(self) -> None:
        response = self.client.post(
            "/api/rwpp/ppo",
            json={"init_mode": "checkpoint", "init_checkpoint": "/app/runs/platform/job/checkpoint.pt"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = self.captured["/ppo"]
        self.assertEqual(forwarded["init_checkpoint"], "runs/platform/job/checkpoint.pt")

    def test_ppo_forwards_the_new_switches(self) -> None:
        response = self.client.post(
            "/api/rwpp/ppo",
            json={"physics_ms": 16, "ai_pilot": True, "ai_difficulty": 2, "arena": True,
                  "unlock_ai": True, "win_mode": "annihilation", "target_slots": 26},
        )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = self.captured["/ppo"]
        self.assertEqual(forwarded["physics_ms"], 16)
        self.assertTrue(forwarded["ai_pilot"])
        self.assertEqual(forwarded["ai_difficulty"], 2)
        self.assertTrue(forwarded["arena"])
        self.assertTrue(forwarded["unlock_ai"])
        self.assertEqual(forwarded["win_mode"], "annihilation")
        self.assertEqual(forwarded["target_slots"], 26)

    def test_ppo_forwards_the_acceleration_switches(self) -> None:
        response = self.client.post(
            "/api/rwpp/ppo", json={"fast_forward": 2, "lockstep": True}
        )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = self.captured["/ppo"]
        self.assertEqual(forwarded["fast_forward"], 2)
        self.assertIs(forwarded["lockstep"], True)

    def test_infer_forwards_frame_ms_and_fast_forward(self) -> None:
        """``/infer`` used to carry no working pacing control at all (pinned at 2)."""

        response = self.client.post(
            "/api/rwpp/infer", json={"ticks": 300, "frame_ms": 0, "fast_forward": 4}
        )
        self.assertEqual(response.status_code, 200, response.text)
        forwarded = self.captured["/infer"]
        self.assertEqual(forwarded["frame_ms"], 0)
        self.assertEqual(forwarded["fast_forward"], 4)

    def test_infer_rewrite_still_works(self) -> None:
        response = self.client.post(
            "/api/rwpp/infer", json={"ticks": 300, "policy": "model", "checkpoint": "/app/runs/x/c.pt"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.captured["/infer"]["checkpoint"], "runs/x/c.pt")


class OldCheckpointLayoutTests(unittest.TestCase):
    """A checkpoint saved before ``global_dim`` grew must still produce a policy."""

    def setUp(self) -> None:
        import torch

        from ironzero.micro_config import DEFAULT_CONFIG, MicroDuelConfig
        from ironzero.training.policy import MaskedActorCritic

        self.torch = torch
        self.DEFAULT_CONFIG = DEFAULT_CONFIG
        self.MicroDuelConfig = MicroDuelConfig
        self.MaskedActorCritic = MaskedActorCritic

    def _write_old_checkpoint(self, directory: Path, units: int) -> Path:
        old = self.MicroDuelConfig(
            global_dim=12,
            type_features=0,
            units_per_team=units,
            max_entities=units,
            target_slots=units + 1,
            map_channels=3,
        )
        policy = self.MaskedActorCritic(old.flat_obs_dim, nvec=(5, 5, 8, 8, units + 1), hidden=64)
        path = directory / "checkpoint.pt"
        self.torch.save(
            {"model": policy.state_dict(), "micro_config": old.__dict__ if hasattr(old, "__dict__") else {},
             "ppo_config": {"hidden": 64}, "update": 0},
            str(path),
        )
        return path

    def test_saved_layout_constants_are_ignored(self) -> None:
        from dataclasses import asdict

        from ironzero.observation.encoding import encode_observation
        from ironzero.training.ppo import load_policy

        docs = Path(tempfile.mkdtemp(prefix="ironzero-oldckpt-"))
        try:
            old = self.MicroDuelConfig(
                global_dim=12, type_features=0, units_per_team=8, max_entities=8,
                target_slots=9, map_channels=3,
            )
            policy = self.MaskedActorCritic(old.flat_obs_dim, nvec=(5, 5, 8, 8, 9), hidden=64)
            path = docs / "checkpoint.pt"
            self.torch.save(
                {"model": policy.state_dict(), "micro_config": asdict(old),
                 "ppo_config": {"hidden": 64}, "update": 0},
                str(path),
            )

            loaded, micro = load_policy(str(path))
            # The layout comes from today's code, not from the file.
            self.assertEqual(micro.global_dim, self.DEFAULT_CONFIG.global_dim)
            self.assertEqual(micro.type_features, self.DEFAULT_CONFIG.type_features)
            # The scenario still comes from the file.
            self.assertEqual(micro.units_per_team, 8)
            self.assertEqual(micro.max_entities, 8)

            state = {
                "tick": 5,
                "own_entities": [{"x": 10.0, "y": 10.0, "hp": 210.0, "max_hp": 210.0}],
                "visible_enemy_entities": [{"x": 90.0, "y": 90.0, "hp": 210.0, "max_hp": 210.0}],
                "global_state": {"funds": 500.0, "income": 3.0},
                "resource_slots": [],
            }
            obs = encode_observation(state, micro)  # used to AssertionError here
            with self.torch.no_grad():
                loaded(self.torch.as_tensor(obs).unsqueeze(0))
        finally:
            import shutil

            shutil.rmtree(docs, ignore_errors=True)

    def test_block_remap_keeps_each_weight_on_its_own_feature(self) -> None:
        """An inserted block must not shift the retained trunk columns."""

        from ironzero.training.ppo import _column_source

        old = {"global_dim": 12, "entity_features": 2, "type_features": 0,
               "max_entities": 2, "position_grid": 2, "map_grid": 2, "map_channels": 1}
        new = {"global_dim": 15, "entity_features": 2, "type_features": 0,
               "max_entities": 2, "position_grid": 2, "map_grid": 2, "map_channels": 1}
        mapping = _column_source(old, new)
        self.assertIsNotNone(mapping)
        # old layout: global[0:12] own[12:16] enemy[16:20] grid[20:24]
        # new layout: global[0:15] own[15:19] enemy[19:23] grid[23:27]
        self.assertEqual(mapping[0], 0)      # global 0 stays global 0
        self.assertEqual(mapping[11], 11)    # global 11 stays global 11
        self.assertIsNone(mapping[12])       # new global 12 did not exist
        self.assertIsNone(mapping[13])
        self.assertIsNone(mapping[14])
        self.assertEqual(mapping[15], 12)    # own entity 0 starts at old column 12
        self.assertEqual(mapping[18], 15)    # own entity 1 last feature
        self.assertEqual(mapping[19], 16)    # enemy entity 0
        self.assertEqual(mapping[23], 20)    # grid block
        self.assertEqual(len(mapping), 27)

    def test_identical_layouts_need_no_remap(self) -> None:
        from ironzero.training.ppo import _column_source

        layout = {"global_dim": 15, "entity_features": 10, "type_features": 7,
                  "max_entities": 25, "position_grid": 8, "map_grid": 8, "map_channels": 4}
        self.assertIsNone(_column_source(layout, dict(layout)))

    def test_remap_reproduces_the_original_logits(self) -> None:
        """Load-time equivalence: same weights, same state, same shared logits."""

        from ironzero.micro_config import MicroDuelConfig
        from ironzero.observation.encoding import encode_observation
        from ironzero.training.ppo import _column_source, _migrate_state_dict

        old = MicroDuelConfig(global_dim=12, type_features=0, units_per_team=4,
                              max_entities=4, target_slots=5, map_channels=3)
        new = MicroDuelConfig(units_per_team=4, max_entities=4, target_slots=5, map_channels=3)

        p_old = self.MaskedActorCritic(old.flat_obs_dim, nvec=old.action_nvec, hidden=32)
        # Deliberately deterministic, distinct weights.
        with self.torch.no_grad():
            for index, parameter in enumerate(p_old.parameters()):
                parameter.copy_(self.torch.full_like(parameter, 0.01 * (index + 1)))
        sd = p_old.state_dict()

        p_new = self.MaskedActorCritic(new.flat_obs_dim, nvec=new.action_nvec, hidden=32)
        merged = _migrate_state_dict(p_new.state_dict(), sd, source_layout=old, target_layout=new)
        p_new.load_state_dict(merged)
        p_old.eval()
        p_new.eval()

        state = {
            "tick": 30,
            "own_entities": [{"x": 400.0, "y": 900.0, "hp": 210.0, "max_hp": 210.0}],
            "visible_enemy_entities": [{"x": 2200.0, "y": 1600.0, "hp": 210.0, "max_hp": 210.0}],
            "global_state": {"funds": 1200.0, "income": 6.0},
            "resource_slots": [],
        }
        v_new = encode_observation(state, new)
        v_old = [v_new[index] for index in range(old.global_dim)] + list(
            v_new[new.global_dim:new.type_block_offset]
        )
        v_old = self.torch.tensor(v_old, dtype=self.torch.float32)

        with self.torch.no_grad():
            old_logits, old_value = p_old(v_old.unsqueeze(0))
            new_logits, new_value = p_new(self.torch.as_tensor(v_new).unsqueeze(0))

        for head in range(len(old_logits)):
            shared = old_logits[head].shape[1]
            self.assertTrue(
                self.torch.allclose(old_logits[head][0], new_logits[head][0][:shared], atol=1e-5),
                f"head {head} drifted after the layout re-map",
            )
        self.assertTrue(self.torch.allclose(old_value, new_value, atol=1e-5))


class ChildProcessDecodingTests(unittest.TestCase):
    """A non-decodable byte must not erase the whole child log."""

    def test_non_locale_bytes_are_replaced_not_lost(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service import proc

        code, output = proc._run_managed_proc(
            "__decode_probe__",
            [sys.executable, "-c",
             "import sys; sys.stdout.buffer.write(b'trainer said: \\xff\\xfe broken\\n'); sys.exit(3)"],
            timeout=60,
        )
        self.assertEqual(code, 3)
        self.assertIn("trainer said", output, "the readable part of the log must survive")
        self.assertIn("\ufffd", output, "undecodable bytes become replacement chars, not a crash")


class ContainerPrefixHelperTests(unittest.TestCase):
    """``/app/`` stripping is shared by ``/infer`` and ``/ppo`` and must not mangle paths."""

    def setUp(self) -> None:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        from rwpp_service.paths import _strip_container_prefix

        self.strip = _strip_container_prefix

    def test_strips_the_container_root(self) -> None:
        self.assertEqual(self.strip("/app/runs/platform/job/checkpoint.pt"),
                         "runs/platform/job/checkpoint.pt")

    def test_leaves_windows_relative_paths_untouched(self) -> None:
        # Normalising backslashes unconditionally used to turn this into
        # "runs//platform//c.pt", which then failed every existence check.
        native = r"runs\platform\c.pt"
        self.assertEqual(self.strip(native), native)

    def test_leaves_posix_and_absolute_paths_untouched(self) -> None:
        for value in ("runs/platform/c.pt", "/home/user/c.pt", r"E:\weights\c.pt"):
            self.assertEqual(self.strip(value), value)

    def test_empty_values(self) -> None:
        self.assertEqual(self.strip(None), "")
        self.assertEqual(self.strip(""), "")


class WarmstartNoLongerHardCodesTheLegacyPathTests(unittest.TestCase):
    """The dashboard must let the host resolve the BC weight for the tank count."""

    def test_frontend_does_not_send_a_hard_coded_bc_path(self) -> None:
        source = (REPO / "web" / "frontend" / "js" / "ppoRun.js").read_text(encoding="utf-8")
        # A quoted literal is what actually got sent; prose in a comment is fine.
        for literal in ('"runs/bc_ironzero25/checkpoint.pt"', "'runs/bc_ironzero25/checkpoint.pt'"):
            self.assertNotIn(literal, source)
        self.assertIn("warmstart", source)

    def test_frontend_pins_physics_ms(self) -> None:
        source = (REPO / "web" / "frontend" / "js" / "ppoRun.js").read_text(encoding="utf-8")
        self.assertIn("physics_ms: 16", source)
        # ``frame_ms`` is only honest while the delta is pinned: with
        # ``physics_ms=0`` the bridge clamps 0 ms back to 1 and every advertised
        # multiplier in the form would be wrong.
        self.assertIn("fast_forward: fastForward", source)
        self.assertIn("lockstep: true", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
