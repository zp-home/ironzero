# SPDX-License-Identifier: Apache-2.0
"""Concurrency caps, async restart and the physics/pacing split.

These guard three separate regressions:

1. the concurrency ceiling is agreed on by the host service, the pydantic
   request models and the dashboard's ``<select>`` -- a mismatch either rejects
   legitimate requests or offers options the service will 400 on;
2. ``VecRwppEnv`` must not block sampling on a slot's JVM boot, and a slot that
   is booting must report a zero-reward self-loop instead of a fabricated
   observation;
3. ``--physics-ms`` must reach the game process only when explicitly requested,
   so existing training runs keep their exact behaviour.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# The host service and the web backend live outside ``src`` (pytest's pythonpath
# is only ``src``), so they are imported by adding their roots explicitly.
for extra in (REPO / "scripts", REPO / "web" / "backend"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


# ---------------------------------------------------------------------------
# 1. concurrency ceiling, everywhere at once
# ---------------------------------------------------------------------------
def test_host_service_default_max_parallel_is_ten() -> None:
    """The host service owns the real ceiling; the dashboard reads it from /health."""

    from rwpp_service import state

    assert state.MAX_PARALLEL == 10, (
        "MAX_PARALLEL is the single authority for concurrent game processes; "
        "the dashboard, the request models and the port pool all follow it"
    )


def test_server_cli_default_follows_state() -> None:
    """``--max-parallel`` must default to ``state.MAX_PARALLEL``, not a stale literal."""

    source = (REPO / "scripts" / "rwpp_service" / "server.py").read_text(encoding="utf-8")
    assert "default=state.MAX_PARALLEL" in source, (
        "the argparse default regressed to a hard-coded number, so bumping "
        "state.MAX_PARALLEL would silently stop affecting the CLI"
    )
    assert "5130 + slot" in source, "the port pool must still be filled from the ceiling"


def test_backend_models_allow_the_full_range() -> None:
    """A request the host service accepts must not be rejected by pydantic.

    The models are loaded from the file directly: importing the app package pulls
    in the whole FastAPI router tree (and its relative imports), which is not
    what this contract is about.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ironzero_request_models", REPO / "web" / "backend" / "app" / "models.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RwppRecordRequest(concurrency=10).concurrency == 10
    assert module.RwppPpoRequest(envs=10).envs == 10
    with pytest.raises(Exception):
        module.RwppRecordRequest(concurrency=99)
    with pytest.raises(Exception):
        module.RwppPpoRequest(envs=99)


def test_ppo_bounds_accept_ten_envs() -> None:
    from rwpp_service.launch import ppo_bounds_error

    base = {
        "frame_ms": 2,
        "vision_radius": 260.0,
        "updates": 10,
        "rollout_steps": 64,
        "episode_ticks": 3600,
    }
    assert ppo_bounds_error({**base, "envs": 10}) is None
    assert ppo_bounds_error({**base, "envs": 16}) is None
    assert ppo_bounds_error({**base, "envs": 17}) is not None


def test_dashboard_offers_every_slot_up_to_the_ceiling() -> None:
    """The concurrency selects must cover 1..10 or the extra slots are unreachable."""

    html = (REPO / "web" / "frontend" / "index.html").read_text(encoding="utf-8")
    for select_id in ("fe-concurrency", "fe-train-concurrency"):
        block = re.search(
            rf'<select id="{select_id}">(.*?)</select>', html, re.S
        )
        assert block, f"{select_id} select not found"
        values = [int(v) for v in re.findall(r'<option value="(\d+)"', block.group(1))]
        assert values == list(range(1, 11)), (
            f"{select_id} offers {values}; a 10-slot service needs 1..10"
        )


def test_ppo_envs_input_matches_the_service_ceiling() -> None:
    html = (REPO / "web" / "frontend" / "index.html").read_text(encoding="utf-8")
    field = re.search(r'<input id="fe-ppo-envs"[^>]*max="(\d+)"', html)
    assert field, "fe-ppo-envs input not found"
    assert int(field.group(1)) == 10


def test_frontend_clamps_match_the_ceiling() -> None:
    """A clamp still at the old ceiling would silently truncate a legal choice."""

    js = REPO / "web" / "frontend" / "js"
    infer = (js / "feInferRun.js").read_text(encoding="utf-8")
    train = (js / "feTrain.js").read_text(encoding="utf-8")
    ppo = (js / "ppoRun.js").read_text(encoding="utf-8")
    assert re.search(r"els\.feConcurrency\.value, 1, 10,", infer), infer
    assert re.search(r"els\.feTrainConcurrency\.value, 1, 10,", train), train
    assert re.search(r"els\.fePpoEnvs\.value, 1, 10,", ppo), ppo


def test_frontend_max_parallel_fallbacks_are_not_stale() -> None:
    """Before /health answers, the UI must not assume a 4-slot service."""

    js = REPO / "web" / "frontend" / "js"
    for name in ("rwpp.js", "feInferRun.js", "feInferUi.js"):
        text = (js / name).read_text(encoding="utf-8")
        assert "maxParallel || 4" not in text, f"{name} still falls back to 4 slots"
        assert re.search(r"max_parallel\) !== null \? num\(payload\.max_parallel, 4\)", text) is None


# ---------------------------------------------------------------------------
# 2. VecRwppEnv: async restart must not block sampling
# ---------------------------------------------------------------------------
class _FakeEnv:
    """Minimal RwppDuelEnv stand-in: no socket, no process, controllable timing."""

    def __init__(self, index: int, boot_delay: float, end_after: int) -> None:
        self.index = index
        self.boot_delay = boot_delay
        self.end_after = end_after
        self.steps = 0
        self.resets = 0
        self.closed = False
        from ironzero.micro_config import MicroDuelConfig

        self.config = MicroDuelConfig()
        self.episode_id = f"fake-{index}"

    def reset(self, *, seed: int | None = None):
        import time

        self.resets += 1
        if self.boot_delay:
            time.sleep(self.boot_delay)
        self.steps = 0
        obs = np.zeros(self.config.flat_obs_dim, dtype=np.float32)
        obs[self.index % self.config.flat_obs_dim] = 1.0
        masks = {
            "kind": np.ones(self.config.action_nvec[0], dtype=bool),
            "group": np.ones(self.config.action_nvec[1], dtype=bool),
            "x": np.ones(self.config.position_grid, dtype=bool),
            "y": np.ones(self.config.position_grid, dtype=bool),
            "target": np.ones(self.config.target_slots, dtype=bool),
        }
        return obs, {"masks": masks, "port": 5000 + self.index}

    def step(self, action):
        self.steps += 1
        obs = np.zeros(self.config.flat_obs_dim, dtype=np.float32)
        obs[0] = float(self.steps)
        masks = {
            "kind": np.ones(self.config.action_nvec[0], dtype=bool),
            "group": np.ones(self.config.action_nvec[1], dtype=bool),
            "x": np.ones(self.config.position_grid, dtype=bool),
            "y": np.ones(self.config.position_grid, dtype=bool),
            "target": np.ones(self.config.target_slots, dtype=bool),
        }
        done = self.steps >= self.end_after
        info = {
            "masks": masks,
            "result": "timeout" if done else None,
            "tick": self.steps * 10,
            "port": 5000 + self.index,
            "reward_components": {"potential": 0.0},
        }
        return obs, 0.0, done, False, info

    def close(self) -> None:
        self.closed = True


def _make_vec(envs, *, async_reset: bool):
    from ironzero.training.vec_rwpp import VecRwppEnv

    vec = VecRwppEnv(
        len(envs),
        game=None,
        micro_config=envs[0].config,
        persistent=True,
        async_reset=async_reset,
    )
    vec.envs = envs
    return vec


def test_async_reset_returns_terminal_without_waiting_for_the_boot() -> None:
    """A finished slot must not stall the batch on its multi-second JVM boot."""

    import time

    envs = [_FakeEnv(0, boot_delay=0.0, end_after=1), _FakeEnv(1, boot_delay=0.0, end_after=99)]
    vec = _make_vec(envs, async_reset=True)
    try:
        vec.reset()
        actions = np.zeros((2, len(vec.config.action_nvec)), dtype=np.int64)
        # Make slot 0's relaunch slow *before* the step that finishes it.
        envs[0].boot_delay = 5.0
        started = time.time()
        batch = vec.step(actions)
        elapsed = time.time() - started
        assert elapsed < 1.0, (
            f"step() blocked {elapsed:.2f}s on a background restart; the whole "
            f"batch waited for one slot's JVM boot"
        )
        # The terminal transition is reported immediately...
        assert bool(batch.terminated[0]) or bool(batch.truncated[0])
        assert batch.infos[0]["episode"] is not None, "the finished episode must be recorded"
        # ...and slot 1 was still stepped in the same call.
        assert envs[1].steps == 1
    finally:
        vec.close()


def test_booting_slot_reports_zero_reward_self_loop() -> None:
    """A relaunching slot must not fabricate a jump to an unrelated observation."""

    envs = [_FakeEnv(0, boot_delay=0.0, end_after=1), _FakeEnv(1, boot_delay=0.0, end_after=99)]
    vec = _make_vec(envs, async_reset=True)
    try:
        vec.reset()
        actions = np.zeros((2, len(vec.config.action_nvec)), dtype=np.int64)
        envs[0].boot_delay = 5.0
        vec.step(actions)  # slot 0 finishes, relaunch starts (slow)
        before = vec.observations[0].copy()
        batch = vec.step(actions)
        # While booting: reported as booting, no reward, no terminal.
        assert batch.infos[0].get("booting") is True
        assert batch.rewards[0] == 0.0
        assert not bool(batch.terminated[0])
        assert not bool(batch.truncated[0])
        # Its observation is left untouched (self-loop), not zeroed or random.
        np.testing.assert_array_equal(vec.observations[0], before)
        # The healthy slot kept advancing.
        assert envs[1].steps == 2
    finally:
        vec.close()


def test_sync_reset_still_awaits_the_relaunch() -> None:
    """The historical behaviour stays available and unchanged."""

    envs = [_FakeEnv(0, boot_delay=0.0, end_after=1), _FakeEnv(1, boot_delay=0.0, end_after=99)]
    vec = _make_vec(envs, async_reset=False)
    try:
        vec.reset()
        actions = np.zeros((2, len(vec.config.action_nvec)), dtype=np.int64)
        before = envs[0].resets
        batch = vec.step(actions)
        assert envs[0].resets == before + 1, "sync mode must reset inline"
        assert batch.infos[0].get("reset") is not None
    finally:
        vec.close()


def test_completed_episode_is_recorded_exactly_once() -> None:
    """Async restarts must not duplicate or drop a finished episode."""

    envs = [_FakeEnv(0, boot_delay=0.0, end_after=2), _FakeEnv(1, boot_delay=0.0, end_after=2)]
    vec = _make_vec(envs, async_reset=True)
    try:
        vec.reset()
        actions = np.zeros((2, len(vec.config.action_nvec)), dtype=np.int64)
        for _ in range(2):
            vec.step(actions)
        completed = vec.drain_completed()
        assert len(completed) == 2
        for episode in completed:
            assert episode["decisions"] == 2
            assert episode["result"] == "timeout"
        assert vec.drain_completed() == []
    finally:
        vec.close()


def test_restart_failure_does_not_kill_the_batch() -> None:
    """A wedged JVM relaunch must leave the slot inactive, not abort training."""

    envs = [_FakeEnv(0, boot_delay=0.0, end_after=1), _FakeEnv(1, boot_delay=0.0, end_after=99)]
    vec = _make_vec(envs, async_reset=True)
    original_reset = envs[0].reset

    def exploding_reset(*, seed=None):
        raise RuntimeError("JVM refused to boot")

    try:
        vec.reset()
        actions = np.zeros((2, len(vec.config.action_nvec)), dtype=np.int64)
        envs[0].reset = exploding_reset  # type: ignore[method-assign]
        batch = vec.step(actions)  # must not raise
        assert bool(batch.terminated[0]) or bool(batch.truncated[0])
        assert batch.infos[0]["episode"] is not None, "the finished episode is still recorded"
        assert envs[1].steps == 1, "the healthy slot keeps sampling"
        # The failure surfaces on the next boundary as "still booting", not a raise.
        batch2 = vec.step(actions)
        assert batch2.infos[0].get("booting") is True
        assert envs[1].steps == 2
    finally:
        envs[0].reset = original_reset  # type: ignore[method-assign]
        vec.close()


# ---------------------------------------------------------------------------
# 3. physics vs. pacing
# ---------------------------------------------------------------------------
def test_physics_ms_defaults_to_off() -> None:
    """Existing runs must keep the historical "frame_ms is the physics delta"."""

    from ironzero.rwpp.config import RwppGameConfig

    assert RwppGameConfig().physics_ms == 0


def test_physics_ms_is_passed_to_the_game_only_when_requested() -> None:
    from ironzero.rwpp.process import _build_command_for_tests

    plain = _build_command_for_tests(physics_ms=0, frame_ms=2)
    assert "--physics-ms" not in plain, "a default run must not change the command line"

    pinned = _build_command_for_tests(physics_ms=16, frame_ms=2)
    assert "--physics-ms" in pinned
    assert pinned[pinned.index("--physics-ms") + 1] == "16"
    # The pacing budget is still whatever the caller asked for.
    assert pinned[pinned.index("--frame-ms") + 1] == "2"


def test_train_cli_exposes_physics_ms() -> None:
    from ironzero.training.train_rwpp import build_parser

    args = build_parser().parse_args(["--physics-ms", "16"])
    assert args.physics_ms == 16
    assert build_parser().parse_args([]).physics_ms == 0


# ---------------------------------------------------------------------------
# 4. deployment drift: the bridge jar must speak the flags Python sends
# ---------------------------------------------------------------------------
def _class_strings(jar: Path) -> str:
    """ASCII text of ``IronZeroBridge.class`` (enough to spot flag literals)."""

    import zipfile

    with zipfile.ZipFile(jar) as archive:
        raw = archive.read("ironzero/IronZeroBridge.class")
    return "".join(chr(b) if 32 <= b < 127 else " " for b in raw)


def test_deployed_bridge_jar_knows_the_flags_python_sends() -> None:
    """A stale ``ironzero-bridge.jar`` silently reverts the bridge contract.

    ``RwppGameConfig`` can be edited to send a new flag while the runtime still
    holds an older jar that ignores it, which looks exactly like "the change did
    nothing". Whenever a runtime is present, its jar must contain every flag the
    command builder can emit. Runtimes are absent on CI, so this skips there.
    """

    from ironzero.rwpp.config import DEFAULT_RUNTIME, RwppGameConfig
    from ironzero.rwpp.process import _bridge_command, _build_command_for_tests

    default_line = _build_command_for_tests()
    # ``--no-console`` sits before the ``--`` separator and belongs to the
    # rwpp-server launcher, not to the bridge, so only bridge args are checked.
    bridge_args = default_line[default_line.index("--") + 1 :]
    always = [token for token in bridge_args if token.startswith("--")]
    assert "--physics-ms" not in bridge_args, "the flag must stay opt-in"

    # Every optional bridge flag the builder can emit, exercised through a config
    # that actually turns each one on.
    optional: set[str] = set()
    for config in (
        RwppGameConfig(physics_ms=16),
        RwppGameConfig(win_mode="annihilation"),
        RwppGameConfig(unlock_ai=True),
    ):
        line = _bridge_command(Path("rwpp-server.bat"), 5130, config)
        line = line[line.index("--") + 1 :]  # drop launcher args, as above
        emitted = {token for token in line if token.startswith("--")}
        new = emitted - set(always) - optional
        assert new, f"{config} emitted no new flag; the helper is not exercised"
        optional |= new
    assert "--physics-ms" in optional, "the helper no longer emits --physics-ms"

    candidates = [
        REPO / ".runtime" / "rwpp-runtime-1.15",
        Path(DEFAULT_RUNTIME),
    ]
    checked = 0
    for runtime_root in dict.fromkeys(candidates):
        jar = runtime_root / "extensions" / "ironzero-bridge.jar"
        if not jar.is_file():
            continue
        blob = _class_strings(jar)
        for flag in sorted({*always, *optional}):
            assert flag in blob, (
                f"{jar} has no {flag}: the deployed bridge jar predates the current "
                "command line. Rebuild the bridge extension and re-sync the jar into "
                "every runtime in use (scripts/rwpp_service/README.md 1.2)."
            )
        checked += 1
    if not checked:
        pytest.skip("no local RWPP runtime to inspect")

