# SPDX-License-Identifier: Apache-2.0
"""Launching and pacing the RWPP headless process.

This module owns the only place that spawns ``rwpp-server.bat``. It imports
``subprocess`` at module level on purpose: the tests patch
``ironzero.rwpp_env.subprocess.Popen``, and because ``subprocess`` is a single
module object shared by every importer, that patch replaces ``Popen`` on this
module's reference too.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .config import RwppGameConfig
from .errors import RwppEnvError

FRAME_MS_WARNED = False


class _ProcLike:
    """Minimal process handle (kept tiny so tests can inject a fake game)."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    def wait(self, timeout: float | None = None) -> int:
        return self._process.wait(timeout=timeout)

    def kill(self) -> None:
        try:
            self._process.kill()
        except OSError:
            pass


def effective_frame_ms(frame_ms: int) -> int:
    """Wall-clock pacing budget per simulated frame, clamped to a valid value.

    ``--frame-ms`` is also the frame *delta* handed to
    ``RWGameEngine.updateFrameInternal(1.0f, frameMs)``. A budget of 0 makes the
    engine advance with a zero delta, which changes damage/cooldown cadence and
    silently produces different battles (measured: the same policy and seed
    lost/timed out at 0 ms and won at 1 ms). 1 ms is therefore the fastest safe
    pacing, and every larger value only slows the wall clock down.
    """

    global FRAME_MS_WARNED
    value = int(frame_ms)
    if value >= 1:
        return value
    if not FRAME_MS_WARNED:
        FRAME_MS_WARNED = True
        print(
            "[rwpp] frame_ms<=0 is not a valid simulation pacing (the engine uses it as the "
            "frame delta); clamping to 1 ms. Pass --frame-ms 1 for the fastest correct run.",
            flush=True,
        )
    return 1


def launch_rwpp_process(game: RwppGameConfig, port: int) -> _ProcLike:
    """Start the pinned RWPP headless runtime with the IronZero bridge."""

    runtime = Path(game.runtime)
    java_home = Path(game.java_home)
    launcher = runtime / "bin" / "rwpp-server.bat"
    if not launcher.exists():
        raise RwppEnvError(f"RWPP runtime not found: {launcher}")
    game.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = game.log_dir / f"game-{port}-{int(time.time() * 1000) % 100000}.log"
    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    env.update(game.env_overrides)
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    command = [
        str(launcher),
        "--no-console",
        "--",
        "--bridge-port",
        str(port),
        "--ticks",
        str(int(game.episode_ticks)),
        "--map",
        game.map_relative,
        "--frame-ms",
        str(effective_frame_ms(game.frame_ms)),
    ]
    # Bridge options are opt-in so older bridge builds keep working unchanged.
    if game.win_mode:
        command.extend(["--win-mode", str(game.win_mode)])
    if game.unlock_ai:
        command.append("--unlock-ai")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(runtime),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        handle.close()
        raise RwppEnvError(f"failed to start rwpp-server: {exc}") from exc
    # The child keeps its own handle; the parent copy is closed once spawned so the
    # log is flushed by the game process and not by us.
    handle.close()
    return _ProcLike(process)


__all__ = [
    "FRAME_MS_WARNED",
    "_ProcLike",
    "effective_frame_ms",
    "launch_rwpp_process",
]
