# SPDX-License-Identifier: Apache-2.0
"""Bridge socket lifecycle for :class:`ironzero.rwpp.env.RwppDuelEnv`.

Connecting to the bridge port, draining the message stream until the next
``obs``/``done``, and shutting the game down are mechanical concerns that sit
below the RL loop. They live here as free functions taking the environment as
``env``; :class:`RwppDuelEnv` keeps the historical private method names
(``_start_game`` / ``_await_observation`` / ``_shutdown_game``) as thin
delegates so nothing outside this package needs to change.
"""

from __future__ import annotations

import socket
import time
from typing import TYPE_CHECKING, Any

from .errors import RwppEnvError
from .transport import _LineReader

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .env import RwppDuelEnv


def start_game(env: "RwppDuelEnv") -> None:
    """Launch the game process and connect to its bridge port."""

    env._process = env._launcher(env.game, env.port)
    deadline = time.time() + env.game.start_timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", env.port), timeout=2.0)
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
            continue
        sock.settimeout(5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        env._socket = sock
        env._reader = _LineReader(sock)
        return
    env._shutdown_game()
    raise RwppEnvError(
        f"bridge port {env.port} never opened (last error: {last_error})"
    )


def await_observation(env: "RwppDuelEnv", timeout: float) -> dict[str, Any] | None:
    """Read until the next ``obs`` (or terminal ``done``); ``None`` on timeout.

    ``receipt`` messages are accumulated on the environment (they are per-step
    reward inputs), ``hello`` is skipped.
    """

    if env._reader is None:
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        message = env._reader.read(max(0.5, min(remaining, 10.0)))
        if message is None:
            if env._reader.closed:
                # Peer closed: fail fast instead of burning the whole timeout.
                return None
            continue
        kind = message.get("type")
        if kind == "obs":
            return message
        if kind == "receipt":
            env.receipts.append(dict(message))
            continue
        if kind == "done":
            env.done = True
            return message
        if kind == "hello":
            continue
    return None


def shutdown_game(env: "RwppDuelEnv") -> None:
    """Best-effort orderly shutdown: tell the bridge, close, kill if stuck."""

    if env._socket is not None:
        try:
            env._socket.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass
        try:
            env._socket.close()
        except OSError:
            pass
        env._socket = None
        env._reader = None
    if env._process is not None:
        try:
            env._process.wait(timeout=15.0)
        except Exception:  # noqa: BLE001 - timeout/OSError both mean "still alive"
            env._process.kill()
        env._process = None
    if env._log_handle is not None:
        try:
            env._log_handle.close()
        except OSError:
            pass
        env._log_handle = None


__all__ = ["await_observation", "shutdown_game", "start_game"]
