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

# Upper bound on how many already-buffered messages one catch-up pass will
# consume. Generous (a whole episode is ~500 observations) but finite, so a
# misbehaving peer streaming forever cannot spin here.
_CATCH_UP_LIMIT = 10_000


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

    After the first observation arrives, any *further* messages that are already
    available are drained without blocking, and the freshest observation wins.
    That is not an optimisation -- it is what stops us from falling behind.

    The bridge is a free-running simulator: it advances on ``--frame-ms`` and
    emits an observation every ``DECISION_INTERVAL`` ticks without waiting for a
    client ``step``, while one decision here costs a policy forward pass. Late in
    a long episode the bridge is therefore *ahead* of us and its observations
    queue up in the socket (measured on a 4-env run: ~50 observations / 57 KB
    still unread when the episode ended).

    The backlog is not cosmetic. When the bridge decides the match it writes
    ``done`` and closes; our next ``sendall`` then hits a closed peer and the
    kernel answers with RST, which *discards the unread receive queue* --
    verdict included. That is exactly why the trainer booked engine victories as
    ``timeout`` (terminal = 0.0, win_rate = 0.0 over 40 updates, while the bridge
    logs for those same games showed 38 victories / 5 defeats).

    Draining here keeps the backlog bounded, so a verdict written just before
    the close is read on the same step. A ``done`` (or an observation carrying a
    ``result``) always beats a stale observation, so the terminal step can never
    overwrite a verdict with an older frame.
    """

    if env._reader is None:
        return None
    deadline = time.time() + timeout
    first = _read_next(env, deadline, wait=min(0.5, timeout))
    if first is None:
        return None
    if _verdict_of(first) is not None:
        return first
    latest = first
    # Catch up on everything that already arrived. Only complete lines are
    # consumed, and a partial line stays buffered for the next call, so a message
    # split across recv() boundaries is not lost.
    for _ in range(_CATCH_UP_LIMIT):
        reader = env._reader
        if reader is not None and reader.closed and b"\n" not in reader.buffer:
            break
        extra = _read_next(env, deadline, wait=0.0)
        if extra is None:
            break
        if _verdict_of(extra) is not None:
            return extra
        if extra.get("type") == "obs":
            latest = extra
    return latest


def _verdict_of(message: dict[str, Any] | None) -> str | None:
    """The decisive match verdict carried by ``message``, if any."""

    if not isinstance(message, dict):
        return None
    result = message.get("result")
    if not result:
        return None
    if message.get("type") in {"done", "obs"}:
        return str(result)
    return None


def _read_next(
    env: "RwppDuelEnv",
    deadline: float,
    wait: float = 0.25,
) -> dict[str, Any] | None:
    """Return the next protocol message, or ``None`` once ``deadline`` passes.

    ``wait`` bounds a single ``recv`` so catch-up calls return promptly instead
    of spending the whole step budget on a peer that has nothing more to say.
    """

    reader = env._reader
    if reader is None:
        return None
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        message = reader.read(max(0.0, min(wait, remaining)))
        if message is None:
            if reader.closed:
                return None
            if wait <= 0.0:
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


def drain_verdict(
    env: "RwppDuelEnv",
    timeout: float = 5.0,
    max_messages: int = 100_000,
) -> dict[str, Any] | None:
    """Read to EOF looking for the verdict the peer already sent, if any.

    Used when the *send* side failed (``ConnectionResetError`` after the bridge
    closed) but the receive stream still holds messages the bridge wrote before
    closing. The bridge is free-running, so it runs *ahead* of us and its
    observations queue up (measured: ~50 unread observations / 57 KB at the
    moment an episode ended, with the verdict another ~28 observations further
    on, i.e. still in flight in the kernel buffer).

    Therefore this drains until the peer closes or a verdict appears -- it does
    **not** stop at the first stale observation. Draining terminates because the
    peer is closed; the wall-clock budget and ``max_messages`` only bound a
    pathological case (peer alive but silent).

    Contract:
      * prefers ``done``; falls back to a final ``obs`` that carries a
        ``result`` (older bridge builds report the verdict on the observation);
      * returns ``None`` when no verdict exists, so a genuine mid-episode
        disconnect still becomes a ``timeout`` -- we did not see a verdict and
        must not invent one.
    """

    reader = env._reader
    if reader is None:
        return None
    deadline = time.time() + max(0.0, float(timeout))
    fallback: dict[str, Any] | None = None
    for _ in range(max_messages):
        if reader.closed and b"\n" not in reader.buffer:
            break
        if time.time() >= deadline:
            break
        message = reader.read(0.25)
        if message is None:
            continue
        kind = message.get("type")
        if kind == "receipt":
            # Deliberately NOT appended to ``env.receipts``: these belong to
            # decisions we are abandoning, and the caller reads
            # ``env.receipts[receipts_before:]`` to price THIS step's ``invalid``
            # penalty. Folding a whole backlog in here would spike ``invalid``
            # on the terminal step with refusals from unrelated decisions.
            continue
        if kind == "done":
            env.done = True
            return message
        if kind == "obs" and message.get("result"):
            # Keep draining: a ``done`` may follow and is preferred, but this
            # already qualifies as an authoritative verdict.
            fallback = message
    if fallback is not None:
        env.done = True
        return fallback
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


__all__ = ["await_observation", "drain_verdict", "shutdown_game", "start_game"]
