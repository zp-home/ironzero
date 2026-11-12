# SPDX-License-Identifier: Apache-2.0
"""NDJSON over loopback TCP transport for the ``ironzero/1`` bridge.

The transport is deliberately boring: one JSON object per line, UTF-8, no
framing magic. That keeps packet captures human-readable while the protocol
module stays the single source of truth for message shapes.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Any, Callable

from .protocol import (
    Envelope,
    ErrorCode,
    ErrorMessage,
    HelloMessage,
    Message,
    canonical_json,
    dump_message,
    parse_message,
)

MAX_MESSAGE_BYTES = 1 << 20  # 1 MiB, matches docs/协议v1说明.md


class BridgeError(RuntimeError):
    """Transport-level bridge failure."""


class BridgeTimeout(BridgeError):
    """No message arrived before the deadline."""


class BridgeClosed(BridgeError):
    """Peer closed the connection."""


def new_message_id(prefix: str = "m") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class NdjsonConnection:
    """Bidirectional NDJSON connection over a connected socket."""

    def __init__(self, sock: socket.socket, max_message_bytes: int = MAX_MESSAGE_BYTES) -> None:
        self.sock = sock
        self.max_message_bytes = max_message_bytes
        self._buffer = b""
        self._closed = False

    # -- low level ------------------------------------------------------
    def _recv_chunk(self, timeout: float | None) -> bytes:
        if self._closed:
            raise BridgeClosed("connection is closed")
        if timeout is not None:
            self.sock.settimeout(max(0.001, timeout))
        try:
            chunk = self.sock.recv(65536)
        except socket.timeout as exc:  # pragma: no cover - timing dependent
            raise BridgeTimeout("timed out waiting for bridge message") from exc
        except OSError as exc:
            raise BridgeClosed(f"socket receive failed: {exc}") from exc
        if not chunk:
            raise BridgeClosed("peer closed the connection")
        return chunk

    def read_line(self, timeout: float | None = None) -> bytes:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            index = self._buffer.find(b"\n")
            if index >= 0:
                line, self._buffer = self._buffer[:index], self._buffer[index + 1 :]
                if len(line) > self.max_message_bytes:
                    raise BridgeError("bridge message exceeds size limit")
                return line
            if len(self._buffer) > self.max_message_bytes:
                raise BridgeError("bridge message exceeds size limit")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise BridgeTimeout("timed out waiting for bridge message")
            self._buffer += self._recv_chunk(remaining)

    # -- message level --------------------------------------------------
    def send_message(self, message: Message) -> None:
        if self._closed:
            raise BridgeClosed("connection is closed")
        payload = canonical_json(dump_message(message)).encode("utf-8") + b"\n"
        if len(payload) > self.max_message_bytes:
            raise BridgeError("bridge message exceeds size limit")
        try:
            self.sock.sendall(payload)
        except OSError as exc:
            raise BridgeClosed(f"socket send failed: {exc}") from exc

    def recv_message(self, timeout: float | None = None) -> Envelope:
        line = self.read_line(timeout)
        if not line.strip():
            raise BridgeError("empty bridge line")
        try:
            return parse_message(line)
        except Exception as exc:
            raise BridgeError(f"invalid bridge message: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            try:
                self.sock.close()
            except OSError:
                pass


class BridgeClient:
    """Blocking client used by the Python environment / evaluator."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5123,
        connect_timeout: float = 5.0,
        recv_timeout: float = 15.0,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self.connection: NdjsonConnection | None = None
        self.server_hello: HelloMessage | None = None

    # -- lifecycle ------------------------------------------------------
    def connect(self) -> "BridgeClient":
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection = NdjsonConnection(sock)
        return self

    def handshake(
        self,
        *,
        adapter_version: str = "0.1.0",
        game_version: str = "ironzero-mock-1.0",
        capabilities: list[str] | None = None,
        timeout: float | None = None,
    ) -> HelloMessage:
        connection = self._require_connection()
        hello = HelloMessage(
            message_id=new_message_id(),
            adapter_version=adapter_version,
            game_version=game_version,
            capabilities=capabilities or ["ndjson", "action_mask", "lockstep"],
        )
        connection.send_message(hello)
        response = connection.recv_message(timeout if timeout is not None else self.recv_timeout)
        if not isinstance(response, HelloMessage):
            raise BridgeError(f"expected hello, got {type(response).__name__}")
        self.server_hello = response
        return response

    def send(self, message: Message) -> None:
        self._require_connection().send_message(message)

    def recv(self, timeout: float | None = None) -> Envelope:
        return self._require_connection().recv_message(timeout if timeout is not None else self.recv_timeout)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _require_connection(self) -> NdjsonConnection:
        if self.connection is None:
            raise BridgeError("bridge client is not connected")
        return self.connection

    def __enter__(self) -> "BridgeClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


MessageHandler = Callable[["BridgeServer", NdjsonConnection, Message], None]


class BridgeServer:
    """Single-client NDJSON server hosting a game-side adapter."""

    def __init__(
        self,
        handler: MessageHandler,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        adapter_version: str = "0.1.0",
        game_version: str = "ironzero-mock-1.0",
        capabilities: list[str] | None = None,
        recv_timeout: float | None = None,
    ) -> None:
        self.handler = handler
        self.host = host
        self.port = port
        self.adapter_version = adapter_version
        self.game_version = game_version
        self.capabilities = capabilities or ["ndjson", "action_mask", "lockstep"]
        self.recv_timeout = recv_timeout
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._connection: NdjsonConnection | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self.connections_served = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> "BridgeServer":
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(1)
        listener.settimeout(0.2)
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="ironzero-bridge", daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                sock, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:  # pragma: no cover - shutdown race
                if not self._stop.is_set():
                    self._error = exc
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection = NdjsonConnection(sock)
            self._connection = connection
            self.connections_served += 1
            try:
                self._session(connection)
            except BridgeError:
                pass
            except BaseException as exc:  # pragma: no cover - defensive
                self._error = exc
            finally:
                connection.close()
                self._connection = None

    def _session(self, connection: NdjsonConnection) -> None:
        while not self._stop.is_set():
            try:
                message = connection.recv_message(self.recv_timeout)
            except BridgeClosed:
                return
            if isinstance(message, HelloMessage):
                connection.send_message(
                    HelloMessage(
                        message_id=new_message_id(),
                        adapter_version=self.adapter_version,
                        game_version=self.game_version,
                        capabilities=self.capabilities,
                    )
                )
                continue
            try:
                self.handler(self, connection, message)
            except ValueError as exc:
                self._send_error(connection, ErrorCode.INVALID_MESSAGE, str(exc))

    def _send_error(
        self,
        connection: NdjsonConnection,
        code: ErrorCode,
        message: str,
        *,
        episode_id: str | None = None,
    ) -> None:
        connection.send_message(
            ErrorMessage(
                message_id=new_message_id("err"),
                episode_id=episode_id,
                error_code=code,
                message=message[:2048],
                recoverable=True,
            )
        )

    # -- outbound helpers ----------------------------------------------
    @property
    def connection(self) -> NdjsonConnection | None:
        return self._connection

    def send(self, message: Message) -> None:
        if self._connection is None:
            raise BridgeClosed("no active bridge connection")
        self._connection.send_message(message)

    def stop(self) -> None:
        self._stop.set()
        if self._connection is not None:
            self._connection.close()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    @property
    def error(self) -> BaseException | None:
        return self._error

    def __enter__(self) -> "BridgeServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
