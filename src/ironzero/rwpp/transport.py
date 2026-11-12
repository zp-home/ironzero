# SPDX-License-Identifier: Apache-2.0
"""Newline-delimited JSON transport for the ``ironzero/1`` bridge socket."""

from __future__ import annotations

import json
import socket
import time
from typing import Any


class _LineReader:
    """Newline-delimited JSON reader with an internal buffer.

    ``closed`` distinguishes "no data for now" (a normal timeout, retry) from
    "the peer closed the socket" (fatal for this episode): without it a dead
    bridge makes the caller spin until its own timeout, which used to leave a
    60 s gap and race the environment reset.
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.closed = False

    def read(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            self.sock.settimeout(min(remaining, 5.0))
            try:
                chunk = self.sock.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError:
                self.closed = True
                return None
            if not chunk:
                self.closed = True
                return None
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except ValueError:
            return None


__all__ = ["_LineReader"]
