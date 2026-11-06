# SPDX-License-Identifier: Apache-2.0
"""Client-side adapter: drive the environment through the TCP bridge.

This is the code path that will be reused verbatim against the real RWPP
extension; only the server side changes. It speaks ``ironzero/1`` and returns
the same raw state dict shape as the in-process mock adapter, so observation
encoding and training code stay identical.
"""

from __future__ import annotations

from typing import Any, Iterable

from .bridge import BridgeClient, BridgeClosed, BridgeError, BridgeTimeout, new_message_id
from .mock_bridge import observation_message_to_state
from .protocol import (
    DoneMessage,
    ErrorMessage,
    HeartbeatMessage,
    ObservationMessage,
    ReceiptMessage,
    ResetRequest,
    StepRequest,
    parse_message,
)


class TcpDuelAdapter:
    """``reset``/``step`` adapter speaking NDJSON to a game-side server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        client: BridgeClient | None = None,
        connect_timeout: float = 10.0,
        recv_timeout: float = 30.0,
    ) -> None:
        self.client = client or BridgeClient(
            host=host, port=port, connect_timeout=connect_timeout, recv_timeout=recv_timeout
        )
        self.last_tick = 0
        self.last_decision_id: str | None = None
        self.episode_id: str | None = None
        self.done = False
        self.last_result: str | None = None
        self.receipts: list[dict[str, Any]] = []
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self, **handshake_kwargs: Any) -> "TcpDuelAdapter":
        if not self._connected:
            self.client.connect()
            self.client.handshake(**handshake_kwargs)
            self._connected = True
        return self

    def close(self) -> None:
        self.client.close()
        self._connected = False

    def __enter__(self) -> "TcpDuelAdapter":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Adapter API (matches MicroDuelAdapter)
    # ------------------------------------------------------------------
    def reset(self, *, episode_id: str, seed: int, scenario: str = "micro_duel") -> dict[str, Any]:
        self.connect()
        self.episode_id = episode_id
        self.done = False
        self.last_result = None
        self.receipts = []
        self.client.send(
            ResetRequest(
                message_id=new_message_id(),
                episode_id=episode_id,
                seed=seed,
                scenario=scenario,
                map_id="micro_duel_fixed",
            )
        )
        state = self._await_observation()
        self.last_tick = int(state["tick"])
        return state

    def step(self, actions: Iterable[dict[str, Any]], *, team: int = 0) -> dict[str, Any]:
        if self.episode_id is None:
            raise BridgeError("reset() must be called before step()")
        decision_id = new_message_id("d")
        self.last_decision_id = decision_id
        self.client.send(
            StepRequest(
                message_id=new_message_id(),
                episode_id=self.episode_id,
                tick=self.last_tick,
                decision_id=decision_id,
                actions=list(actions),
            )
        )
        self.receipts = []
        state = self._await_observation()
        self.last_tick = int(state["tick"])
        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _await_observation(self) -> dict[str, Any]:
        while True:
            try:
                message = self.client.recv()
            except (BridgeTimeout, BridgeClosed) as exc:
                raise BridgeError(f"bridge closed before observation: {exc}") from exc
            if isinstance(message, ObservationMessage):
                state = observation_message_to_state(message)
                self.done = bool(state.get("terminated") or state.get("truncated"))
                self.last_result = state.get("result")
                self.receipts = list(state.get("receipts", []))
                return state
            if isinstance(message, ReceiptMessage):
                self.receipts.append(
                    message.model_dump(exclude_none=True, mode="json")
                )
                continue
            if isinstance(message, DoneMessage):
                self.done = True
                self.last_result = str(message.result)
                continue
            if isinstance(message, ErrorMessage):
                raise BridgeError(f"{message.error_code}: {message.message}")
            if isinstance(message, HeartbeatMessage):
                continue
            raise BridgeError(f"unexpected message while waiting for obs: {message.type}")


__all__ = ["TcpDuelAdapter", "observation_message_to_state", "parse_message"]
