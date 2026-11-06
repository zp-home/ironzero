# SPDX-License-Identifier: Apache-2.0
"""TCP bridge adapter that exposes the deterministic micro-duel engine.

The env can talk to this adapter over loopback exactly the way it will talk to
the future RWPP extension, which keeps the transport code exercised long
before the real game runtime is wired in.
"""

from __future__ import annotations

from typing import Any

from .bridge import (
    BridgeClosed,
    BridgeError,
    BridgeServer,
    BridgeTimeout,
    new_message_id,
)
from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .mock_engine import ACTION_KINDS, MicroDuelAdapter
from .observations import build_masks, masks_as_lists
from .protocol import (
    DoneMessage,
    ErrorCode,
    ErrorMessage,
    HeartbeatMessage,
    HeartbeatStatus,
    ObservationMessage,
    ReceiptMessage,
    ResetRequest,
    SCHEMA_VERSION,
    ShutdownMessage,
    StepRequest,
    canonical_json,
)

ADAPTER_VERSION = "0.1.0-mock"
GAME_VERSION = "ironzero-mock-1.0"


def state_to_observation_message(
    state: dict[str, Any],
    *,
    episode_id: str,
    decision_id: str,
    config: MicroDuelConfig = DEFAULT_CONFIG,
    message_id: str | None = None,
    info_extra: dict[str, Any] | None = None,
) -> ObservationMessage:
    """Convert a raw adapter state into a validated ``obs`` message."""

    masks = build_masks(state, config)
    info = {
        "result": state.get("result"),
        "target_ids": list(state.get("target_ids", [])),
        "receipts": state.get("receipts", []),
        "eval": state.get("eval"),
    }
    if info_extra:
        info.update(info_extra)
    return ObservationMessage(
        message_id=message_id or new_message_id("obs"),
        episode_id=episode_id,
        tick=int(state.get("tick", 0)),
        decision_id=decision_id,
        game_time=float(state.get("game_time", 0.0)),
        controlled_team=int(state.get("controlled_team", 0)),
        visibility_revision=int(state.get("tick", 0)),
        observation_schema_version=SCHEMA_VERSION,
        global_state=state.get("global_state", {}),
        own_entities=state.get("own_entities", []),
        visible_enemy_entities=state.get("visible_enemy_entities", []),
        remembered_enemies=state.get("remembered_enemies", []),
        local_maps=state.get("local_maps", {}),
        masks=masks_as_lists(masks, config),
        reward=float(state.get("reward", 0.0)),
        terminated=bool(state.get("terminated", False)),
        truncated=bool(state.get("truncated", False)),
        info=info,
    )


def observation_message_to_state(message: ObservationMessage) -> dict[str, Any]:
    """Convert an ``obs`` message back into the raw dict the encoder expects."""

    info = message.info or {}
    return {
        "tick": message.tick,
        "game_time": message.game_time,
        "controlled_team": message.controlled_team,
        "global_state": message.global_state.model_dump(exclude_none=True),
        "own_entities": [entity.model_dump(exclude_none=True) for entity in message.own_entities],
        "visible_enemy_entities": [
            entity.model_dump(exclude_none=True) for entity in message.visible_enemy_entities
        ],
        "remembered_enemies": [
            entity.model_dump(exclude_none=True) for entity in message.remembered_enemies
        ],
        "local_maps": message.local_maps,
        "target_ids": list(info.get("target_ids", [])),
        "reward": message.reward,
        "terminated": message.terminated,
        "truncated": message.truncated,
        "result": info.get("result"),
        "receipts": info.get("receipts", []),
    }


class MockDuelBridge(BridgeServer):
    """Protocol-speaking server wrapper around :class:`MicroDuelAdapter`."""

    def __init__(
        self,
        *,
        config: MicroDuelConfig | None = None,
        opponent: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        recv_timeout: float | None = 30.0,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.adapter = MicroDuelAdapter(config=self.config, opponent=opponent)
        self.last_tick = 0
        self.active_episode: str | None = None
        super().__init__(
            self._handle,
            host=host,
            port=port,
            adapter_version=ADAPTER_VERSION,
            game_version=GAME_VERSION,
            capabilities=["ndjson", "action_mask", "lockstep", "deterministic"],
            recv_timeout=recv_timeout,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def _handle(self, server: BridgeServer, connection, message) -> None:
        if isinstance(message, ResetRequest):
            self._on_reset(connection, message)
        elif isinstance(message, StepRequest):
            self._on_step(connection, message)
        elif isinstance(message, HeartbeatMessage):
            connection.send_message(
                HeartbeatMessage(
                    message_id=new_message_id("hb"),
                    status=HeartbeatStatus.RUNNING,
                    last_tick=self.last_tick,
                )
            )
        elif isinstance(message, ShutdownMessage):
            connection.send_message(
                ShutdownMessage(
                    message_id=new_message_id("shutdown"),
                    reason="mock bridge shutting down",
                )
            )
            self._stop.set()
        elif isinstance(message, ErrorMessage):
            return
        else:
            connection.send_message(
                ErrorMessage(
                    message_id=new_message_id("err"),
                    episode_id=getattr(message, "episode_id", None),
                    error_code=ErrorCode.INVALID_MESSAGE,
                    message=f"unsupported message type: {message.type}",
                    recoverable=True,
                )
            )

    def _on_reset(self, connection, message: ResetRequest) -> None:
        if message.scenario != "micro_duel":
            connection.send_message(
                ErrorMessage(
                    message_id=new_message_id("err"),
                    episode_id=message.episode_id,
                    error_code=ErrorCode.INVALID_MESSAGE,
                    message=f"unsupported scenario: {message.scenario}",
                    recoverable=True,
                )
            )
            return
        state = self.adapter.reset(
            episode_id=message.episode_id,
            seed=message.seed,
            scenario=message.scenario,
        )
        self.active_episode = message.episode_id
        decision_id = new_message_id("d")
        self.last_tick = int(state["tick"])
        state["decision_id"] = decision_id
        connection.send_message(
            state_to_observation_message(
                state,
                episode_id=message.episode_id,
                decision_id=decision_id,
                config=self.config,
            )
        )

    def _on_step(self, connection, message: StepRequest) -> None:
        if self.active_episode != message.episode_id:
            connection.send_message(
                ErrorMessage(
                    message_id=new_message_id("err"),
                    episode_id=message.episode_id,
                    error_code=ErrorCode.STALE_EPISODE,
                    message="step for an inactive episode",
                    recoverable=True,
                )
            )
            return
        if message.tick != self.last_tick:
            connection.send_message(
                ErrorMessage(
                    message_id=new_message_id("err"),
                    episode_id=message.episode_id,
                    error_code=ErrorCode.STALE_TICK,
                    message=f"step tick {message.tick} != expected {self.last_tick}",
                    recoverable=True,
                )
            )
            return
        actions = [action.model_dump(exclude_none=True, mode="json") for action in message.actions]
        state = self.adapter.step(actions)
        state["decision_id"] = message.decision_id
        end_tick = int(state["tick"])
        receipts = state.get("receipts", [])
        executed_at = end_tick
        for receipt in receipts:
            reason = receipt.get("rejected_reason")
            connection.send_message(
                ReceiptMessage(
                    message_id=new_message_id("r"),
                    episode_id=message.episode_id,
                    tick=end_tick,
                    decision_id=message.decision_id,
                    accepted=bool(receipt.get("accepted")),
                    executed_at_tick=executed_at if receipt.get("accepted") else None,
                    rejected_reason=ErrorCode(reason) if reason else None,
                )
            )
        self.last_tick = end_tick
        connection.send_message(
            state_to_observation_message(
                state,
                episode_id=message.episode_id,
                decision_id=message.decision_id,
                config=self.config,
            )
        )
        if state.get("terminated") or state.get("truncated"):
            result = state.get("result") or "failure"
            connection.send_message(
                DoneMessage(
                    message_id=new_message_id("done"),
                    episode_id=message.episode_id,
                    tick=end_tick,
                    result=result,
                    reward=float(state.get("reward", 0.0)),
                    terminated=not bool(state.get("truncated")),
                    truncated=bool(state.get("truncated")),
                    info={"eval": state.get("eval")},
                )
            )


__all__ = [
    "ADAPTER_VERSION",
    "GAME_VERSION",
    "MockDuelBridge",
    "state_to_observation_message",
    "observation_message_to_state",
]
