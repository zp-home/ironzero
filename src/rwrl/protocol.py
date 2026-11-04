"""Versioned wire protocol for the Rusted Warfare RL bridge.

The protocol is intentionally transport-agnostic. The first transport is
newline-delimited JSON (NDJSON) over loopback TCP.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Mapping, TypeAlias, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

PROTOCOL_VERSION = "rwrl/1"
SCHEMA_VERSION = "1.0.0"


class ActionKind(StrEnum):
    MOVE_GROUP = "move_group"
    ATTACK_MOVE = "attack_move"
    ATTACK_TARGET = "attack_target"
    STOP = "stop"
    HOLD = "hold"


class ErrorCode(StrEnum):
    STALE_EPISODE = "STALE_EPISODE"
    STALE_TICK = "STALE_TICK"
    DUPLICATE_DECISION = "DUPLICATE_DECISION"
    UNKNOWN_UNIT = "UNKNOWN_UNIT"
    UNIT_NOT_OWNED = "UNIT_NOT_OWNED"
    TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"
    TARGET_INVALID = "TARGET_INVALID"
    POSITION_BLOCKED = "POSITION_BLOCKED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    ACTION_MASKED = "ACTION_MASKED"
    POLICY_TIMEOUT = "POLICY_TIMEOUT"
    BRIDGE_DISCONNECTED = "BRIDGE_DISCONNECTED"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"


class DoneResult(StrEnum):
    VICTORY = "victory"
    DEFEAT = "defeat"
    DRAW = "draw"
    TIMEOUT = "timeout"
    FAILURE = "failure"


class HeartbeatStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


class MessageBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    protocol: Literal["rwrl/1"] = PROTOCOL_VERSION
    message_id: str = Field(min_length=1, max_length=128)
    episode_id: str | None = Field(default=None, max_length=128)
    tick: int | None = Field(default=None, ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)


class _ActionBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    unit_group: int = Field(ge=0, le=64)


class MoveGroupAction(_ActionBase):
    kind: Literal[ActionKind.MOVE_GROUP] = ActionKind.MOVE_GROUP
    x: float
    y: float


class AttackMoveAction(_ActionBase):
    kind: Literal[ActionKind.ATTACK_MOVE] = ActionKind.ATTACK_MOVE
    x: float
    y: float


class AttackTargetAction(_ActionBase):
    kind: Literal[ActionKind.ATTACK_TARGET] = ActionKind.ATTACK_TARGET
    target_entity_id: int = Field(ge=0)


class StopAction(_ActionBase):
    kind: Literal[ActionKind.STOP] = ActionKind.STOP


class HoldAction(_ActionBase):
    kind: Literal[ActionKind.HOLD] = ActionKind.HOLD


Action: TypeAlias = Annotated[
    Union[
        MoveGroupAction,
        AttackMoveAction,
        AttackTargetAction,
        StopAction,
        HoldAction,
    ],
    Field(discriminator="kind"),
]


class ActionMasks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_mask: list[bool]
    unit_group_mask: list[bool]
    target_mask: list[bool]
    position_mask: list[list[bool]]


class GlobalState(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    funds: float | None = None
    income: float | None = None
    expense: float | None = None
    population: float | None = None
    population_cap: float | None = None
    unit_count: int | None = Field(default=None, ge=0)
    building_count: int | None = Field(default=None, ge=0)
    army_value: float | None = None
    visible_enemy_value: float | None = None
    kills: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)


class Entity(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    entity_id: int = Field(ge=0)
    entity_type_id: int = Field(ge=0)
    team_id: int = Field(ge=0)
    x: float
    y: float
    hp: float | None = None
    max_hp: float | None = None
    shield: float | None = None
    facing: float | None = None
    velocity: float | None = None
    is_building: bool = False
    build_progress: float | None = None
    production_queue_length: int | None = Field(default=None, ge=0)
    order_type: str | None = None
    attack_target_id: int | None = Field(default=None, ge=0)
    attack_cooldown: float | None = None
    carried_resource: float | None = None
    is_selected: bool = False
    is_alive: bool = True
    is_visible: bool = True
    last_seen_tick: int | None = Field(default=None, ge=0)


class HelloMessage(MessageBase):
    type: Literal["hello"] = "hello"
    adapter_version: str = Field(min_length=1, max_length=64)
    game_version: str = Field(min_length=1, max_length=128)
    capabilities: list[str] = Field(default_factory=list)


class ResetRequest(MessageBase):
    type: Literal["reset"] = "reset"
    episode_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0)
    scenario: str = Field(min_length=1, max_length=128)
    map_id: str = Field(min_length=1, max_length=256)
    options: dict[str, Any] = Field(default_factory=dict)


class StepRequest(MessageBase):
    type: Literal["step"] = "step"
    episode_id: str = Field(min_length=1, max_length=128)
    tick: int = Field(ge=0)
    decision_id: str = Field(min_length=1, max_length=128)
    actions: list[Action] = Field(min_length=1, max_length=64)


class ObservationMessage(MessageBase):
    type: Literal["obs"] = "obs"
    episode_id: str = Field(min_length=1, max_length=128)
    tick: int = Field(ge=0)
    decision_id: str = Field(min_length=1, max_length=128)
    game_time: float = Field(ge=0)
    controlled_team: int = Field(ge=0)
    visibility_revision: int = Field(ge=0)
    observation_schema_version: str = Field(min_length=1, max_length=64)
    global_state: GlobalState
    own_entities: list[Entity] = Field(default_factory=list)
    visible_enemy_entities: list[Entity] = Field(default_factory=list)
    remembered_enemies: list[Entity] = Field(default_factory=list)
    local_maps: dict[str, list[list[float]]] = Field(default_factory=dict)
    masks: ActionMasks
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ReceiptMessage(MessageBase):
    type: Literal["receipt"] = "receipt"
    episode_id: str = Field(min_length=1, max_length=128)
    tick: int = Field(ge=0)
    decision_id: str = Field(min_length=1, max_length=128)
    accepted: bool
    executed_at_tick: int | None = Field(default=None, ge=0)
    rejected_reason: ErrorCode | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "ReceiptMessage":
        if self.accepted and self.executed_at_tick is None:
            raise ValueError("accepted receipt requires executed_at_tick")
        if not self.accepted and self.rejected_reason is None:
            raise ValueError("rejected receipt requires rejected_reason")
        return self


class DoneMessage(MessageBase):
    type: Literal["done"] = "done"
    episode_id: str = Field(min_length=1, max_length=128)
    tick: int = Field(ge=0)
    result: DoneResult
    reward: float = 0.0
    terminated: bool = True
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(MessageBase):
    type: Literal["error"] = "error"
    error_code: ErrorCode
    message: str = Field(min_length=1, max_length=2048)
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class HeartbeatMessage(MessageBase):
    type: Literal["heartbeat"] = "heartbeat"
    status: HeartbeatStatus = HeartbeatStatus.RUNNING
    last_tick: int | None = Field(default=None, ge=0)


class ShutdownMessage(MessageBase):
    type: Literal["shutdown"] = "shutdown"
    reason: str = Field(min_length=1, max_length=512)
    deadline_ticks: int | None = Field(default=None, ge=0)


Message: TypeAlias = Union[
    HelloMessage,
    ResetRequest,
    StepRequest,
    ObservationMessage,
    ReceiptMessage,
    DoneMessage,
    ErrorMessage,
    HeartbeatMessage,
    ShutdownMessage,
]

Envelope: TypeAlias = Annotated[Message, Field(discriminator="type")]
EnvelopeAdapter: TypeAdapter[Envelope] = TypeAdapter(Envelope)


class ProtocolError(ValueError):
    """Raised when a message violates the bridge protocol."""

    def __init__(self, code: ErrorCode | str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


def parse_message(value: Mapping[str, Any] | str | bytes) -> Envelope:
    """Parse and validate one protocol message."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProtocolError(ErrorCode.INVALID_MESSAGE, f"invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError(ErrorCode.INVALID_MESSAGE, "message must be a JSON object")
    try:
        return EnvelopeAdapter.validate_python(value)
    except Exception as exc:
        raise ProtocolError(ErrorCode.INVALID_MESSAGE, str(exc)) from exc


def dump_message(message: BaseModel) -> dict[str, Any]:
    """Return a JSON-safe message dictionary without null optional fields."""

    return message.model_dump(mode="json", exclude_none=True)


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and replay comparisons."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def message_hash(message: BaseModel) -> str:
    payload = canonical_json(dump_message(message)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def message_schema() -> dict[str, Any]:
    """Build the JSON Schema for the discriminated message union."""

    schema = EnvelopeAdapter.json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/zp-home/rusted-warfare-rl/schemas/rwrl/1/messages.schema.json"
    schema["title"] = "RWRL v1 bridge message"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema


class SequenceGuard:
    """Minimal per-episode ordering guard for bridge messages."""

    def __init__(self, episode_id: str, max_seen_decisions: int = 4096) -> None:
        self.episode_id = episode_id
        self.last_tick = -1
        self.seen_decisions: set[str] = set()
        self.max_seen_decisions = max_seen_decisions

    def check(self, message: MessageBase) -> None:
        if message.episode_id != self.episode_id:
            raise ProtocolError(ErrorCode.STALE_EPISODE, "message belongs to another episode")
        if message.tick is not None:
            if message.tick < self.last_tick:
                raise ProtocolError(ErrorCode.STALE_TICK, "message tick moved backwards")
            self.last_tick = message.tick
        if isinstance(message, StepRequest):
            if message.decision_id in self.seen_decisions:
                raise ProtocolError(ErrorCode.DUPLICATE_DECISION, "decision was already submitted")
            self.seen_decisions.add(message.decision_id)
            if len(self.seen_decisions) > self.max_seen_decisions:
                self.seen_decisions.clear()

