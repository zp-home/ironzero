# SPDX-License-Identifier: Apache-2.0
"""Real-game RL environment for Rusted Warfare 1.15 (RWPP headless + IronZero bridge).

This is the "policy in the loop" counterpart of ``scripts/record_rwpp_replay.py``:
instead of a scripted controller, an external policy decides every
``DECISION_INTERVAL`` game ticks. One environment instance owns:

* one ``rwpp-server`` process launched with the IronZero bridge extension,
* one loopback TCP connection speaking ``ironzero/1`` (obs stream + step actions),
* the conversion from bridge observations to the flat observation vector and
  action masks used by :mod:`ironzero.observations`,
* hp-differential reward shaping plus win/loss/terminal bookkeeping.

The game itself is not resettable in-process, so ``reset()`` starts a fresh
headless process. That costs a few seconds of JVM startup, which is why
:class:`ironzero.training.vec_rwpp.VecRwppEnv` resets several instances in
parallel and keeps them running across PPO rollouts.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .micro_config import DEFAULT_CONFIG, MicroDuelConfig
from .mock_engine import ACTION_KINDS
from .observations import build_masks, encode_observation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME = Path(os.environ.get("IRONZERO_RWPP_RUNTIME", r"E:\IronZeroBuild\rwpp-runtime-1.15"))
DEFAULT_JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))
DEFAULT_MAP = "maps/challenge/Artillery.tmx"
DEFAULT_MAP_NAME = "Artillery"
# The challenge map ships pre-placed armies (22 player units vs 25 AI units), so a
# fresh instance starts as a real micro battle instead of an economy skirmish.
DEFAULT_MAP_PX_W = 2700.0
DEFAULT_MAP_PX_H = 2500.0
# The bridge streams one observation per 10 simulated ticks (DECISION_INTERVAL in
# IronZeroBridge.java); this is part of the bridge contract, not a tunable here.
DECISION_TICKS = 10


class RwppEnvError(RuntimeError):
    """Raised when the real game process or bridge connection fails."""


@dataclass
class RwppGameConfig:
    """Everything needed to start one headless real-game instance."""

    runtime: Path = DEFAULT_RUNTIME
    java_home: Path = DEFAULT_JAVA_HOME
    map_relative: str = DEFAULT_MAP
    map_name: str = DEFAULT_MAP_NAME
    episode_ticks: int = 1200
    map_width_px: float = DEFAULT_MAP_PX_W
    map_height_px: float = DEFAULT_MAP_PX_H
    start_timeout: float = 90.0
    obs_timeout: float = 60.0
    # Dense shaping: reward closing the distance to the enemy army (the skirmish
    # AI often starts far away, so pure hp-diff reward has no gradient early on).
    engagement_coef: float = 2.0
    log_dir: Path = REPO_ROOT / "runs" / "_rwpp_ppo"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def micro_config(self) -> MicroDuelConfig:
        """Observation/action contract aligned with the real map geometry."""

        return replace(
            DEFAULT_CONFIG,
            map_width=float(self.map_width_px),
            map_height=float(self.map_height_px),
            episode_ticks=int(self.episode_ticks),
            vision_radius=0.0,
        )


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
    try:
        process = subprocess.Popen(
            [
                str(launcher),
                "--no-console",
                "--",
                "--bridge-port",
                str(port),
                "--ticks",
                str(int(game.episode_ticks)),
                "--map",
                game.map_relative,
            ],
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


class _LineReader:
    """Newline-delimited JSON reader with an internal buffer."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""

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
                return None
            if not chunk:
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


def observation_to_state(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one bridge ``obs`` message into the encoder's raw state dict.

    The real bridge only reports positions/hp for alive units; group and order
    metadata are optional and default to group 0 / idle in the encoder.
    """

    enemies = [dict(entity) for entity in message.get("visible_enemy_entities", [])]
    state = {
        "tick": int(message.get("tick", 0)),
        "game_time": float(message.get("game_time", 0.0)),
        "controlled_team": int(message.get("controlled_team", 0)),
        "global_state": dict(message.get("global_state") or {}),
        "own_entities": [dict(entity) for entity in message.get("own_entities", [])],
        "visible_enemy_entities": enemies,
        "target_ids": [int(entity.get("entity_id", -1)) for entity in enemies],
        "receipts": [],
        "result": None,
        "terminated": bool(message.get("terminated", False)),
        "truncated": bool(message.get("truncated", False)),
    }
    return state


def decode_action(
    action: np.ndarray,
    state: dict[str, Any],
    config: MicroDuelConfig,
) -> list[dict[str, Any]]:
    """Decode one policy action into ``ironzero/1`` step actions.

    The real-game bridge always orders every living friendly unit, so the group
    head selects a target group only in mock engines; here it is accepted and
    ignored (broadcast semantics), matching the scripted controller behaviour.
    """

    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    if vector.shape[0] != 5:
        raise ValueError(f"action must have 5 heads, got {vector.shape[0]}")
    kind = ACTION_KINDS[int(vector[0]) % len(ACTION_KINDS)]
    gx = int(np.clip(vector[2], 0, config.position_grid - 1))
    gy = int(np.clip(vector[3], 0, config.position_grid - 1))
    target_index = int(vector[4])
    enemies = state.get("visible_enemy_entities", [])

    if kind in {"move_group", "attack_move"}:
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": kind, "unit_group": 0, "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "attack_target":
        if 0 <= target_index < len(enemies):
            return [
                {
                    "kind": kind,
                    "unit_group": 0,
                    "target_entity_id": int(enemies[target_index].get("entity_id", -1)),
                }
            ]
        return _hold_action(state)
    if kind in {"stop", "hold"}:
        return _hold_action(state)
    return [{"kind": kind, "unit_group": 0}]


def _hold_action(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The bridge has no native "stop": move onto the current army centroid."""

    own = state.get("own_entities", [])
    if not own:
        return []
    cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
    cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
    return [{"kind": "move_group", "unit_group": 0, "x": round(cx, 2), "y": round(cy, 2)}]


def hp_potential(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> float:
    """Normalised hp advantage in ``[-1, 1]`` (own minus enemy).

    Dead units disappear from the bridge stream, so the reference totals from
    the first observation of the episode are used when available; otherwise the
    alive-unit maximum is the denominator.
    """

    own, enemy = hp_fractions(state, own_max_ref, enemy_max_ref)
    return own - enemy


def hp_fractions(
    state: dict[str, Any],
    own_max_ref: float | None = None,
    enemy_max_ref: float | None = None,
) -> tuple[float, float]:
    """Return ``(own_hp_frac, enemy_hp_frac)`` from a raw state."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    own_hp = sum(float(unit.get("hp") or 0.0) for unit in own)
    enemy_hp = sum(float(unit.get("hp") or 0.0) for unit in enemies)
    own_max = sum(float(unit.get("max_hp") or 0.0) for unit in own) or 1.0
    enemy_max = sum(float(unit.get("max_hp") or 0.0) for unit in enemies) or 1.0
    own_ref = float(own_max_ref) if own_max_ref else own_max
    enemy_ref = float(enemy_max_ref) if enemy_max_ref else enemy_max
    # The AI may field reinforcements, so fractions are clamped to [0, 1].
    return (
        max(0.0, min(1.0, own_hp / max(1.0, own_ref))),
        max(0.0, min(1.0, enemy_hp / max(1.0, enemy_ref))),
    )


def army_distance_norm(state: dict[str, Any], map_diagonal: float) -> float:
    """Normalised distance between the two army centroids (0 = same spot)."""

    own = state.get("own_entities", [])
    enemies = state.get("visible_enemy_entities", [])
    if not own or not enemies:
        return 0.0
    own_cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
    own_cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
    enemy_cx = sum(float(unit.get("x") or 0.0) for unit in enemies) / len(enemies)
    enemy_cy = sum(float(unit.get("y") or 0.0) for unit in enemies) / len(enemies)
    diagonal = max(1.0, float(map_diagonal))
    return min(1.0, float(np.hypot(enemy_cx - own_cx, enemy_cy - own_cy)) / diagonal)


class RwppDuelEnv:
    """Single real-game environment driven through the IronZero bridge."""

    metadata = {"render_modes": [], "name": "IronZeroRwppDuel-v0"}

    def __init__(
        self,
        game: RwppGameConfig | None = None,
        *,
        port: int = 5140,
        seed: int = 0,
        micro_config: MicroDuelConfig | None = None,
        launcher: Callable[[RwppGameConfig, int], _ProcLike] | None = None,
        episode_prefix: str = "rwpp",
    ) -> None:
        self.game = game or RwppGameConfig()
        self.port = int(port)
        self.seed = int(seed)
        self.micro = micro_config or self.game.micro_config()
        self._launcher = launcher or launch_rwpp_process
        self.episode_prefix = episode_prefix

        self._process: _ProcLike | None = None
        self._socket: socket.socket | None = None
        self._reader: _LineReader | None = None
        self._log_handle = None
        self._episode_counter = 0
        self.episode_id = ""
        self.state: dict[str, Any] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.last_tick = 0
        self.last_potential = 0.0
        self.receipts: list[dict[str, Any]] = []
        self.done = False
        self.last_result: str | None = None
        self.episode_steps = 0
        self.observation_space = _BoxSpace((self.micro.flat_obs_dim,))
        self.action_space = _MultiDiscreteSpace(self.micro.action_nvec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed = int(seed)
        self._episode_counter += 1
        self.episode_id = f"{self.episode_prefix}-{self.seed}-{self._episode_counter}"
        self._shutdown_game()
        self._start_game()
        message = self._await_observation(self.game.start_timeout + self.game.obs_timeout)
        if message is None:
            raise RwppEnvError(f"no observation from real game on port {self.port}")
        self.state = observation_to_state(message)
        self.last_tick = int(self.state["tick"])
        self.masks = build_masks(self.state, self.micro)
        # Reference totals: dead units vanish from the bridge stream, so reward and
        # evaluation fractions are measured against the episode-start army value.
        self.own_max_ref = sum(
            float(unit.get("max_hp") or 0.0) for unit in self.state.get("own_entities", [])
        )
        self.enemy_max_ref = sum(
            float(unit.get("max_hp") or 0.0) for unit in self.state.get("visible_enemy_entities", [])
        )
        self.last_potential = hp_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        self.last_distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        self.receipts = []
        self.done = False
        self.last_result = None
        self.episode_steps = 0
        obs = encode_observation(self.state, self.micro)
        info = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "masks": self.masks,
            "raw_state": self.state,
            "tick": self.state["tick"],
            "port": self.port,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._socket is None or self._reader is None:
            raise RwppEnvError("reset() must be called before step()")
        self.episode_steps += 1
        actions = decode_action(action, self.state, self.micro)
        decision_id = f"d-{self.episode_id}-{self.episode_steps}"
        payload = {
            "protocol": "ironzero/1",
            "type": "step",
            "message_id": f"m-{decision_id}",
            "episode_id": self.episode_id,
            "tick": self.last_tick,
            "decision_id": decision_id,
            "actions": actions,
        }
        try:
            self._socket.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except OSError as exc:
            raise RwppEnvError(f"bridge send failed: {exc}") from exc

        message = self._await_observation(self.game.obs_timeout)
        truncated_by_timeout = False
        if message is None:
            # The bridge is gone or the game died mid-episode.
            truncated_by_timeout = True
            message = {}
        kind = message.get("type")
        if kind == "done":
            self.done = True
            self.last_result = "timeout"
            state = dict(self.state)
        else:
            state = observation_to_state(message) if message else dict(self.state)
            self.state = state
            self.last_tick = int(state["tick"])
        self.masks = build_masks(self.state, self.micro)

        potential = hp_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        reward = float(self.micro.potential_coef) * (potential - self.last_potential)
        self.last_potential = potential

        distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        reward += float(self.game.engagement_coef) * (self.last_distance - distance)
        self.last_distance = distance

        own_alive = sum(1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0)
        enemy_alive = sum(
            1 for unit in self.state.get("visible_enemy_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        terminated = False
        if enemy_alive == 0 and own_alive > 0:
            terminated = True
            self.last_result = "victory"
            reward += float(self.micro.win_reward)
        elif own_alive == 0:
            terminated = True
            self.last_result = "defeat"
            reward += float(self.micro.loss_reward)
        truncated = self.done or truncated_by_timeout or (
            int(self.state.get("tick", 0)) >= int(self.game.episode_ticks)
        )
        if truncated and self.last_result is None:
            self.last_result = "timeout"
        if terminated or truncated:
            self._shutdown_game()

        obs = encode_observation(self.state, self.micro)
        own_frac, enemy_frac = hp_fractions(self.state, self.own_max_ref, self.enemy_max_ref)
        info = {
            "episode_id": self.episode_id,
            "tick": int(self.state.get("tick", 0)),
            "result": self.last_result,
            "receipts": self.receipts,
            "masks": self.masks,
            "raw_state": self.state,
            "actions": actions,
            "own_alive": own_alive,
            "enemy_alive": enemy_alive,
            "distance": distance,
            "eval": {"own_hp_frac": own_frac, "enemy_hp_frac": enemy_frac},
            "port": self.port,
        }
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> dict[str, np.ndarray]:
        return self.masks

    def close(self) -> None:
        self._shutdown_game()

    def __enter__(self) -> "RwppDuelEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Process / bridge plumbing
    # ------------------------------------------------------------------
    def _start_game(self) -> None:
        self._process = self._launcher(self.game, self.port)
        deadline = time.time() + self.game.start_timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
            except OSError as exc:
                last_error = exc
                time.sleep(0.25)
                continue
            sock.settimeout(5.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket = sock
            self._reader = _LineReader(sock)
            return
        self._shutdown_game()
        raise RwppEnvError(
            f"bridge port {self.port} never opened (last error: {last_error})"
        )

    def _await_observation(self, timeout: float) -> dict[str, Any] | None:
        if self._reader is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            message = self._reader.read(max(0.5, min(remaining, 10.0)))
            if message is None:
                continue
            kind = message.get("type")
            if kind == "obs":
                return message
            if kind == "receipt":
                self.receipts.append(dict(message))
                continue
            if kind == "done":
                self.done = True
                return message
            if kind == "hello":
                continue
        return None

    def _shutdown_game(self) -> None:
        if self._socket is not None:
            try:
                self._socket.sendall(b'{"type":"shutdown"}\n')
            except OSError:
                pass
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
            self._reader = None
        if self._process is not None:
            try:
                self._process.wait(timeout=15.0)
            except Exception:  # noqa: BLE001 - timeout/OSError both mean "still alive"
                self._process.kill()
            self._process = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None


class _BoxSpace:
    def __init__(self, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:  # pragma: no cover - diagnostics only
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, value: np.ndarray) -> bool:
        return getattr(value, "shape", None) == self.shape


class _MultiDiscreteSpace:
    def __init__(self, nvec: tuple[int, ...]) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.nvec),)

    def sample(self) -> np.ndarray:
        return np.array([np.random.randint(0, int(n)) for n in self.nvec], dtype=np.int64)

    def contains(self, value: np.ndarray) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.shape == self.shape
            and np.all(value >= 0)
            and np.all(value < self.nvec)
        )


__all__ = [
    "DECISION_TICKS",
    "RwppDuelEnv",
    "RwppEnvError",
    "RwppGameConfig",
    "decode_action",
    "hp_potential",
    "launch_rwpp_process",
    "observation_to_state",
]

