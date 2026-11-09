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
import warnings
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
    # Simulated-tick cap per episode. 3600 ticks = 60 s of game time: enough for a
    # 22-vs-25 battle to resolve, while the native game-over signal ends decided
    # battles early (a defeat typically arrives well before the cap).
    episode_ticks: int = 3600
    map_width_px: float = DEFAULT_MAP_PX_W
    map_height_px: float = DEFAULT_MAP_PX_H
    start_timeout: float = 90.0
    obs_timeout: float = 60.0
    # Wall-clock budget per simulated frame. 16 ms = real time (60 fps); 2 ms runs
    # faster on the wall clock. This never changes how many ticks an episode
    # simulates: that is ``episode_ticks`` below (60 ticks = 1 s of game time).
    frame_ms: int = 2
    vision_radius: float = 0.0  # >0 enables observation-level fog of war (enemy filtering)
    # Army size / observation contract. Defaults keep the legacy 8-unit scenario;
    # the fixed 25-vs-25 map sets units_per_team=25, max_entities>=25 and
    # target_slots=26 (or leaves target_slots=0 for the automatic value).
    units_per_team: int = DEFAULT_CONFIG.units_per_team
    max_entities: int = DEFAULT_CONFIG.max_entities
    target_slots: int = 0  # 0 = max_entities + 1 (25 enemies + "no target")
    # Terrain passability JSON (TerrainGrid ``map.json`` format). When it resolves
    # to a readable file the observation gains a static terrain channel and
    # ``obs_dim`` grows accordingly; a missing file falls back to the legacy
    # 3-channel observation with a warning.
    terrain_path: Path | None = None
    # Bridge pass-through options. Both are appended to the rwpp-server command
    # line only when set, so legacy invocations stay byte-identical.
    win_mode: str = ""  # e.g. "annihilation" -> --win-mode annihilation
    unlock_ai: bool = False  # -> --unlock-ai
    # Dense shaping: reward closing the distance to the enemy army (the skirmish
    # AI often starts far away, so pure hp-diff reward has no gradient early on).
    engagement_coef: float = 2.0
    # Reward terms (docs 7.2/7.4): kills/own losses plus small penalties for
    # rejected orders and for idle decisions while enemies are in sight.
    # LLM/rule strategic layer (optional): the directive nudges reward shaping only,
    # it never emits low-level actions (doc 12.4 / stage 5).
    strategy_coef: float = 0.5
    kill_bonus: float = 0.25
    loss_penalty: float = 0.25
    invalid_penalty: float = 0.05
    idle_penalty: float = 0.02
    log_dir: Path = REPO_ROOT / "runs" / "_rwpp_ppo"
    env_overrides: dict[str, str] = field(default_factory=dict)

    def resolved_max_entities(self) -> int:
        """Entity slots per block; never fewer than one full army."""

        return max(1, int(self.max_entities), int(self.units_per_team))

    def resolved_target_slots(self) -> int:
        """Target head size: explicit value, else ``max_entities + 1``."""

        explicit = int(self.target_slots or 0)
        if explicit > 0:
            return explicit
        return self.resolved_max_entities() + 1

    def resolved_terrain(self) -> Path | None:
        """Resolve ``terrain_path`` relative to cwd (then repo root)."""

        if not self.terrain_path:
            return None
        candidate = Path(self.terrain_path)
        if candidate.is_file():
            return candidate.resolve()
        if not candidate.is_absolute():
            fallback = (REPO_ROOT / candidate).resolve()
            if fallback.is_file():
                return fallback
        warnings.warn(
            f"terrain file not found: {self.terrain_path}; keeping the legacy "
            f"{DEFAULT_CONFIG.map_channels}-channel observation (no terrain channel)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def micro_config(self) -> MicroDuelConfig:
        """Observation/action contract aligned with the real map geometry."""

        terrain = self.resolved_terrain()
        channels = DEFAULT_CONFIG.map_channels + (1 if terrain is not None else 0)
        return replace(
            DEFAULT_CONFIG,
            map_width=float(self.map_width_px),
            map_height=float(self.map_height_px),
            episode_ticks=int(self.episode_ticks),
            vision_radius=float(self.vision_radius or 0.0),
            units_per_team=int(self.units_per_team),
            max_entities=self.resolved_max_entities(),
            target_slots=self.resolved_target_slots(),
            map_channels=channels,
            terrain_path=str(terrain) if terrain is not None else None,
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


FRAME_MS_WARNED = False


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
        "result": message.get("result"),
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

    The bridge understands the same group head as the mock engine: 0..G-1 order
    the units whose id maps to that group, and the last slot (G) broadcasts to
    every living friendly unit.
    """

    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    if vector.shape[0] != 5:
        raise ValueError(f"action must have 5 heads, got {vector.shape[0]}")
    kind = ACTION_KINDS[int(vector[0]) % len(ACTION_KINDS)]
    group = int(vector[1])
    gx = int(np.clip(vector[2], 0, config.position_grid - 1))
    gy = int(np.clip(vector[3], 0, config.position_grid - 1))
    target_index = int(vector[4])
    enemies = state.get("visible_enemy_entities", [])

    if kind in {"move_group", "attack_move"}:
        x, y = config.grid_to_world(gx, gy)
        return [{"kind": kind, "unit_group": group, "x": round(float(x), 2), "y": round(float(y), 2)}]
    if kind == "attack_target":
        if 0 <= target_index < len(enemies):
            return [
                {
                    "kind": kind,
                    "unit_group": group,
                    "target_entity_id": int(enemies[target_index].get("entity_id", -1)),
                }
            ]
        return _hold_action(state, group)
    if kind in {"stop", "hold"}:
        return _hold_action(state, group)
    return [{"kind": kind, "unit_group": group}]


def _hold_action(state: dict[str, Any], group: int = 4) -> list[dict[str, Any]]:
    """Stop/hold order; the bridge keeps each unit at its current position."""

    return [{"kind": "hold", "unit_group": int(group)}]


def apply_vision(state: dict[str, Any], radius: float) -> dict[str, Any]:
    """Observation-level fog of war: keep only enemies in sight of our units.

    The native bridge streams every unit; this filter makes the policy's view
    match what a human player would see, which is what ``vision_radius`` in the
    micro config describes. ``radius <= 0`` disables fog (full information).
    """

    if radius <= 0.0:
        return state
    own = state.get("own_entities", [])
    if not own:
        return state
    limit = float(radius) ** 2
    visible: list[dict[str, Any]] = []
    for enemy in state.get("visible_enemy_entities", []):
        ex = float(enemy.get("x") or 0.0)
        ey = float(enemy.get("y") or 0.0)
        for unit in own:
            dx = ex - float(unit.get("x") or 0.0)
            dy = ey - float(unit.get("y") or 0.0)
            if dx * dx + dy * dy <= limit:
                visible.append(enemy)
                break
    state["visible_enemy_entities"] = visible
    state["target_ids"] = [int(enemy.get("entity_id", -1)) for enemy in visible]
    return state


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


def army_hp_reference(
    state: dict[str, Any],
    config: MicroDuelConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """Episode-start army max-hp totals used to normalise hp fractions.

    This must run on the **ground-truth** state, never on the fogged policy
    view: the bridge streams every unit at reset, so the first frame carries the
    full army value even when ``vision_radius`` hides the enemies from the
    policy. Summing the visible enemy list instead would collapse the reference
    to a single unit (or to the configured fallback) at the start of a fogged
    episode and make every later hp fraction read as near-zero damage.

    ``max_hp`` falls back to ``config.unit_hp`` per entity; if the stream is
    empty the configured ``units_per_team * unit_hp`` is used.
    """

    expected = float(max(1, int(config.units_per_team)) * float(config.unit_hp))
    own_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("own_entities", [])
    )
    enemy_total = sum(
        float(unit.get("max_hp") or config.unit_hp)
        for unit in state.get("visible_enemy_entities", [])
    )
    own_ref = own_total if own_total > 0.0 else expected
    enemy_ref = enemy_total if enemy_total > 0.0 else expected
    return float(own_ref), float(enemy_ref)


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
        strategy: Any | None = None,
    ) -> None:
        self.game = game or RwppGameConfig()
        self.port = int(port)
        self.seed = int(seed)
        self.micro = micro_config or self.game.micro_config()
        self._launcher = launcher or launch_rwpp_process
        self.episode_prefix = episode_prefix
        self.strategy = strategy
        self.directive: Any | None = None
        self.last_strategy_potential: float | None = None

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
        self.last_distance = 0.0
        self.previous_own_alive = 0
        self.previous_enemy_alive = 0
        self.episode_components: dict[str, float] = {}
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
        # ``state`` stays the ground truth (used for termination / hp / reward);
        # ``obs_state`` is what the policy is allowed to see when fog is enabled.
        self.obs_state = self._observed(self.state)
        self.last_tick = int(self.state["tick"])
        self.masks = build_masks(self.obs_state, self.micro)
        # Reference totals: dead units vanish from the bridge stream, so reward and
        # evaluation fractions are measured against the episode-start army value.
        # Computed from ``self.state`` (ground truth: the bridge streams every
        # unit), NOT from ``self.obs_state`` (fogged policy view), otherwise a
        # fogged episode would normalise enemy hp against the few visible units.
        self.own_max_ref, self.enemy_max_ref = army_hp_reference(self.state, self.micro)
        self.last_potential = hp_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        self.last_distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        self.previous_own_alive = sum(
            1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        self.previous_enemy_alive = sum(
            1
            for unit in self.state.get("visible_enemy_entities", [])
            if float(unit.get("hp") or 0.0) > 0.0
        )
        self.episode_components: dict[str, float] = {}
        self.receipts = []
        self.done = False
        self.last_result = None
        self.episode_steps = 0
        self._refresh_strategy()
        self.last_strategy_potential = self._strategy_potential()
        obs = encode_observation(self.obs_state, self.micro)
        info = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "masks": self.masks,
            "raw_state": self.state,
            "visible_enemies": len(self.obs_state.get("visible_enemy_entities", [])),
            "tick": self.state["tick"],
            "max_refs": {
                "own": float(self.own_max_ref),
                "enemy": float(self.enemy_max_ref),
            },
            "port": self.port,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._socket is None or self._reader is None:
            raise RwppEnvError("reset() must be called before step()")
        self.episode_steps += 1
        previous_potential = float(self.last_potential)
        previous_distance = float(self.last_distance)
        previous_own_alive = int(self.previous_own_alive)
        previous_enemy_alive = int(self.previous_enemy_alive)
        receipts_before = len(self.receipts)
        # Decode against what the policy actually saw: with fog enabled the
        # target slots index the *visible* enemy list (as masked in obs_state),
        # so using the ground-truth list here would fire at the wrong enemy.
        actions = decode_action(action, self.obs_state, self.micro)
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
        except OSError:
            # The game died / the bridge was closed under us. Do not raise: let the
            # episode end and let the vector env restart this slot.
            disconnected = True
        else:
            disconnected = False

        message = None if disconnected else self._await_observation(self.game.obs_timeout)
        receipts_this_step = self.receipts[receipts_before:]
        if message is None:
            disconnected = True
            message = {}
        kind = message.get("type")
        # The bridge reports the native game result on the final observation and
        # on the ``done`` message; it is authoritative when present.
        bridge_result = message.get("result")
        if kind == "done":
            self.done = True
            # A ``done`` message ends the episode. Some bridge builds attach the
            # final entities/tick to it, others only the result: merge whatever is
            # present so the returned observation/masks describe the real terminal
            # state instead of the previous decision's state.
            state = dict(self.state)
            if message.get("tick") is not None:
                try:
                    state["tick"] = int(message["tick"])
                except (TypeError, ValueError):
                    pass
            if message.get("game_time") is not None:
                state["game_time"] = float(message["game_time"])
            if message.get("result") is not None:
                state["result"] = message["result"]
            if message.get("terminated") is not None:
                state["terminated"] = bool(message["terminated"])
            if message.get("truncated") is not None:
                state["truncated"] = bool(message["truncated"])
            for key in ("own_entities", "visible_enemy_entities"):
                if isinstance(message.get(key), list):
                    state[key] = [dict(entity) for entity in message[key]]
            if isinstance(message.get("own_entities"), list) or isinstance(
                message.get("visible_enemy_entities"), list
            ):
                state["target_ids"] = [
                    int(entity.get("entity_id", -1))
                    for entity in state.get("visible_enemy_entities", [])
                ]
            self.state = state
            self.obs_state = self._observed(state)
            self.last_tick = int(state.get("tick", self.last_tick))
        else:
            state = observation_to_state(message) if message else dict(self.state)
            self.state = state
            self.obs_state = self._observed(state)
            self.last_tick = int(state["tick"])
        if bridge_result is None:
            bridge_result = state.get("result")
        # Masks/observation are rebuilt from the (possibly fogged) state that the
        # policy actually saw, including on the terminal step.
        self.masks = build_masks(self.obs_state, self.micro)

        distance = army_distance_norm(
            self.state, float(np.hypot(self.micro.map_width, self.micro.map_height))
        )
        potential = hp_potential(self.state, self.own_max_ref, self.enemy_max_ref)
        own_alive = sum(1 for unit in self.state.get("own_entities", []) if float(unit.get("hp") or 0.0) > 0.0)
        enemy_alive = sum(
            1 for unit in self.state.get("visible_enemy_entities", []) if float(unit.get("hp") or 0.0) > 0.0
        )
        rejected_orders = sum(1 for receipt in receipts_this_step if not receipt.get("accepted", True))
        previous_strategy = self.last_strategy_potential
        self._refresh_strategy()
        strategy_now = self._strategy_potential()
        strategy_component = (
            float(self.game.strategy_coef) * (strategy_now - previous_strategy)
            if strategy_now is not None and previous_strategy is not None
            else 0.0
        )
        self.last_strategy_potential = strategy_now
        # ---- reward decomposition (docs 7.2/7.4) --------------------------
        components: dict[str, float] = {
            "potential": float(self.micro.potential_coef) * (potential - previous_potential),
            "engagement": float(self.game.engagement_coef) * (previous_distance - distance),
            "strategy": strategy_component,
            "kills": float(self.game.kill_bonus) * max(0, previous_enemy_alive - enemy_alive),
            "losses": -float(self.game.loss_penalty) * max(0, previous_own_alive - own_alive),
            "invalid": -float(self.game.invalid_penalty) * rejected_orders,
            "idle": (
                -float(self.game.idle_penalty)
                if (actions and actions[0].get("kind") in {"stop", "hold"} and enemy_alive > 0)
                else 0.0
            ),
            "terminal": 0.0,
            "timeout": 0.0,
        }
        self.last_potential = potential
        self.last_distance = distance
        self.previous_own_alive = own_alive
        self.previous_enemy_alive = enemy_alive

        terminated = False
        if bridge_result in {"victory", "defeat"}:
            # Native game-over signal (main buildings destroyed / team defeated):
            # trust the engine instead of waiting for total annihilation.
            terminated = True
            self.last_result = str(bridge_result)
            components["terminal"] = float(
                self.micro.win_reward if bridge_result == "victory" else self.micro.loss_reward
            )
        elif bridge_result is None:
            # Legacy/offline bridge without a result field: fall back to the
            # annihilation heuristic.
            if enemy_alive == 0 and own_alive > 0:
                terminated = True
                self.last_result = "victory"
                components["terminal"] = float(self.micro.win_reward)
            elif own_alive == 0:
                terminated = True
                self.last_result = "defeat"
                components["terminal"] = float(self.micro.loss_reward)
        reward = float(sum(components.values()))
        for key, value in components.items():
            self.episode_components[key] = self.episode_components.get(key, 0.0) + float(value)
        # A bridge-declared timeout (tick cap) or a dead connection truncates the
        # episode; a decided battle terminates it.
        truncated = (not terminated) and (
            self.done
            or disconnected
            or int(self.state.get("tick", 0)) >= int(self.game.episode_ticks)
        )
        if truncated:
            own_frac, enemy_frac = hp_fractions(self.state, self.own_max_ref, self.enemy_max_ref)
            # In arena annihilation battles that reach the tick limit, decide by
            # army survival / HP advantage rather than writing off clear-cut wins as
            # 0-win timeouts (matches standard RTS tournament rules).
            is_arena = (
                self.game.win_mode == "annihilation"
                or "ironzero" in str(self.game.map_relative).lower()
                or "ironzero" in str(self.game.map_name).lower()
            )
            if is_arena and not disconnected:
                if own_alive > enemy_alive and own_frac >= enemy_frac + 0.10:
                    self.last_result = "victory"
                    components["terminal"] = float(self.micro.win_reward) * 0.8
                elif enemy_alive > own_alive and enemy_frac >= own_frac + 0.10:
                    self.last_result = "defeat"
                    components["terminal"] = float(self.micro.loss_reward) * 0.8
                else:
                    self.last_result = "timeout"
            else:
                self.last_result = "timeout"
            reward = float(sum(components.values()))
            self.episode_components["terminal"] = float(
                self.episode_components.get("terminal", 0.0) + components["terminal"]
            )
        if terminated or truncated:
            self._shutdown_game()

        obs = encode_observation(self.obs_state, self.micro)
        own_frac, enemy_frac = hp_fractions(self.state, self.own_max_ref, self.enemy_max_ref)
        strategy_payload = None
        if self.directive is not None:
            to_payload = getattr(self.directive, "to_payload", None)
            if callable(to_payload):
                try:
                    strategy_payload = to_payload()
                except Exception:  # noqa: BLE001 - never break the env on logging
                    strategy_payload = None
        info = {
            "episode_id": self.episode_id,
            "tick": int(self.state.get("tick", 0)),
            "result": self.last_result,
            "bridge_result": bridge_result,
            "disconnected": bool(disconnected),
            "receipts": self.receipts,
            "masks": self.masks,
            "raw_state": self.state,
            "visible_enemies": len(self.obs_state.get("visible_enemy_entities", [])),
            "actions": actions,
            "own_alive": own_alive,
            "enemy_alive": enemy_alive,
            "distance": distance,
            "reward_breakdown": dict(components),
            "reward_components": dict(self.episode_components),
            "strategy": strategy_payload,
            "eval": {"own_hp_frac": own_frac, "enemy_hp_frac": enemy_frac},
            "max_refs": {
                "own": float(self.own_max_ref),
                "enemy": float(self.enemy_max_ref),
            },
            "port": self.port,
        }
        return obs, reward, terminated, truncated, info

    def _observed(self, state: dict[str, Any]) -> dict[str, Any]:
        """State as the policy sees it (fog of war applied when configured)."""

        radius = float(self.game.vision_radius or 0.0)
        if radius <= 0.0:
            return state
        return apply_vision(dict(state), radius)

    # ------------------------------------------------------------------
    # Strategic layer (LLM or rule based; reward shaping only)
    # ------------------------------------------------------------------
    def _strategy_summary(self) -> dict[str, Any]:
        own = self.state.get("own_entities", [])
        enemies = self.state.get("visible_enemy_entities", [])
        return {
            "tick": int(self.state.get("tick", 0)),
            "own_units": [
                {
                    "x": float(unit.get("x") or 0.0),
                    "y": float(unit.get("y") or 0.0),
                    "hp": float(unit.get("hp") or 0.0),
                    "max_hp": float(unit.get("max_hp") or 0.0),
                }
                for unit in own
            ],
            "enemy_units": [
                {
                    "x": float(unit.get("x") or 0.0),
                    "y": float(unit.get("y") or 0.0),
                    "hp": float(unit.get("hp") or 0.0),
                    "max_hp": float(unit.get("max_hp") or 0.0),
                }
                for unit in enemies
            ],
            "map": {"width": float(self.micro.map_width), "height": float(self.micro.map_height)},
        }

    def _refresh_strategy(self) -> None:
        if self.strategy is None:
            return
        propose = getattr(self.strategy, "propose", None)
        if propose is None:
            return
        try:
            self.directive = propose(self._strategy_summary())
        except Exception:  # noqa: BLE001 - strategy must never break training
            self.directive = None

    def _strategy_potential(self) -> float | None:
        """Progress towards the directive's target point, normalised to map size."""

        directive = self.directive
        if directive is None:
            return None
        target = None
        region = getattr(directive, "attack_region", None)
        rally = getattr(directive, "rally_point", None)
        intent = str(getattr(directive, "intent", ""))
        if intent in {"push", "harass"} and region:
            target = (float(region[0]), float(region[1]))
        elif rally:
            target = (float(rally[0]), float(rally[1]))
        elif region:
            target = (float(region[0]), float(region[1]))
        own = self.state.get("own_entities", [])
        if target is None or not own:
            return None
        cx = sum(float(unit.get("x") or 0.0) for unit in own) / len(own)
        cy = sum(float(unit.get("y") or 0.0) for unit in own) / len(own)
        diagonal = float(np.hypot(self.micro.map_width, self.micro.map_height)) or 1.0
        return -float(np.hypot(target[0] - cx, target[1] - cy)) / diagonal

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
                if self._reader.closed:
                    # Peer closed: fail fast instead of burning the whole timeout.
                    return None
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
    "apply_vision",
    "army_hp_reference",
    "decode_action",
    "hp_potential",
    "launch_rwpp_process",
    "observation_to_state",
]

