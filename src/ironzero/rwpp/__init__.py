# SPDX-License-Identifier: Apache-2.0
"""Real-game Rusted Warfare environment (RWPP headless + IronZero bridge).

Layering, outermost first:

* :mod:`ironzero.rwpp.env` -- ``RwppDuelEnv``: reset/step orchestration, episode
  bookkeeping, terminal adjudication.
* :mod:`ironzero.rwpp.reward_shaping` -- per-step reward decomposition.
* :mod:`ironzero.rwpp.reward` -- hp/distance potential terms.
* :mod:`ironzero.rwpp.decoding` / :mod:`ironzero.rwpp.vision` -- protocol
  translation and fog of war.
* :mod:`ironzero.rwpp.transport` -- NDJSON line framing on the bridge socket.
* :mod:`ironzero.rwpp.process` / :mod:`ironzero.rwpp.config` -- process launch,
  pacing and the game configuration contract.

``ironzero.rwpp_env`` remains as a compatibility shim re-exporting this package.
"""

from __future__ import annotations

from .config import (
    DECISION_TICKS,
    DEFAULT_JAVA_HOME,
    DEFAULT_MAP,
    DEFAULT_MAP_NAME,
    DEFAULT_MAP_PX_H,
    DEFAULT_MAP_PX_W,
    DEFAULT_RUNTIME,
    REPO_ROOT,
    RwppGameConfig,
)
from .decoding import _hold_action, decode_action, observation_to_state
from .env import RwppDuelEnv
from .errors import RwppEnvError
from .process import FRAME_MS_WARNED, _ProcLike, effective_frame_ms, launch_rwpp_process
from .reward import army_distance_norm, army_hp_reference, hp_fractions, hp_potential
from .reward_shaping import StepShaping, compute_step_shaping
from .spaces import _BoxSpace, _MultiDiscreteSpace
from .transport import _LineReader
from .vision import apply_vision

__all__ = [
    "DECISION_TICKS",
    "DEFAULT_JAVA_HOME",
    "DEFAULT_MAP",
    "DEFAULT_MAP_NAME",
    "DEFAULT_MAP_PX_H",
    "DEFAULT_MAP_PX_W",
    "DEFAULT_RUNTIME",
    "FRAME_MS_WARNED",
    "REPO_ROOT",
    "RwppDuelEnv",
    "RwppEnvError",
    "RwppGameConfig",
    "StepShaping",
    "_BoxSpace",
    "_LineReader",
    "_MultiDiscreteSpace",
    "_ProcLike",
    "_hold_action",
    "apply_vision",
    "army_distance_norm",
    "army_hp_reference",
    "compute_step_shaping",
    "decode_action",
    "effective_frame_ms",
    "hp_fractions",
    "hp_potential",
    "launch_rwpp_process",
    "observation_to_state",
]
