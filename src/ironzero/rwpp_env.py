# SPDX-License-Identifier: Apache-2.0
"""兼容 shim：实现已迁移至 :mod:`ironzero.rwpp`。

保留这个模块名是因为它是历史 import 路径（脚本、训练代码与测试都用
``from ironzero.rwpp_env import ...``），而且 ``tests/test_rwpp_25v25.py``
用 ``mock.patch("ironzero.rwpp_env.subprocess.Popen")`` 打补丁：``patch`` 解析
该路径后把 ``Popen`` 替换到 **共享的** ``subprocess`` 模块对象上，所以
``ironzero.rwpp.process`` 里的 ``subprocess.Popen`` 调用同样会被拦住。
为此 shim 必须保留下面这行 ``import subprocess``，否则 ``patch`` 找不到
``ironzero.rwpp_env.subprocess`` 属性而直接报错。
"""

from __future__ import annotations

import subprocess  # noqa: F401 - monkeypatch target: ironzero.rwpp_env.subprocess.Popen

from .rwpp import (  # noqa: F401
    DECISION_TICKS,
    DEFAULT_JAVA_HOME,
    DEFAULT_MAP,
    DEFAULT_MAP_NAME,
    DEFAULT_MAP_PX_H,
    DEFAULT_MAP_PX_W,
    DEFAULT_RUNTIME,
    FRAME_MS_WARNED,
    REPO_ROOT,
    RwppDuelEnv,
    RwppEnvError,
    RwppGameConfig,
    StepShaping,
    _BoxSpace,
    _LineReader,
    _MultiDiscreteSpace,
    _ProcLike,
    _hold_action,
    apply_vision,
    army_distance_norm,
    army_hp_reference,
    compute_step_shaping,
    decode_action,
    effective_frame_ms,
    hp_fractions,
    hp_potential,
    launch_rwpp_process,
    observation_to_state,
)

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
