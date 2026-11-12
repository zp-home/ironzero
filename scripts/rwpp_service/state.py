# SPDX-License-Identifier: Apache-2.0
"""进程内共享可变状态：任务表、端口池、子进程登记表、超时预算常量。

本模块是 ``rwpp_service`` 唯一的全局态所有者。

约定：

* ``JOBS`` / ``ACTIVE_PROCESSES`` / ``STOPPED_JOBS`` / ``PPO_PROCESSES`` /
  ``PPO_STOP`` / ``PORT_SLOTS`` 都是**可变容器对象**，其它模块用
  ``from .state import JOBS, JOBS_LOCK, ...`` 按引用共享，只允许原地增删，
  **不得重新赋值**。
* ``MAX_PARALLEL`` 是运行期由 ``server.main()`` 改写的 ``int``，因此读取方
  必须用 ``from . import state`` 再取 ``state.MAX_PARALLEL``（`from ... import`
  会在 import 时把旧值绑定死）。
"""

from __future__ import annotations

import queue
import subprocess
import threading

__all__ = [
    "JOBS",
    "JOBS_LOCK",
    "PORT_SLOTS",
    "MAX_PARALLEL",
    "TICK_OVERHEAD_MS",
    "EPISODE_TIMEOUT_MARGIN_S",
    "ACTIVE_PROCESSES",
    "STOPPED_JOBS",
    "PPO_PROCESSES",
    "PPO_STOP",
]

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
PORT_SLOTS: "queue.Queue[int]" = queue.Queue()
MAX_PARALLEL = 4

#: Wall-clock cost of one simulated tick *beyond* the pacing sleep. Measured on
#: the challenge maps, the engine + bridge + client pipeline adds ~6 ms per tick.
TICK_OVERHEAD_MS = 6.0
#: Extra wall clock for engine startup, map load and the final replay write.
EPISODE_TIMEOUT_MARGIN_S = 90.0


ACTIVE_PROCESSES: dict[str, list[subprocess.Popen]] = {}
STOPPED_JOBS: set[str] = set()

# ----------------------------------------------------------------------
# PPO jobs: strategy-in-the-loop training on real games
# ----------------------------------------------------------------------
PPO_PROCESSES: dict[str, subprocess.Popen] = {}
PPO_STOP: set[str] = set()
