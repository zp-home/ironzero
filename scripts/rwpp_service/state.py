# SPDX-License-Identifier: Apache-2.0
"""杩涚▼鍐呭叡浜彲鍙樼姸鎬侊細浠诲姟琛ㄣ€佺鍙ｆ睜銆佸瓙杩涚▼鐧昏琛ㄣ€佽秴鏃堕绠楀父閲忋€?
鏈ā鍧楁槸 ``rwpp_service`` 鍞竴鐨勫叏灞€鎬佹墍鏈夎€呫€?
绾﹀畾锛?
* ``JOBS`` / ``ACTIVE_PROCESSES`` / ``STOPPED_JOBS`` / ``PPO_PROCESSES`` /
  ``PPO_STOP`` / ``PORT_SLOTS`` 閮芥槸**鍙彉瀹瑰櫒瀵硅薄**锛屽叾瀹冩ā鍧楃敤
  ``from .state import JOBS, JOBS_LOCK, ...`` 鎸夊紩鐢ㄥ叡浜紝鍙厑璁稿師鍦板鍒狅紝
  **涓嶅緱閲嶆柊璧嬪€?*銆?* ``MAX_PARALLEL`` 鏄繍琛屾湡鐢?``server.main()`` 鏀瑰啓鐨?``int``锛屽洜姝よ鍙栨柟
  蹇呴』鐢?``from . import state`` 鍐嶅彇 ``state.MAX_PARALLEL``锛坄from ... import`
  浼氬湪 import 鏃舵妸鏃у€肩粦瀹氭锛夈€?"""

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
#: Default concurrent game instances (each one is a full headless JVM). Bumped
#: 4 -> 10: on the 16-core dev box (RTX 4070) a single game process is mostly
#: paced by ``Thread.sleep`` and idles at well under one core, so 10 slots fit
#: while keeping the batch collector saturated. ``--max-parallel`` still
#: overrides it per deployment, and the port pool is sized from whatever value
#: wins, so raising this number needs no other change.
MAX_PARALLEL = 10

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
