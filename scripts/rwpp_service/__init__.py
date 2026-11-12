# SPDX-License-Identifier: Apache-2.0
"""Host-side "real game" service for the IronZero platform.

The platform backend runs inside Docker without Java, so the real Rusted
Warfare 1.15 runtime lives on the Windows host behind this small HTTP service.

包内分层（只搬结构，不改行为）：

* :mod:`rwpp_service.state`   全部进程内可变全局态（JOBS / 端口池 / 子进程表）
* :mod:`rwpp_service.paths`   仓库内路径与任务记录读写辅助
* :mod:`rwpp_service.proc`    受管子进程执行、单任务停止、全量停止
* :mod:`rwpp_service.runners` ``/infer`` 与 ``/train`` 的采集执行体
* :mod:`rwpp_service.ppo`     ``/ppo`` 训练执行体 + 预设收敛 + live 事件读取
* :mod:`rwpp_service.launch`  由 HTTP payload 创建任务（StartMixin）
* :mod:`rwpp_service.handler` HTTP 路由（Handler）
* :mod:`rwpp_service.server`  ``main(argv)`` 与 ThreadingHTTPServer 装配

入口仍是脚本 ``scripts/rwpp_real_service.py``（薄 shim）。
"""

__all__ = [
    "handler",
    "launch",
    "paths",
    "ppo",
    "proc",
    "runners",
    "server",
    "state",
]
