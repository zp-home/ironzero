# SPDX-License-Identifier: Apache-2.0
"""进程入口：argparse + 端口池填充 + ThreadingHTTPServer 装配。

原来的 ``main()`` 用 ``global MAX_PARALLEL`` 改写模块级整数；拆分后该整数住在
:mod:`rwpp_service.state`，所以这里改写 ``state.MAX_PARALLEL``，所有读取方
都以属性访问取值（``from . import state``），语义保持一致。
"""

from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer

from . import state
from .handler import Handler

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IronZero host-side real-game service")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-parallel", type=int, default=4, help="maximum concurrent game instances")
    args = parser.parse_args(argv)
    state.MAX_PARALLEL = max(1, args.max_parallel)
    for slot in range(state.MAX_PARALLEL):
        state.PORT_SLOTS.put(5130 + slot)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[rwpp-service] listening on http://{args.host}:{args.port} max_parallel={state.MAX_PARALLEL}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[rwpp-service] stopping", flush=True)
    finally:
        server.server_close()
    return 0


def cli() -> None:  # pragma: no cover - 便捷入口
    raise SystemExit(main(sys.argv[1:]))
