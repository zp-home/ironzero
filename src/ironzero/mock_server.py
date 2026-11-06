# SPDX-License-Identifier: Apache-2.0
"""Run the mock micro-duel bridge as a standalone NDJSON server.

Useful for manual protocol inspection and for exercising the same TCP path the
future RWPP extension will use::

    python -m ironzero.mock_server --port 5123

Then in another shell attach the TCP adapter / training code to port 5123.
"""

from __future__ import annotations

import argparse
import sys
import time

from .mock_bridge import GAME_VERSION, MockDuelBridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IronZero mock micro-duel NDJSON bridge")
    parser.add_argument("--port", type=int, default=5123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "idle", "random"])
    parser.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N seconds (0 = forever)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = MockDuelBridge(port=args.port, host=args.host, opponent=args.opponent)
    server.start()
    print(
        f"mock bridge listening on {args.host}:{server.port} "
        f"(adapter={GAME_VERSION}, opponent={args.opponent})",
        flush=True,
    )
    started = time.monotonic()
    try:
        while args.seconds <= 0 or time.monotonic() - started < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("interrupted, shutting down", flush=True)
    finally:
        server.stop()
    if server.error is not None:
        print(f"server error: {server.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
