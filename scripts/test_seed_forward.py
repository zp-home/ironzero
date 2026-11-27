"""Regression test: env.seed must reach the engine as --seed (game.seed)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.process import _bridge_command
from ironzero.rwpp.session import start_game


class _FakeProc:
    def poll(self):
        return 0


def test_env_seed_forwarded_to_engine():
    """The seed passed to RwppDuelEnv must become --seed on the bridge command."""
    game = RwppGameConfig()
    captured: dict = {}

    class _Env:
        def __init__(self, seed: int):
            self.game = game
            self.seed = seed
            self.port = 5140
            self._process = None

        def _launcher(self, g, port):
            captured["game"] = g
            return _FakeProc()

        def _shutdown_game(self):
            pass

    env = _Env(seed=4242)
    # start_game also opens a socket; stub socket.create_connection to fail fast by
    # using a tiny timeout and catching the expected error after the launcher ran.
    import ironzero.rwpp.session as session

    orig = session.socket.create_connection

    def _fail(*_a, **_k):
        raise OSError("stub")

    session.socket.create_connection = _fail
    try:
        env.game = game
        # Patch start_timeout so the loop exits immediately.
        fast = game
        try:
            object.__setattr__(fast, "start_timeout", 0)
        except Exception:
            fast.start_timeout = 0
        env.game = fast
        try:
            start_game(env)
        except Exception:
            pass
    finally:
        session.socket.create_connection = orig

    launched = captured["game"]
    assert launched.seed == 4242, f"expected engine seed 4242, got {launched.seed}"
    assert game.seed == 0, "the shared config must not be mutated (VecRwppEnv contract)"
    cmd = _bridge_command(Path("rwpp-server.bat"), 5140, launched)
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "4242"
    print("PASS: env.seed=4242 -> --seed 4242, shared config unchanged")


if __name__ == "__main__":
    test_env_seed_forwarded_to_engine()
