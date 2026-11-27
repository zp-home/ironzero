# SPDX-License-Identifier: Apache-2.0
"""Regression: the gymnasium-facing env seed must reach the RWPP engine.

``_bridge_command`` reads ``game.seed`` for ``--seed``. The env's seed lives on
``RwppDuelEnv.seed`` and is set by both the constructor and ``reset(seed=N)``.
Before this fix the two were disconnected: probes/evals that passed ``seed=``
silently ran with the engine's random seed, making win rates irreproducible.
"""
from __future__ import annotations

from pathlib import Path

from ironzero.rwpp.config import RwppGameConfig
from ironzero.rwpp.process import _bridge_command
from ironzero.rwpp.session import start_game


class _FakeProc:
    def poll(self) -> int:
        return 0


class _Env:
    def __init__(self, value: int, game: RwppGameConfig) -> None:
        self.game = game
        self.seed = value
        self.port = 5140
        self._process = None

    def _launcher(self, config: RwppGameConfig, _port: int) -> _FakeProc:
        self.launched = config  # type: ignore[attr-defined]
        return _FakeProc()

    def _shutdown_game(self) -> None:
        pass


def test_env_seed_forwarded_without_mutating_shared_config(monkeypatch) -> None:
    import ironzero.rwpp.session as session

    game = RwppGameConfig()
    env = _Env(4242, game)
    monkeypatch.setattr(session.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("stub")))
    game.start_timeout = 0
    try:
        start_game(env)  # type: ignore[arg-type]
    except Exception:
        pass

    launched = env.launched  # type: ignore[attr-defined]
    assert isinstance(launched, RwppGameConfig)
    assert launched.seed == 4242
    assert game.seed == 0, "VecRwppEnv shares one config across differently-seeded slots"
    cmd = _bridge_command(Path("rwpp-server.bat"), 5140, launched)
    assert cmd[cmd.index("--seed") + 1] == "4242"


def test_zero_seed_is_not_forwarded(monkeypatch) -> None:
    import ironzero.rwpp.session as session

    game = RwppGameConfig()
    env = _Env(0, game)
    monkeypatch.setattr(session.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("stub")))
    game.start_timeout = 0
    try:
        start_game(env)  # type: ignore[arg-type]
    except Exception:
        pass
    launched = env.launched  # type: ignore[attr-defined]
    assert "--seed" not in _bridge_command(Path("rwpp-server.bat"), 5140, launched)
