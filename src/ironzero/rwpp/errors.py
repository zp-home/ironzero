# SPDX-License-Identifier: Apache-2.0
"""Environment-level exception type for the real-game RWPP bridge."""

from __future__ import annotations


class RwppEnvError(RuntimeError):
    """Raised when the real game process or bridge connection fails."""


__all__ = ["RwppEnvError"]
