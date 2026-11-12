# SPDX-License-Identifier: Apache-2.0
"""Route groups for the platform backend.

One module per resource family so the dashboard's hard-coded URLs stay easy to
locate: ``system`` (health/dashboard/favicon), ``runs`` (PPO training jobs),
``infer`` (inference jobs + frame stream), ``replays`` (episode library),
``checkpoints``, ``maps`` (real-game assets) and ``rwpp`` (host-side real game
service proxy).
"""

from __future__ import annotations

from . import checkpoints, infer, maps, replays, rwpp, runs, system

__all__ = [
    "checkpoints",
    "infer",
    "maps",
    "replays",
    "rwpp",
    "runs",
    "system",
]
