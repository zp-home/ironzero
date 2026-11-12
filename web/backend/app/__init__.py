# SPDX-License-Identifier: Apache-2.0
"""IronZero training / inference platform backend.

A small FastAPI service that turns the CLI trainers into observable jobs:

* ``/api/runs``    — launch and monitor PPO training runs
* ``/api/infer``   — run inference episodes and stream live frames
* ``/api/replays`` — recorded episode library
* ``/``            — the dashboard (``platform/frontend``)

Everything runs locally: the service only listens where you bind it, jobs are
local subprocesses, and all artifacts live under ``IRONZERO_RUNS``.

The implementation was split into this package by responsibility:

``paths.py``    environment-derived roots and path-safety helpers
``models.py``   pydantic request bodies
``jobs.py``     job directories, meta files, subprocess supervision
``server.py``   ``create_app()``: FastAPI instance, CORS, routers, static mount
``routers/``    one module per resource family (system/runs/infer/replays/
                checkpoints/maps/rwpp)
"""

from __future__ import annotations

from .server import app, create_app

__all__ = ["app", "create_app"]
