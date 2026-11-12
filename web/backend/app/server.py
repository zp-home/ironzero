# SPDX-License-Identifier: Apache-2.0
"""Application factory: FastAPI instance, CORS, routers and static mount.

``uvicorn web.backend.app:app`` and ``web/Dockerfile``'s CMD both resolve the
module-level ``app`` exported by ``web.backend.app.__init__``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .paths import APP_VERSION, FRONTEND_DIR
from .routers import checkpoints, infer, maps, replays, rwpp, runs
from .routers import system


def create_app() -> FastAPI:
    """Build the ASGI application (single instance per process)."""

    application = FastAPI(title="IronZero Platform", version=APP_VERSION)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Route order mirrors the pre-split single-module app.
    application.include_router(system.router)
    application.include_router(runs.router)
    application.include_router(infer.router)
    application.include_router(replays.router)
    application.include_router(checkpoints.router)
    application.include_router(maps.router)
    application.include_router(rwpp.router)

    # Static frontend assets
    application.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), check_dir=False),
        name="static",
    )

    # Must stay after the mount to preserve the historical route order.
    application.include_router(system.favicon_router)
    return application


app = create_app()
