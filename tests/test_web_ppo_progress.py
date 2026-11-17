# SPDX-License-Identifier: Apache-2.0
"""The Web console must show a PPO run that has already finished.

Reported symptom: a training run completed on the host (``status=done``,
``updates_done=80/80``, and ``/api/rwpp/ppo/{id}/events`` returned all 163 live
events) while the Web page showed an empty form and no result after a refresh. From
the operator's side that is indistinguishable from "the run never happened".

Cause: ``feRestorePpoJob`` attached only to a job whose ``status`` was
``running``, so a finished job was never restored. Opening the page mid-run worked;
opening it after the run finished showed nothing.

These tests pin the contract at the source level, because the logic lives in a
browser ES module: the backend already serves finished jobs correctly, so the bug
could only be in the restore predicate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PPO_RUN_JS = REPO / "web" / "frontend" / "js" / "ppoRun.js"


def _source() -> str:
    return PPO_RUN_JS.read_text(encoding="utf-8")


def test_restore_no_longer_requires_a_running_job() -> None:
    """A finished job must be attachable, not just a live one."""

    source = _source()
    assert "const running = jobs.find((job) => job.status === 'running');" in source
    # The old code returned early when nothing was running.
    old_guard = "if (!running) {\n      if (state.rwpp.ppoRunning > 0"
    assert old_guard not in source, (
        "restoring only a running job makes a finished run invisible after refresh"
    )
    assert "const finished = jobs" in source, (
        "the newest finished job must be a fallback for restoration"
    )
    assert "const attach = running || finished;" in source


def test_restore_prefers_a_running_job_over_a_finished_one() -> None:
    """Live progress must win over a stale result."""

    source = _source()
    attach_line = next(
        line for line in source.splitlines() if "const attach = " in line
    )
    assert attach_line.index("running") < attach_line.index("finished"), (
        "a running job must take priority, or a new run would show the old result"
    )


def test_finished_job_is_attached_without_claiming_it_is_live() -> None:
    """The console must not present a completed run as stoppable."""

    source = _source()
    assert "const isRunning = ppo.status === 'running';" in source
    assert "feSetPpoRunning(isRunning);" in source, (
        "a finished job must not enable the stop button"
    )
    assert "已载入最近一次 PPO 训练结果" in source, (
        "the user must be told this is a loaded result, not a live run"
    )


def test_polling_stops_for_a_non_running_job() -> None:
    """Attaching to a finished job must not start an endless poll loop."""

    source = _source()
    assert "if (status !== 'running') {" in source
    assert "stopFePpoPolling();" in source


def test_frontend_reads_the_events_endpoint_the_backend_exposes() -> None:
    """The client and the route must agree on the path and the ``after`` cursor."""

    api_file = REPO / "web" / "frontend" / "js" / "api.js"
    api_source = api_file.read_text(encoding="utf-8")
    assert "rwppPpoEvents" in api_source
    assert "/events" in api_source
    router = REPO / "web" / "backend" / "app" / "routers" / "rwpp.py"
    router_source = router.read_text(encoding="utf-8")
    assert '@router.get("/api/rwpp/ppo/{job_id}/events")' in router_source


def test_backend_forwards_the_event_cursor() -> None:
    router_source = (
        REPO / "web" / "backend" / "app" / "routers" / "rwpp.py"
    ).read_text(encoding="utf-8")
    assert "events?after={after}" in router_source, (
        "the cursor must be forwarded or the client re-reads every event"
    )


def test_host_service_reports_updates_done_for_finished_jobs() -> None:
    """``updates_done`` is what the dashboard renders; it must survive completion."""

    handler_source = (
        REPO / "scripts" / "rwpp_service" / "handler.py"
    ).read_text(encoding="utf-8")
    for field in ("updates_total", "updates_done", "status"):
        assert f'"{field}"' in handler_source, (
            f"the events endpoint must return {field} so a restored job renders"
        )
