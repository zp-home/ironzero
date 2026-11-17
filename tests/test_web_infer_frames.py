# SPDX-License-Identifier: Apache-2.0
"""The two frame routes must serve windows out of the index, not a full parse.

These are contract tests for the endpoints the dashboard polls every 700 ms:
``GET /api/infer/{id}/frames`` and ``GET /api/rwpp/infer/{id}/frames``. They pin
the three properties the incremental reader depends on:

* the payload is what the old ``_read_jsonl(path)[after:]`` produced;
* ``next`` is an absolute record cursor, so a bounded response is resumable;
* the routes never parse the whole file per call (asserted by counting parses,
  which is the actual regression: cost must not grow with the episode).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from web.backend.app import jsonl_index as index_module


def _write_stream(path: Path, count: int, *, episode_end: bool = True) -> list[dict]:
    records: list[dict] = [{"type": "start", "meta": {"map_id": "m"}}]
    for tick in range(1, count + 1):
        records.append(
            {
                "type": "frame",
                "tick": tick,
                "time": round(tick / 60.0, 3),
                "units": [{"id": 0, "x": float(tick), "y": 0.0, "team_id": 0}],
                "events": [],
                "stats": {"own_hp_frac": 1.0, "enemy_hp_frac": 1.0},
            }
        )
    if episode_end:
        records.append({"type": "episode_end", "episode": 0, "result": "victory"})
    records.append({"type": "done", "episodes": []})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


@pytest.fixture()
def infer_job(tmp_path, monkeypatch):
    """A fake inference job directory wired into the router's roots."""

    from web.backend.app import jobs
    from web.backend.app.routers import infer as infer_router

    index_module.clear_indexes()
    runs = tmp_path / "platform"
    job_dir = runs / "infer-20260101-000000-abcdef"
    records = _write_stream(job_dir / "frames.jsonl", 40)
    (job_dir / "meta.json").write_text(
        json.dumps({"id": job_dir.name, "kind": "infer", "status": "succeeded"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(infer_router, "RUNS_ROOT", runs)
    monkeypatch.setattr(jobs, "RUNS_ROOT", runs)
    try:
        yield job_dir.name, records
    finally:
        index_module.clear_indexes()


class _CountingIndex(index_module.JsonlIndex):
    """Counts *byte-reading* passes, i.e. the whole-file parse we removed.

    ``refresh`` is called on every access (it is just a ``stat`` when nothing
    changed, so it is the cheap path); ``_extend`` is the call that actually
    reads and parses lines, and that is what must happen once per new record
    rather than once per poll.
    """

    extends = 0

    def _extend(self) -> None:
        _CountingIndex.extends += 1
        super()._extend()


def test_frames_route_agrees_with_the_old_slice(infer_job) -> None:
    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, records = infer_job
    client = TestClient(app)
    for after in (0, 1, 17, len(records) - 1, len(records) + 5):
        response = client.get(f"/api/infer/{job_id}/frames", params={"after": after})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["frames"] == records[after:], f"cursor {after} diverged"
        assert payload["next"] == len(records)


def test_frames_route_defaults_to_the_whole_tail(infer_job) -> None:
    """No ``limit`` must return everything, exactly like the old slice.

    The replay loader depends on this: it fetches a finished job's whole stream
    in one request to build the playback buffer.
    """

    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, records = infer_job
    payload = TestClient(app).get(f"/api/infer/{job_id}/frames").json()
    assert payload["frames"] == records
    assert payload["next"] == len(records)


def test_frames_route_honours_limit_and_stays_resumable(infer_job) -> None:
    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, records = infer_job
    client = TestClient(app)
    seen: list[dict] = []
    cursor = 0
    for _ in range(len(records) + 2):
        payload = client.get(
            f"/api/infer/{job_id}/frames", params={"after": cursor, "limit": 7}
        ).json()
        assert len(payload["frames"]) <= 7
        seen.extend(payload["frames"])
        cursor = payload["next"]
        if not payload["frames"]:
            break
    assert seen == records
    assert cursor == len(records)


def test_frames_route_accepts_limit_zero_for_a_server_chosen_chunk(infer_job) -> None:
    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, records = infer_job
    client = TestClient(app)
    seen: list[dict] = []
    cursor = 0
    for _ in range(len(records) + 2):
        payload = client.get(
            f"/api/infer/{job_id}/frames", params={"after": cursor, "limit": 0}
        ).json()
        seen.extend(payload["frames"])
        cursor = payload["next"]
        if not payload["frames"]:
            break
    assert seen == records
    assert cursor == len(records)


def test_frames_route_rejects_an_absurd_limit(infer_job) -> None:
    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, _ = infer_job
    client = TestClient(app)
    over = index_module.MAX_WINDOW_RECORDS + 1
    assert client.get(
        f"/api/infer/{job_id}/frames", params={"limit": over}
    ).status_code == 422
    assert client.get(f"/api/infer/{job_id}/frames", params={"limit": -1}).status_code == 422


def test_status_route_answers_from_the_index(infer_job) -> None:
    """``frames_total``/``latest_frame``/``results`` must not need a full scan."""

    from fastapi.testclient import TestClient

    from web.backend.app import app

    job_id, records = infer_job
    payload = TestClient(app).get(f"/api/infer/{job_id}").json()
    frames = [item for item in records if item.get("type") == "frame"]
    assert payload["frames_total"] == len(frames)
    assert payload["latest_frame"] == frames[-1]
    assert payload["results"] == [item for item in records if item.get("type") == "episode_end"]
    assert payload["episodes_done"] == 1
    assert payload["done"] is True


def test_polling_does_not_reparse_the_stream(tmp_path, monkeypatch) -> None:
    """The regression: work per poll must stay proportional to the *new* records.

    A whole-file parse per poll is what made playback degrade as the episode
    grew, so this asserts the index is built once and merely extended.
    """

    from web.backend.app import jobs
    from web.backend.app.routers import infer as infer_router

    index_module.clear_indexes()
    runs = tmp_path / "platform"
    job_dir = runs / "infer-20260101-000000-countme"
    records = _write_stream(job_dir / "frames.jsonl", 500)
    (job_dir / "meta.json").write_text(json.dumps({"id": job_dir.name}), encoding="utf-8")
    monkeypatch.setattr(infer_router, "RUNS_ROOT", runs)
    monkeypatch.setattr(jobs, "RUNS_ROOT", runs)

    from fastapi.testclient import TestClient

    from web.backend.app import app

    client = TestClient(app)
    _CountingIndex.extends = 0
    real = index_module.JsonlIndex
    index_module.JsonlIndex = _CountingIndex  # type: ignore[misc]
    try:
        # A bounded follow, which is the pattern that must stay cheap. (An
        # unbounded first read deliberately does a combined read-and-index pass,
        # so it is not the case being measured here.)
        cursor = 0
        while True:
            page = client.get(
                f"/api/infer/{job_dir.name}/frames",
                params={"after": cursor, "limit": 200},
            ).json()
            cursor = page["next"]
            if not page["frames"]:
                break
        extends_after_drain = _CountingIndex.extends
        assert extends_after_drain >= 1, "the first poll did not index anything"
        # Ten more polls for the same unchanged file must not read any byte.
        for _ in range(10):
            again = client.get(
                f"/api/infer/{job_dir.name}/frames",
                params={"after": cursor, "limit": 200},
            ).json()
            assert again["frames"] == []
            assert again["next"] == cursor
        assert _CountingIndex.extends == extends_after_drain, (
            "an unchanged stream was re-read"
        )
    finally:
        index_module.JsonlIndex = real  # type: ignore[misc]
        index_module.clear_indexes()


def test_rwpp_frames_route_uses_the_index(tmp_path, monkeypatch) -> None:
    """The real-game route proxies job state, but must read frames the same way."""

    from web.backend.app.routers import rwpp as rwpp_router

    index_module.clear_indexes()
    stream = tmp_path / "frames.jsonl"
    records = _write_stream(stream, 30)
    monkeypatch.setattr(rwpp_router, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        rwpp_router,
        "_rwpp_call",
        lambda *a, **k: {
            "id": "job-1",
            "kind": "infer",
            "status": "running",
            "frames_file": "frames.jsonl",
        },
    )
    try:
        from fastapi.testclient import TestClient

        from web.backend.app import app

        client = TestClient(app)
        payload = client.get("/api/rwpp/infer/job-1/frames", params={"after": 5}).json()
        assert payload["frames"] == records[5:]
        assert payload["next"] == len(records)
        assert payload["status"] == "running"

        windowed = client.get(
            "/api/rwpp/infer/job-1/frames", params={"after": 0, "limit": 3}
        ).json()
        assert windowed["frames"] == records[:3]
        assert windowed["next"] == 3
    finally:
        index_module.clear_indexes()
