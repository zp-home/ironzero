# SPDX-License-Identifier: Apache-2.0
"""Incremental frame reads: the index must be cheap, exact, and crash-safe.

Reported symptom: the sandbox playback got choppier the longer an inference run
went. Cause: every poll answered ``[after:]`` by parsing the *whole* frame
stream first, so the work per poll grew with the episode (a 154 MB / 11 727
record stream measured 1 970 ms per call, and ten concurrent jobs multiplied
that inside the single backend process).

These tests pin the replacement contract on ``web.backend.app.jsonl_index``:

* ``read_window`` returns exactly the records a full parse would, from the same
  cursor arithmetic the dashboard uses;
* ``next`` is an absolute record index, so a size-capped window is catch-up-able;
* a partially flushed trailing line is never handed out;
* a truncated/rewritten stream is detected instead of serving stale offsets;
* the count/get helpers used by the status endpoint agree with a full scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from web.backend.app.jsonl_index import (
    DEFAULT_WINDOW_BYTES,
    DEFAULT_WINDOW_RECORDS,
    MAX_WINDOW_RECORDS,
    JsonlIndex,
    clear_indexes,
    jsonl_index,
)


def _frame(tick: int, units: int = 2) -> dict:
    return {
        "type": "frame",
        "tick": tick,
        "time": round(tick / 60.0, 3),
        "units": [{"id": index, "x": tick * 2.0 + index, "y": 1.0} for index in range(units)],
        "events": [],
        "stats": {"own_hp_frac": 1.0, "enemy_hp_frac": 1.0},
    }


def _write_stream(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _full_parse(path: Path) -> list[dict]:
    """Reference implementation: the previous whole-file behaviour."""

    out = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


@pytest.fixture()
def stream(tmp_path: Path) -> Path:
    clear_indexes()
    path = tmp_path / "frames.jsonl"
    records = [{"type": "start", "meta": {"map": "x"}}]
    records.extend(_frame(tick) for tick in range(1, 61))
    records.append({"type": "episode_end", "episode": 0, "result": "victory"})
    records.append({"type": "done", "episodes": []})
    _write_stream(path, records)
    try:
        yield path
    finally:
        clear_indexes()


def test_window_matches_a_full_parse_for_every_cursor(stream: Path) -> None:
    """``read_window`` must be a drop-in for ``full[i:]`` at any cursor."""

    expected = _full_parse(stream)
    index = jsonl_index(stream)
    for after in range(0, len(expected) + 3):
        records, next_cursor = index.read_window(after, limit=MAX_WINDOW_RECORDS)
        assert records == expected[after:], f"cursor {after} diverged"
        assert next_cursor == len(expected)


def test_default_window_is_unbounded(tmp_path: Path) -> None:
    """``limit=None`` must behave exactly like the old ``full[after:]``.

    This is what the replay loader depends on: it fetches a finished job's whole
    stream in one call, so a silent cap would leave the sandbox playing a
    fraction of the match.
    """

    clear_indexes()
    try:
        path = tmp_path / "frames.jsonl"
        records = [{"type": "start"}] + [_frame(tick) for tick in range(1, 700)]
        _write_stream(path, records)
        got, next_cursor = jsonl_index(path).read_window(0)
        assert got == records, "a default read dropped records"
        assert next_cursor == len(records)
        tail, next_cursor = jsonl_index(path).read_window(600)
        assert tail == records[600:]
        assert next_cursor == len(records)
    finally:
        clear_indexes()


def test_limit_zero_selects_a_bounded_chunk(stream: Path) -> None:
    """``limit=0`` means "a sensible chunk", and it must stay resumable."""

    expected = _full_parse(stream)
    index = jsonl_index(stream)
    seen: list[dict] = []
    cursor = 0
    for _ in range(len(expected) + 2):
        records, cursor = index.read_window(cursor, limit=0)
        assert len(records) <= DEFAULT_WINDOW_RECORDS
        if not records:
            break
        seen.extend(records)
    assert seen == expected
    assert cursor == len(expected)


def test_default_window_caps_an_explicit_limit(stream: Path) -> None:
    """An explicit small ``limit`` must not lose records: asking again finishes."""

    expected = _full_parse(stream)
    index = jsonl_index(stream)

    seen: list[dict] = []
    cursor = 0
    for _ in range(len(expected) + 2):
        records, cursor = index.read_window(cursor, limit=1)
        assert len(records) <= 1
        if not records:
            break
        seen.extend(records)
    assert seen == expected
    assert cursor == len(expected)


def test_bounded_window_respects_the_byte_ceiling(tmp_path: Path) -> None:
    """Big frames must cap by bytes even when the record cap is far away."""

    clear_indexes()
    try:
        path = tmp_path / "frames.jsonl"
        # ~900 KB per record, so a single pair blows the 1.5 MB default budget.
        _write_stream(path, [_frame(tick, units=25000) for tick in range(1, 7)])
        assert path.stat().st_size > DEFAULT_WINDOW_BYTES * 2
        records, next_cursor = jsonl_index(path).read_window(0, limit=0)
        assert 1 <= len(records) < 6, "the byte ceiling did not bite"
        total = sum(len(json.dumps(item)) for item in records)
        assert total <= DEFAULT_WINDOW_BYTES * 1.2, total
        # Always at least one record, even if that record alone is over budget.
        assert next_cursor == len(records)
    finally:
        clear_indexes()


def test_appended_records_are_picked_up_without_rereading(stream: Path) -> None:
    """The poll after a write must return only the new record."""

    index = jsonl_index(stream)
    total = index.total
    with stream.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_frame(999)) + "\n")
    records, next_cursor = index.read_window(total, limit=MAX_WINDOW_RECORDS)
    assert next_cursor == total + 1
    assert [item["tick"] for item in records] == [999]


def test_partial_trailing_line_is_not_handed_out(stream: Path) -> None:
    """A half-flushed write is deferred, then delivered whole."""

    expected = _full_parse(stream)
    index = jsonl_index(stream)
    total = index.total
    payload = json.dumps(_frame(1234))
    cut = len(payload) // 2
    with stream.open("a", encoding="utf-8") as handle:
        handle.write(payload[:cut])

    records, next_cursor = index.read_window(total, limit=MAX_WINDOW_RECORDS)
    assert records == []
    assert next_cursor == total

    with stream.open("a", encoding="utf-8") as handle:
        handle.write(payload[cut:] + "\n")
    records, next_cursor = index.read_window(total, limit=MAX_WINDOW_RECORDS)
    assert next_cursor == total + 1
    assert len(records) == 1 and records[0]["tick"] == 1234
    assert index.read_window(0, limit=MAX_WINDOW_RECORDS)[0] == expected + records


def test_truncated_stream_is_detected_and_rebuilt(stream: Path) -> None:
    """A restarted job rewrites the same path; stale offsets must not survive."""

    index = jsonl_index(stream)
    assert index.total > 3
    _write_stream(stream, [_frame(7)])
    records, next_cursor = index.read_window(0, limit=MAX_WINDOW_RECORDS)
    assert next_cursor == 1
    assert [item["tick"] for item in records] == [7]


def test_kind_helpers_agree_with_a_full_scan(stream: Path) -> None:
    """The status endpoint's counts/lookups must match the old list filtering."""

    expected = _full_parse(stream)
    index = jsonl_index(stream)
    assert index.kind_count("frame") == len([i for i in expected if i.get("type") == "frame"])
    assert index.read_kind("episode_end") == [i for i in expected if i.get("type") == "episode_end"]
    assert index.read_kind("done") == [i for i in expected if i.get("type") == "done"]
    assert index.read_last("frame") == [i for i in expected if i.get("type") == "frame"][-1]
    assert index.has("done") is True
    assert index.has("missing") is False


def test_records_without_a_type_are_counted_as_frames(tmp_path: Path) -> None:
    """``applyInferBatch`` plays a bare ``{"units": [...]}`` record as a frame."""

    clear_indexes()
    try:
        path = tmp_path / "frames.jsonl"
        _write_stream(path, [{"units": [{"id": 1}], "tick": 1}, {"type": "done"}])
        index = jsonl_index(path)
        assert index.kind_count("frame") == 1
        assert index.read_last("frame")["tick"] == 1
    finally:
        clear_indexes()


def test_unparseable_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Matches ``_read_jsonl``: one bad line must not stall the stream.

    The byte-probe index keeps a position for the malformed line (it cannot know
    it is malformed without parsing it), so the cursor advances past it and the
    caller never re-requests it; the record itself is dropped when read. Blank
    lines carry no position at all.
    """

    clear_indexes()
    try:
        path = tmp_path / "frames.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(_frame(1)) + "\n")
            handle.write("\n")
            handle.write("{not json}\n")
            handle.write(json.dumps(_frame(2)) + "\n")
        records, next_cursor = jsonl_index(path).read_window(0, limit=MAX_WINDOW_RECORDS)
        assert [item["tick"] for item in records] == [1, 2]
        # 3 indexed lines (frame, garbage, frame) -- the garbage one is consumed,
        # not revisited, so the cursor cannot loop on it.
        assert next_cursor == 3
        assert jsonl_index(path).kind_count("frame") == 2, "garbage counted as a frame"
        # Reading again from the cursor yields nothing and stays put.
        again, again_cursor = jsonl_index(path).read_window(next_cursor)
        assert again == [] and again_cursor == 3
    finally:
        clear_indexes()


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    clear_indexes()
    try:
        path = tmp_path / "nope.jsonl"
        index = jsonl_index(path)
        assert index.total == 0
        assert index.read_window(0) == ([], 0)
        assert index.read_kind("done") == []
        assert index.read_last("frame") is None
    finally:
        clear_indexes()


def test_cursor_past_the_end_is_clamped(tmp_path: Path) -> None:
    clear_indexes()
    try:
        path = tmp_path / "frames.jsonl"
        _write_stream(path, [_frame(1), _frame(2)])
        records, next_cursor = jsonl_index(path).read_window(99, limit=10)
        assert records == []
        assert next_cursor == 2
    finally:
        clear_indexes()


def test_index_is_reused_across_calls(stream: Path) -> None:
    """The whole point: repeated polls reuse one index and never re-read bytes."""

    first = jsonl_index(stream)
    second = jsonl_index(stream)
    assert first is second
    # A bounded tail read builds the index once...
    first.read_window(0, limit=5)
    scanned = first.scanned
    assert scanned > 0
    # ...and every later poll on an unchanged file is a stat, not a re-read.
    for _ in range(5):
        first.read_window(0, limit=5)
        first.read_window(scanned, limit=5)
        assert first.scanned == scanned, "an unchanged file re-read bytes"


def test_direct_construction_does_not_share_state(tmp_path: Path) -> None:
    """Tests and callers can hold a private index without touching the LRU."""

    path = tmp_path / "frames.jsonl"
    _write_stream(path, [_frame(1)])
    private = JsonlIndex(path)
    private.refresh()
    assert private.total == 1
    clear_indexes()
    assert private.total == 1
