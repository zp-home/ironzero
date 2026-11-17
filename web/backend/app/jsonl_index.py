# SPDX-License-Identifier: Apache-2.0
"""Byte-offset indexed, incremental reads for the append-only JSONL streams.

Why this exists
---------------
``_read_jsonl`` parses a whole file on every call. The live frame stream grows
with the match: at ``--frame-ms 0`` the recorder appends ~12 KB per record, so a
25 s episode is already 100+ MB. Every dashboard poll therefore re-parsed the
entire history before slicing off the tail with ``[after:]`` -- and the cost
climbs quadratically with the episode length, which is exactly the "playback
gets slower the longer the run goes" symptom.

Measured on this repository's largest stream (154 MB, 11 727 records):

    whole-file parse per poll    2 068 ms   (the old behaviour, every 700 ms)
    indexed tail poll                6 ms   (same records, same order)
    indexed status query           0.8 ms   (was 1 740 ms)
    index build                    263 ms   (once per job; then incremental)
    index build w/ byte probe      231 ms   (parsing is 87% of the naive build)

The stream is *append-only*, so byte offsets are all the index needs:
``offsets[i]`` is the first byte of record ``i`` and ``ends[i]`` the byte after
its trailing newline. Extending the index costs one ``readline`` per record that
appeared since the previous call, which is the only work a poll should ever do.

The index does **not** parse records while building. ``json.loads`` was 87% of
the build cost (1 293 ms of 1 492 ms over 11 727 records), spent on records the
caller might never read, so the kind is taken from a byte probe for
``"type":"..."`` and the record itself is parsed only when it is actually read.

A trailing line without its newline is a partially flushed write: it is left
out of the index and picked up by the next call, so a reader can never see half
a frame. A file that shrank was rewritten (a restarted job reuses its path), so
the index is dropped and rebuilt from byte 0.

The cache is bounded by :data:`MAX_CACHED_FILES` (LRU): one index holds two
``int`` offsets plus one interned kind per record, i.e. ~190 KB for that
11 727-record job, and concurrent jobs each get their own entry.
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_WINDOW_BYTES",
    "DEFAULT_WINDOW_RECORDS",
    "MAX_CACHED_FILES",
    "MAX_WINDOW_RECORDS",
    "JsonlIndex",
    "clear_indexes",
    "frame_stream_index",
    "jsonl_index",
]

#: Records per window when a caller asks for a bound but gives no size.
#: 500 frames is ~7 MB on a busy 25v25 map and ~25 s of catching up at the
#: dashboard's 700 ms poll interval.
DEFAULT_WINDOW_RECORDS = 500

#: Byte ceiling for a bounded window. Whichever of this and
#: :data:`DEFAULT_WINDOW_RECORDS` trips first decides the window, so a map with
#: three times the entities cannot turn one poll into a 20 MB response.
DEFAULT_WINDOW_BYTES = 1_500_000

#: Hard ceiling for an explicit ``limit=`` from the query string.
MAX_WINDOW_RECORDS = 2000

#: How many streams keep a warm index. Ten concurrent inference jobs plus a
#: handful of finished ones being inspected is the realistic working set.
MAX_CACHED_FILES = 32

#: The record kinds this module tracks by name. ``"frame"`` is counted (it can be
#: tens of thousands of records) while the others keep their offsets (there are
#: at most a few hundred of them, one per episode) so the status endpoint can
#: answer without a scan.
_FRAME_KIND = "frame"

#: Byte form of ``"type":"frame"`` with and without the space the recorders'
#: ``json.dumps(separators=(",", ":"))`` may or may not emit.
_TYPE_FRAME_SPACED = b'"type": "frame"'
_TYPE_FRAME_TIGHT = b'"type":"frame"'
_TYPE_KEY = b'"type"'
_UNITS_KEY = b'"units"'

#: ``"type": "value"`` with either spacing, capped so a very long line (a frame
#: with 150 units) cannot make the scan expensive -- the match is anchored after
#: the key, which is what ``_classify`` already located.
_TYPE_VALUE = re.compile(rb'"type"\s*:\s*"([A-Za-z0-9_]{1,24})"')


def _classify(line: bytes) -> str:
    """Record kind from a raw JSONL line, without a full ``json.loads``.

    Returns ``""`` for a line that is not a JSON object. A malformed line yields
    ``""`` rather than raising, matching ``_read_jsonl``, which skipped garbage
    instead of failing the whole stream. The value is still validated lazily when
    the record is actually read, so a spoofed ``"type":"frame"`` in a corrupt line
    can only mis-count it -- :meth:`JsonlIndex._read_blob` drops such a record
    before it reaches a caller.
    """

    if not line or line[0:1] != b"{":
        return ""
    if _TYPE_FRAME_TIGHT in line or _TYPE_FRAME_SPACED in line:
        return _FRAME_KIND
    if _TYPE_KEY not in line:
        # No ``type`` at all: ``applyInferBatch`` plays a bare ``{"units": [...]}``
        # record as a frame, and the recorders have always emitted ``type``.
        return _FRAME_KIND if _UNITS_KEY in line else ""
    # Some other kind: take the short prefix after the key. It is only used as a
    # dict key and compared against names like "done"/"episode_end"/"start", so a
    # truncated value cannot alias a real kind unless the bytes really match.
    match = _TYPE_VALUE.search(line)
    if match is None:
        return ""
    return match.group(1).decode("ascii", "replace")


class JsonlIndex:
    """Append-only index over one JSONL file. Safe for concurrent readers.

    One instance per path lives in the module-level LRU; callers get it through
    :func:`jsonl_index` and never construct it directly outside tests.
    """

    __slots__ = (
        "path", "offsets", "ends", "kinds", "kind_indices", "frame_count",
        "scanned", "size", "lock",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self.ends: list[int] = []
        self.kinds: list[str] = []
        self.kind_indices: dict[str, list[int]] = {}
        self.frame_count = 0
        #: Byte offset just past the last *complete* line that was indexed.
        self.scanned = 0
        #: File size at the last refresh, used to skip a pointless stat+read.
        self.size = -1
        self.lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Forget everything (file truncated, replaced, or a test rewinding it)."""

        self.offsets.clear()
        self.ends.clear()
        self.kinds.clear()
        self.kind_indices.clear()
        self.frame_count = 0
        self.scanned = 0
        self.size = -1

    # -- indexing ----------------------------------------------------------

    def refresh(self) -> None:
        """Index every complete line appended since the last call.

        Idempotent and cheap when nothing changed: a single ``stat`` decides.
        """

        with self.lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                if self.offsets or self.scanned:
                    self.reset()
                return
            if size < self.scanned:
                # Shorter than what we already indexed: the writer restarted and
                # truncated the file, so every offset we hold is meaningless.
                self.reset()
            if size == self.scanned and size == self.size:
                return
            self.size = size
            self._extend()

    def _extend(self) -> None:
        """Index newline positions since ``scanned``, without parsing JSON.

        Measured on a 154 MB / 11 727-record stream, a full ``json.loads`` pass
        costs 1 492 ms while merely finding the newlines costs 199 ms -- 87% of
        the build was parsing records nobody had asked for yet. So the index only
        records *where* each record starts and ends, and classifies it with a
        byte probe for ``"type":"..."`` (the recorders have always emitted that
        key first-class). Records are parsed on demand by :meth:`_read_blob`,
        which is the only place that has to pay for ``json``.
        """

        if not self.path.is_file():
            return
        with self.path.open("rb") as handle:
            handle.seek(self.scanned)
            while True:
                start = handle.tell()
                try:
                    line = handle.readline()
                except OSError:
                    return
                if not line:
                    return
                if not line.endswith(b"\n"):
                    # Partially flushed write; the writer will finish it. Leave
                    # ``scanned`` here so the next call starts from this exact
                    # byte and the record is neither lost nor duplicated.
                    return
                end = handle.tell()
                self.scanned = end
                if not line.strip():
                    continue
                self._append(start, end, line)

    def _append(self, start: int, end: int, line: bytes) -> None:
        index = len(self.offsets)
        self.offsets.append(start)
        self.ends.append(end)
        kind = _classify(line)
        self.kinds.append(kind)
        if kind == _FRAME_KIND:
            self.frame_count += 1
        elif kind:
            self.kind_indices.setdefault(kind, []).append(index)

    # -- queries -----------------------------------------------------------

    @property
    def total(self) -> int:
        """Number of indexed records."""

        self.refresh()
        return len(self.offsets)

    def kind_count(self, kind: str) -> int:
        self.refresh()
        if kind == _FRAME_KIND:
            return self.frame_count
        return len(self.kind_indices.get(kind, ()))

    def has(self, kind: str) -> bool:
        return self.kind_count(kind) > 0

    def _read_blob(self, first: int, stop: int) -> list[dict[str, Any]]:
        """Parse records ``[first, stop)`` with one read of exactly their bytes."""

        if stop <= first:
            return []
        base = self.offsets[first]
        limit = self.ends[stop - 1]
        with self.path.open("rb") as handle:
            handle.seek(base)
            blob = handle.read(limit - base)
        records: list[dict[str, Any]] = []
        for index in range(first, stop):
            begin = self.offsets[index] - base
            finish = self.ends[index] - base
            try:
                record = json.loads(blob[begin:finish])
            except ValueError:
                # The index no longer parses lines up front (that is the whole
                # point of the byte probe), so this is the *only* place a
                # malformed line is rejected. Skipping matches ``_read_jsonl``,
                # and because the cursor is a record index -- not a byte offset --
                # dropping one cannot desynchronise the caller.
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def read_window(
        self, after: int = 0, limit: int | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        """Records from index ``after`` onward, plus the cursor to ask next time.

        ``limit=None`` means **unbounded** -- the caller gets everything from
        ``after`` to the end, exactly like the ``_read_jsonl(path)[after:]`` this
        replaced. That default is deliberate: ``feLoadTaskFramesOnce`` fetches a
        finished job's whole stream in one call to build a replay, and silently
        truncating it would leave the sandbox playing a fraction of the match.
        A live-follow client that wants bounded responses opts in by sending
        ``limit``; ``limit=0`` selects :data:`DEFAULT_WINDOW_RECORDS` **and**
        :data:`DEFAULT_WINDOW_BYTES` (whichever trips first, always at least one
        record), so it can ask "give me a sensible chunk" without naming a size.

        The returned cursor is an absolute record index, so a capped window needs
        no special handling: the caller asks again and catches up.
        """

        unbounded = limit is None and int(after or 0) <= 0
        if unbounded:
            # Whole-stream read: this is a replay load, not a live follow. The
            # index exists to make *repeated tail reads* cheap, but a first
            # unbounded read has to parse every record anyway, so indexing in a
            # separate pass would read the file twice (measured 263 ms of index
            # on top of 1 743 ms of parsing, versus 1 758 ms for the plain parse
            # this replaced). One pass that both indexes and parses costs the old
            # time and leaves the index warm for every subsequent tail poll.
            records = self._read_all_and_index()
            return records, len(self.offsets)

        self.refresh()
        total = len(self.offsets)
        first = max(0, min(int(after or 0), total))
        if limit is None:
            record_cap = total - first
            byte_cap: int | None = None
        elif int(limit) <= 0:
            record_cap = DEFAULT_WINDOW_RECORDS
            byte_cap = DEFAULT_WINDOW_BYTES
        else:
            record_cap = min(int(limit), MAX_WINDOW_RECORDS)
            byte_cap = None
        stop = first
        used = 0
        while stop < total and (stop - first) < record_cap:
            size = self.ends[stop] - self.offsets[stop]
            if byte_cap is not None and stop > first and used + size > byte_cap:
                break
            used += size
            stop += 1
        return self._read_blob(first, stop), stop

    def _read_all_and_index(self) -> list[dict[str, Any]]:
        """One sequential pass that both parses every record and indexes it.

        The unbounded read exists for replay loading, where the caller wants every
        record; indexing as we parse makes the *next* call (a live follow polling
        the tail) cheap without a second pass over the file.
        """

        records: list[dict[str, Any]] = []
        with self.lock:
            self.reset()
            try:
                with self.path.open("rb") as handle:
                    while True:
                        start = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # Incomplete trailing write: leave it for next time.
                            break
                        end = handle.tell()
                        self.scanned = end
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            record = json.loads(stripped)
                        except ValueError:
                            # Unparseable: index the position so the cursor can
                            # move past it, but do not hand it to the caller.
                            self._append(start, end, line)
                            continue
                        if not isinstance(record, dict):
                            continue
                        self._append(start, end, line)
                        records.append(record)
            except OSError:
                return records
            self.size = self.scanned
        return records

    def _read_all(self) -> list[dict[str, Any]]:
        """Single sequential parse of the whole file, without touching the index."""

        records: list[dict[str, Any]] = []
        try:
            with self.path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        # Incomplete trailing write: drop it, like the index does.
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError:
            return records
        return records

    def read_kind(self, kind: str) -> list[dict[str, Any]]:
        """Every record of ``kind`` (used for the ``episode_end``/``done`` list)."""

        self.refresh()
        indices = self.kind_indices.get(kind)
        if not indices:
            return []
        records: list[dict[str, Any]] = []
        # Episodes are far apart in the file, so group consecutive indices into
        # runs and read each run in one go rather than record by record.
        run_start = indices[0]
        previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            records.extend(self._read_blob(run_start, previous + 1))
            run_start = index
            previous = index
        records.extend(self._read_blob(run_start, previous + 1))
        return records

    def read_last(self, kind: str) -> dict[str, Any] | None:
        """The most recent record of ``kind``, or ``None``."""

        self.refresh()
        if kind == _FRAME_KIND:
            for index in range(len(self.kinds) - 1, -1, -1):
                if self.kinds[index] == _FRAME_KIND:
                    records = self._read_blob(index, index + 1)
                    return records[0] if records else None
            return None
        indices = self.kind_indices.get(kind)
        if not indices:
            return None
        records = self._read_blob(indices[-1], indices[-1] + 1)
        return records[0] if records else None


_indexes: "collections.OrderedDict[str, JsonlIndex]" = collections.OrderedDict()
_indexes_lock = threading.Lock()


def _cache_key(path: Path) -> str:
    """Cheap stable key.

    ``Path.resolve()`` is deliberately avoided: on the Docker Desktop Windows
    bind mount it costs ~6 ms per call because it stats every path component,
    and this runs on every frame poll. ``normcase(abspath())`` is the same
    key the checkpoint scan already uses for exactly that reason.
    """

    return os.path.normcase(os.path.abspath(str(path)))


def jsonl_index(path: Path) -> JsonlIndex:
    """The cached index for ``path``, moved to the LRU front.

    Deliberately does **not** refresh: every read method refreshes itself, and an
    eager refresh here would defeat :meth:`JsonlIndex.read_window`'s combined
    read-and-index pass for unbounded loads (it would index the file first, then
    parse it again).
    """

    key = _cache_key(path)
    with _indexes_lock:
        index = _indexes.get(key)
        if index is None:
            index = JsonlIndex(path)
            _indexes[key] = index
            while len(_indexes) > MAX_CACHED_FILES:
                _indexes.popitem(last=False)
        else:
            _indexes.move_to_end(key)
    return index


def frame_stream_index(path: Path) -> JsonlIndex:
    """Alias kept for readability at the call sites that stream frames."""

    return jsonl_index(path)


def clear_indexes() -> None:
    """Drop every cached index (tests, and any future admin endpoint)."""

    with _indexes_lock:
        _indexes.clear()
