"""Benchmark the frame-stream read path against the old whole-file parse.

Point the dashboard's frame route at the largest real ``frames.jsonl`` on this
machine and time the polls it actually performs, comparing each one with
``_read_jsonl(path)[after:]`` -- the implementation that was replaced. Every
comparison also asserts the two produce identical records, so this is a pure
speed measurement, not a behaviour change.

The call has to share the run tree with the real backend, so run it from the
repository root with the platform dependencies installed::

    python tools/bench_frame_stream.py

Symptoms this measures: sandbox playback degrading over a long inference run.
The old read cost 2 s per poll on a 154 MB stream and was paid every 700 ms per
concurrent instance; the indexed read is a few milliseconds and a single ``stat``
when nothing was appended.
"""
from __future__ import annotations

import concurrent.futures
import pathlib
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from web.backend.app import app  # noqa: E402
from web.backend.app.jobs import _read_jsonl  # noqa: E402
from web.backend.app.jsonl_index import clear_indexes, jsonl_index  # noqa: E402

streams = [p for p in REPO.glob("runs/platform/rwpp-live-*/frames.jsonl") if p.is_file()]
if not streams:
    raise SystemExit(
        "No live frame stream found under runs/platform/rwpp-live-*/frames.jsonl. "
        "Run one real-game inference first, then re-run this benchmark."
    )
job_dir = max((p.parent for p in streams), key=lambda p: (p / "frames.jsonl").stat().st_size)
frames_path = job_dir / "frames.jsonl"
size = frames_path.stat().st_size
print(f"job    : {job_dir.name}  {size / 1024 / 1024:.1f} MB")

clear_indexes()
client = TestClient(app)

# 1. Warm-up: one default poll returns the whole tail and builds the index.
t0 = time.perf_counter()
page = client.get(f"/api/infer/{job_dir.name}/frames", params={"after": 0}).json()
warm_ms = (time.perf_counter() - t0) * 1000
total = page["next"]
print(f"warmup : {warm_ms:8.0f} ms  -> {total} records, whole tail in one response "
      f"(index built once)")

# 2. Steady-state tail polls, the pattern a live follow produces every 700 ms.
tail_cursor = max(0, total - 1)
new_times = []
old_times = []
for _ in range(5):
    t0 = time.perf_counter()
    payload = client.get(
        f"/api/infer/{job_dir.name}/frames", params={"after": tail_cursor}
    ).json()
    new_times.append((time.perf_counter() - t0) * 1000)
    assert payload["next"] == total
    # The old path, measured on the same cursor.
    t0 = time.perf_counter()
    old_slice = _read_jsonl(frames_path)[tail_cursor:]
    old_times.append((time.perf_counter() - t0) * 1000)
    assert old_slice == payload["frames"]

print(f"tail   : route {statistics.median(new_times):7.1f} ms  "
      f"vs old {statistics.median(old_times):7.0f} ms  "
      f"({statistics.median(old_times) / statistics.median(new_times):.0f}x)")

# 3. The status endpoint the same page polls alongside it.
#    (Only for jobs with a meta.json -- the rwpp-live dirs are created by the
#    host recorder and carry none, so /api/infer 404s on them by design.)
t0 = time.perf_counter()
status_response = client.get(f"/api/infer/{job_dir.name}")
status_ms = (time.perf_counter() - t0) * 1000
if status_response.status_code == 200:
    status = status_response.json()
    t0 = time.perf_counter()
    stream = _read_jsonl(frames_path)
    old_status = {
        "frames_total": len([i for i in stream if i.get("type") == "frame"]),
        "latest_frame": [i for i in stream if i.get("type") == "frame"][-1],
        "done": any(i.get("type") == "done" for i in stream),
    }
    old_status_ms = (time.perf_counter() - t0) * 1000
    assert status["frames_total"] == old_status["frames_total"]
    assert status["latest_frame"] == old_status["latest_frame"]
    assert status["done"] == old_status["done"]
    print(f"status : route {status_ms:7.1f} ms  vs old {old_status_ms:7.0f} ms  "
          f"({old_status_ms / status_ms:.0f}x, same values)")
else:
    # No meta.json here; measure the index-backed pieces the route uses directly.
    t0 = time.perf_counter()
    index = jsonl_index(frames_path)
    counted = index.kind_count("frame")
    latest = index.read_last("frame")
    done = index.has("done")
    parts_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    stream = _read_jsonl(frames_path)
    old_status = {
        "frames_total": len([i for i in stream if i.get("type") == "frame"]),
        "latest_frame": [i for i in stream if i.get("type") == "frame"][-1],
        "done": any(i.get("type") == "done" for i in stream),
    }
    old_status_ms = (time.perf_counter() - t0) * 1000
    assert counted == old_status["frames_total"]
    assert latest == old_status["latest_frame"]
    assert done == old_status["done"]
    print(f"status : parts {parts_ms:7.1f} ms  vs old {old_status_ms:7.0f} ms  "
          f"({old_status_ms / parts_ms:.0f}x, same values)  [no meta.json in this job]")

# 4. Ten concurrent instances, i.e. the 10-slot service ceiling.

def poll() -> float:
    t0 = time.perf_counter()
    client.get(f"/api/infer/{job_dir.name}/frames", params={"after": tail_cursor})
    return (time.perf_counter() - t0) * 1000

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    concurrent_times = list(pool.map(lambda _: poll(), range(10)))
print(f"x10    : {statistics.median(concurrent_times):7.1f} ms median, "
      f"{max(concurrent_times):7.1f} ms worst")
print(f"         (old path would be ~{statistics.median(old_times) * 10 / 1000:.1f} s of "
      f"serialised parsing per dashboard tick at 10 instances)")
