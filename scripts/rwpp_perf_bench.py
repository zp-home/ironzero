# SPDX-License-Identifier: Apache-2.0
"""Measure the wall-clock cost of one RL decision on the *real* RWPP runtime.

One "decision" is ``DECISION_INTERVAL=10`` engine ticks: the bridge pumps ten
frames, sends one ``obs``, then ``Thread.sleep(frame_ms)`` after every frame. So
a decision costs ``10 * engine_frame_cost + 10 * sleep_cost``. This benchmark
separates those terms instead of guessing:

* ``--frame-ms 0``  -> no sleep at all, i.e. pure engine cost (upper bound of the
  speed a pacing fix could reach).
* ``--frame-ms 1``  -> 10 x ``Thread.sleep(1)``.
* ``--frame-ms 2``  -> 10 x ``Thread.sleep(2)``. If Windows sleeps in 15.6 ms
  timer ticks, these two are *nearly identical*; if the JVM actually honours
  millisecond sleeps, ``frame_ms=2`` costs ~10 ms/decision more than ``=1``.

It also measures the two cold-start phases apart, because they dominate an
episode's fixed cost:

* ``connect_s``  -- Popen -> bridge TCP accept (JVM boot + extension load).
* ``map_load_s`` -- TCP accept -> first ``obs`` (the bridge only loads the map
  after ``waitForClient()`` returns, so this whole gap *is* map loading).

Usage::

    python scripts/rwpp_perf_bench.py --frame-ms 1 --instances 1 --decisions 40
    python scripts/rwpp_perf_bench.py --frame-ms 2 --instances 4 --decisions 40
    python scripts/rwpp_perf_bench.py --frame-ms 0 --ticks 3600 --decisions 340

stdout is a single JSON document; a short human summary goes to stderr.

Exit codes: 0 = every instance produced the requested number of observations,
2 = the runtime launcher is missing, 1 = at least one instance timed out or
failed (the partial numbers are still printed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path(os.environ.get("IRONZERO_RWPP_RUNTIME", r"E:\IronZeroBuild\rwpp-runtime-1.15"))
DEFAULT_JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))
DEFAULT_MAP = "maps/skirmish/[p2]Lake (2p).tmx"
OUT_DIR = REPO / "runs" / "_rwpp_perf"


# ----------------------------------------------------------------------------
# small statistics helpers (no numpy dependency in this repo's scripts)
# ----------------------------------------------------------------------------
def episode_point(message: dict) -> dict:
    """Compact, comparable snapshot of one observation.

    Two runs at different ``frame_ms`` must agree on every number here if
    ``frame_ms`` only changes the wall clock. ``own_hp``/``enemy_hp`` are the
    sensitive term: they integrate every damage tick, so a delta change shows up
    immediately.
    """
    state = message.get("global_state") or {}
    return {
        "tick": message.get("tick"),
        "own_alive": state.get("own_alive"),
        "enemy_alive": state.get("enemy_alive"),
        "own_hp": state.get("own_hp"),
        "enemy_hp": state.get("enemy_hp"),
        "funds": state.get("funds"),
        "unit_count": state.get("unit_count"),
        "result": message.get("result"),
    }


def scripted_step(obs: dict, decision: int) -> dict:
    """Deterministic, frame_ms-independent order: one attack-move, always.

    The point of the episode mode is to make the *engine* the only source of
    difference between two runs, so the command stream is a pure function of the
    decision index -- never of the observation contents.
    """
    tick = int(obs.get("tick") or 0)
    # Walk a fixed circuit of the map so the units keep moving and shooting; a
    # "hold" order would leave most of the battle to the AI and hide timing bugs.
    waypoints = [
        (1400.0, 1250.0),
        (1900.0, 900.0),
        (2100.0, 1600.0),
        (1500.0, 1900.0),
        (900.0, 1300.0),
    ]
    x, y = waypoints[(decision // 15) % len(waypoints)]
    return {
        "protocol": "ironzero/1",
        "type": "step",
        "message_id": f"m-bench-{decision}",
        "episode_id": "rwpp-perf-bench",
        "tick": tick,
        "decision_id": f"d-bench-{decision}",
        "actions": [{"kind": "attack_move", "unit_group": 4, "x": x, "y": y}],
    }


def trace_digest(trace: list[dict]) -> dict:
    """Summarise an episode trace and hash its physics-relevant columns."""
    if not trace:
        return {"points": 0}
    last = trace[-1]
    # A digest over every sampled tick: equal digests mean identical simulation.
    blob = ";".join(
        f"{p['tick']}:{p['own_alive']}:{p['enemy_alive']}:{p['own_hp']}:{p['enemy_hp']}"
        for p in trace
    )
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    hps = [p["own_hp"] for p in trace if isinstance(p["own_hp"], (int, float))]
    enemy_hps = [p["enemy_hp"] for p in trace if isinstance(p["enemy_hp"], (int, float))]
    return {
        "points": len(trace),
        "final": last,
        "digest": digest,
        "own_hp_min": round(min(hps), 2) if hps else None,
        "enemy_hp_min": round(min(enemy_hps), 2) if enemy_hps else None,
    }


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


# ----------------------------------------------------------------------------
# optional java-process sampling (RSS / cumulative CPU)
# ----------------------------------------------------------------------------
# Hand-rolled JSON: Windows PowerShell 5.1's ConvertTo-Json has no -AsArray and
# unwraps a single-element array, either of which breaks a strict parser.
_SAMPLE_PS = (
    "$a=@(Get-Process java -ErrorAction SilentlyContinue | ForEach-Object {"
    " '{\"Id\":' + $_.Id + ',\"WorkingSet64\":' + $_.WorkingSet64 + ',\"CPU\":'"
    " + $(if ($null -ne $_.CPU) { $_.CPU } else { 0 }) + '}' });"
    " '[' + ($a -join ',') + ']'"
)


def _powershell_executable() -> str | None:
    for candidate in ("pwsh", "powershell"):
        found = None
        for directory in (os.environ.get("PATH") or "").split(os.pathsep):
            if not directory:
                continue
            for name in (candidate + ".exe", candidate):
                path = Path(directory) / name
                if path.is_file():
                    found = str(path)
                    break
            if found:
                break
        if found:
            return found
    return None


def sample_java_processes() -> list[dict] | None:
    """Return ``[{id, rss_mb, cpu_s}, ...]`` for every running java process.

    Uses PowerShell because the repo scripts deliberately avoid third-party
    dependencies (no psutil). Returns ``None`` when sampling is unavailable.
    """
    shell = _powershell_executable()
    if shell is None:
        return None
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", _SAMPLE_PS],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rss = item.get("WorkingSet64")
        rows.append(
            {
                "id": item.get("Id"),
                "rss_mb": round(float(rss) / (1 << 20), 1) if rss is not None else None,
                "cpu_s": round(float(item.get("CPU") or 0.0), 2),
            }
        )
    return rows


class MemorySampler(threading.Thread):
    """Poll ``Get-Process java`` every ``interval`` s until ``stop`` is set."""

    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True)
        self.interval = max(0.2, interval)
        self._halt = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:  # pragma: no cover - thread body
        while not self._halt.is_set():
            rows = sample_java_processes()
            if rows is not None:
                self.samples.append({"t": time.time(), "procs": rows})
            self._halt.wait(self.interval)

    def snapshot(self) -> dict:
        samples = [s for s in self.samples if s["procs"]]
        if not samples:
            return {"available": False, "samples": len(self.samples)}
        peak = max(samples, key=lambda s: sum(p["rss_mb"] or 0.0 for p in s["procs"]))
        first, last = samples[0], samples[-1]
        first_cpu = sum(p["cpu_s"] or 0.0 for p in first["procs"])
        last_cpu = sum(p["cpu_s"] or 0.0 for p in last["procs"])
        wall = max(1e-6, last["t"] - first["t"])
        peak_rss = [p["rss_mb"] or 0.0 for p in peak["procs"]]
        return {
            "available": True,
            "samples": len(samples),
            "max_procs_seen": max(len(s["procs"]) for s in samples),
            "peak_proc_count": len(peak["procs"]),
            "peak_total_rss_mb": round(sum(peak_rss), 1),
            "peak_rss_mb_per_proc": sorted(round(v, 1) for v in peak_rss),
            "mean_rss_mb_per_proc": (
                round(sum(sum(p["rss_mb"] or 0.0 for p in s["procs"]) / len(s["procs"])
                          for s in samples) / len(samples), 1)
            ),
            "cpu_seconds_delta": round(last_cpu - first_cpu, 2),
            "cpu_cores_busy": round(max(0.0, last_cpu - first_cpu) / wall, 2),
        }

    def stop(self) -> None:
        self._halt.set()


class LogWatcher(threading.Thread):
    """Tail the game log and stamp the ``[ironzero]`` boot markers.

    The bridge prints ``bridge extension loaded`` -> ``bridge listening`` ->
    ``platform connected`` -> ``world ready`` in that order, which is exactly the
    JVM/extension vs. map-load split. The lines themselves carry no timestamp, so
    they are stamped on the wall clock as they appear.
    """

    MARKERS = (
        ("bridge_loaded", "[ironzero] bridge extension loaded"),
        ("bridge_listening", "[ironzero] bridge listening"),
        ("platform_connected", "[ironzero] platform connected"),
        ("world_ready", "[ironzero] world ready"),
        ("bridge_finished", "[ironzero] bridge finished"),
    )
    MAX_POLLS = 40 * 60 * 30  # ~30 min at 25 ms

    def __init__(self, path: Path, started: float) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.started = started
        self._halt = threading.Event()
        self.marks: dict[str, float] = {}
        self.error: str | None = None

    def run(self) -> None:  # pragma: no cover - thread body
        offset = 0
        pending = ""
        polls = 0
        while not self._halt.is_set() and polls < self.MAX_POLLS:
            polls += 1
            try:
                with self.path.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset = handle.tell()
            except OSError as exc:  # file may not exist yet
                self.error = f"{type(exc).__name__}: {exc}"
                chunk = b""
            if chunk:
                now = time.perf_counter()
                pending += chunk.decode("utf-8", errors="replace")
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    for key, needle in self.MARKERS:
                        if key not in self.marks and needle in line:
                            self.marks[key] = round(now - self.started, 3)
            if len(self.marks) == len(self.MARKERS):
                return
            self._halt.wait(0.025)

    def stop(self) -> None:
        self._halt.set()


# ----------------------------------------------------------------------------
# one game instance
# ----------------------------------------------------------------------------
class Benchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = Path(args.runtime)
        self.java_home = Path(args.java_home)
        self.ts = time.strftime("%Y%m%d-%H%M%S")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- process ------------------------------------------------------------
    def _spawn(self, port: int) -> tuple[subprocess.Popen, object, Path, float]:
        log_path = OUT_DIR / f"game-{port}-{self.ts}.log"
        log = log_path.open("w", encoding="utf-8", errors="replace")
        env = dict(os.environ)
        env["JAVA_HOME"] = str(self.java_home)
        env["PATH"] = str(self.java_home / "bin") + os.pathsep + env.get("PATH", "")
        command = [
            str(self.runtime / "bin" / "rwpp-server.bat"),
            "--no-console",
            "--",
            "--bridge-port",
            str(port),
            "--ticks",
            str(self.args.ticks),
            "--map",
            self.args.map,
            "--frame-ms",
            str(self.args.frame_ms),
        ]
        if self.args.unlock_ai:
            command.append("--unlock-ai")
        if self.args.win_mode:
            command.extend(["--win-mode", self.args.win_mode])
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=str(self.runtime),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return process, log, log_path, started

    def _connect(self, port: int, deadline: float) -> tuple[socket.socket | None, str]:
        last_error = "connect timeout"
        while time.perf_counter() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock, ""
            except OSError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.02)
        return None, last_error

    # -- measurement --------------------------------------------------------
    def measure(self, index: int) -> dict:
        args = self.args
        if args.stagger_s > 0:
            # Staggered mode: instance i starts its JVM i*stagger_s later than the
            # first one, which tells apart "N JVMs booting at once breaks" from
            # "N JVMs running at once is slow".
            time.sleep(index * args.stagger_s)
        port = args.port + index
        result: dict = {
            "index": index,
            "port": port,
            "ok": False,
            "requested_decisions": args.decisions,
            "decisions": 0,
            "obs_read": 0,
            "sock_timeout_s": 0.0,
            "intervals_ms": [],
        }
        process = log = None
        sock = None
        watcher = None
        notes: list[str] = []
        try:
            process, log, log_path, started = self._spawn(port)
            watcher = LogWatcher(log_path, started)
            watcher.start()
            result["log"] = str(log_path.relative_to(REPO)) if log_path.is_relative_to(REPO) else str(log_path)
            deadline = time.perf_counter() + args.timeout
            sock, connect_error = self._connect(port, deadline)
            connected = time.perf_counter()
            result["connect_s"] = round(connected - started, 3)
            if sock is None:
                result["error"] = f"bridge port never accepted ({connect_error})"
                return result

            # Every gap below is measured from the *previous* event, anchored at
            # the spawn timestamp, so the phases stay additive.
            sock.settimeout(max(30.0, args.timeout))
            buffer = b""
            obs_times: list[float] = []
            obs_ticks: list[int] = []
            # Episode mode keeps a compact per-decision trace of the world state so
            # two frame_ms values can be compared state-for-state, not just by
            # their final result.
            trace: list[dict] = []
            done_message: dict | None = None
            done = False
            reply_every = max(1, args.reply_every)
            while not done and time.perf_counter() < deadline:
                if args.episode:
                    if args.max_decisions > 0 and len(obs_times) >= args.max_decisions:
                        break
                elif len(obs_times) >= args.decisions + 1:
                    break
                try:
                    chunk = sock.recv(1 << 16)
                except socket.timeout:
                    notes.append(f"read timed out after {len(obs_times)} obs")
                    break
                except OSError as exc:
                    notes.append(f"socket error: {type(exc).__name__}: {exc}")
                    break
                if not chunk:
                    notes.append(f"bridge closed after {len(obs_times)} obs")
                    break
                now = time.perf_counter()
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        notes.append("unparseable bridge line")
                        continue
                    kind = message.get("type")
                    if kind == "obs":
                        obs_times.append(now)
                        tick = message.get("tick")
                        obs_ticks.append(int(tick) if isinstance(tick, (int, float)) else -1)
                        if args.episode:
                            trace.append(episode_point(message))
                            # Drive the episode on a fixed schedule so every
                            # frame_ms value sees the *same* commands at the
                            # *same* ticks -- otherwise any difference in the
                            # outcome could just be a different policy.
                            if len(obs_times) % reply_every == 0 and not args.no_actions:
                                try:
                                    sock.sendall(
                                        (json.dumps(scripted_step(message, len(obs_times)))
                                         + "\n").encode("utf-8")
                                    )
                                except OSError as exc:
                                    notes.append(f"send failed: {type(exc).__name__}: {exc}")
                                    done = True
                                    break
                        elif len(obs_times) >= args.decisions + 1:
                            break
                    elif kind == "done":
                        done_message = message
                        done = True
                        break
                    elif kind == "hello":
                        result["hello_s"] = round(now - started, 3)

            result["obs_read"] = len(obs_times)
            if not obs_times:
                result["error"] = "no observation received"
                result["notes"] = notes
                return result

            first_obs = obs_times[0]
            result["first_obs_s"] = round(first_obs - started, 3)
            result["map_load_s"] = round(first_obs - connected, 3)
            result["t_first_obs_rel_connect_s"] = result["map_load_s"]

            if len(obs_times) < 2:
                result["error"] = "only one observation; no interval to measure"
                result["notes"] = notes
                return result

            intervals_ms = [(b - a) * 1000.0 for a, b in zip(obs_times, obs_times[1:])]
            decisions = len(intervals_ms)
            steady = obs_times[-1] - obs_times[0]
            result.update(
                {
                    "ok": decisions >= (args.decisions if not args.episode else 1),
                    "decisions": decisions,
                    "last_obs_s": round(obs_times[-1] - started, 3),
                    "total_s": round(obs_times[-1] - started, 3),
                    "steady_s": round(steady, 3),
                    "ms_per_decision": round(steady / decisions * 1000.0, 2),
                    "intervals_ms": [round(v, 2) for v in intervals_ms],
                    "obs_interval_ms": {
                        "p50": round(percentile(intervals_ms, 50) or 0.0, 2),
                        "p95": round(percentile(intervals_ms, 95) or 0.0, 2),
                        "max": round(max(intervals_ms), 2),
                        "mean": round(mean(intervals_ms) or 0.0, 2),
                        "min": round(min(intervals_ms), 2),
                    },
                }
            )

            if args.episode:
                # What ended the episode, and did the world evolve identically?
                result["episode"] = trace_digest(trace)
                final = trace[-1] if trace else {}
                result["episode"]["result"] = final.get("result")
                result["episode"]["end_tick"] = final.get("tick")
                result["episode"]["done_message"] = done_message
                result["episode"]["wall_s"] = result["total_s"]
                result["episode"]["ticks"] = final.get("tick")
                result["episode"]["tick_per_s"] = (
                    round(final["tick"] / result["total_s"], 2)
                    if final.get("tick") and result["total_s"] > 0 else None
                )
                if args.trace_out:
                    trace_path = Path(args.trace_out)
                    if args.instances > 1:
                        trace_path = trace_path.with_name(
                            f"{trace_path.stem}.{index}{trace_path.suffix}"
                        )
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        json.dumps({"frame_ms": args.frame_ms, "port": port, "trace": trace},
                                   ensure_ascii=False),
                        encoding="utf-8",
                    )
                    result["episode"]["trace_path"] = str(trace_path)

            # Did the per-decision cost drift upwards as the world filled up?
            window = max(3, len(intervals_ms) // 4)
            head = intervals_ms[:window]
            tail = intervals_ms[-window:]
            result["drift"] = {
                "head_mean_ms": round(mean(head) or 0.0, 2),
                "tail_mean_ms": round(mean(tail) or 0.0, 2),
                "delta_ms": round((mean(tail) or 0.0) - (mean(head) or 0.0), 2),
                "window": window,
            }

            tick_deltas = [b - a for a, b in zip(obs_ticks, obs_ticks[1:])]
            real_ticks = sum(d for d in tick_deltas if d > 0)
            result["tick_first"] = obs_ticks[0]
            result["tick_last"] = obs_ticks[-1]
            result["ticks_measured"] = real_ticks
            result["expected_ticks"] = 10 * decisions
            result["ticks_per_s"] = round(real_ticks / steady, 2) if steady > 0 else None
            if 0 < real_ticks and real_ticks != 10 * decisions:
                notes.append(f"tick delta {real_ticks} != 10x{decisions} (bridge skipped/merged obs)")

            # Per-decision engine-vs-sleep split: engine cost is whichever part of
            # the interval is not the pacing sleep.
            if args.frame_ms > 0:
                pacing_ms = 10.0 * args.frame_ms
                result["pacing_budget_ms"] = pacing_ms
                result["engine_ms_per_decision"] = round(result["ms_per_decision"] - pacing_ms, 2)
                result["sleep_overhead_per_frame_ms"] = round(
                    (result["ms_per_decision"] - pacing_ms) / 10.0, 3
                )
            else:
                result["pacing_budget_ms"] = 0.0
                result["engine_ms_per_decision"] = result["ms_per_decision"]

            result["notes"] = notes
            return result
        except Exception as exc:  # noqa: BLE001 - never lose the partial numbers
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["notes"] = notes
            return result
        finally:
            if watcher is not None:
                watcher.stop()
            if sock is not None:
                try:
                    sock.sendall(b'{"type":"shutdown"}\n')
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            if process is not None:
                try:
                    code = process.wait(timeout=60)
                    result["process_exit_code"] = code
                except subprocess.TimeoutExpired:
                    process.kill()
                    result["process_exit_code"] = None
                    result.setdefault("notes", []).append("process killed after 60s shutdown wait")
            if log is not None:
                log.close()
            if watcher is not None:
                watcher.stop()
                watcher.join(timeout=5)
                result["boot_marks_s"] = dict(watcher.marks)
                marks = watcher.marks
                listening = marks.get("bridge_listening")
                connected = marks.get("platform_connected")
                ready = marks.get("world_ready")
                if listening is not None:
                    result["jvm_boot_s"] = listening
                if listening is not None and connected is not None:
                    result["accept_wait_s"] = round(connected - listening, 3)
                if connected is not None and ready is not None:
                    result["log_map_load_s"] = round(ready - connected, 3)

    # -- run ----------------------------------------------------------------
    def run(self) -> dict:
        args = self.args
        launcher = self.runtime / "bin" / "rwpp-server.bat"
        if not launcher.is_file():
            print(f"runtime launcher missing: {launcher}", file=sys.stderr)
            return {"error": f"runtime launcher missing: {launcher}"}

        sampler = None
        if args.mem_sample:
            sampler = MemorySampler(args.mem_sample_interval)
            sampler.start()

        results: list[dict | None] = [None] * args.instances
        gate = threading.Barrier(args.instances)

        def worker(index: int) -> None:
            try:
                gate.wait(timeout=120)
            except threading.BrokenBarrierError:
                pass
            try:
                results[index] = self.measure(index)
            except Exception as exc:  # noqa: BLE001 - one bad lane must not kill the run
                results[index] = {
                    "index": index,
                    "port": args.port + index,
                    "ok": False,
                    "decisions": 0,
                    "error": f"worker crashed: {type(exc).__name__}: {exc}",
                }

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"bench-{i}", daemon=True)
            for i in range(args.instances)
        ]
        wall_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_s = time.perf_counter() - wall_start

        if sampler is not None:
            sampler.stop()
            sampler.join(timeout=30)

        complete = [r for r in results if r]
        ok = [r for r in complete if r.get("ok")]
        total_decisions = sum(r.get("decisions") or 0 for r in complete)
        active_windows = [
            (r["first_obs_s"], r["last_obs_s"])
            for r in complete
            if r.get("first_obs_s") is not None and r.get("last_obs_s") is not None
        ]
        active_s = 0.0
        if len(active_windows) > 1:
            active_s = max(w[1] for w in active_windows) - min(w[0] for w in active_windows)
        payload = {
            "schema": "ironzero/rwpp-perf-bench/1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "params": {
                "frame_ms": args.frame_ms,
                "instances": args.instances,
                "decisions": args.decisions,
                "ticks": args.ticks,
                "map": args.map,
                "win_mode": args.win_mode,
                "unlock_ai": bool(args.unlock_ai),
                "base_port": args.port,
                "runtime": str(self.runtime),
                "java_home": str(self.java_home),
            },
            "instances": complete,
            "summary": {
                "instances_requested": args.instances,
                "instances_ok": len(ok),
                "instances_failed": args.instances - len(ok),
                "wall_s": round(wall_s, 2),
                "total_decisions": total_decisions,
                "decisions_per_s_wall": round(total_decisions / wall_s, 3) if wall_s > 0 else None,
                "decisions_per_s_active": (
                    round(total_decisions / active_s, 3) if active_s > 0 else None
                ),
                "active_window_s": round(active_s, 2),
                "ms_per_decision_median": percentile(
                    [r["ms_per_decision"] for r in ok], 50
                ),
                "ms_per_decision_max": max((r["ms_per_decision"] for r in ok), default=None),
                "connect_s_median": percentile([r.get("connect_s") or 0.0 for r in ok], 50),
                "map_load_s_median": percentile([r.get("map_load_s") or 0.0 for r in ok], 50),
            },
        }
        if sampler is not None:
            payload["memory"] = sampler.snapshot()
        return payload


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real-runtime RWPP decision cost (engine + pacing sleep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5400,
                        help="base bridge port; instance i uses port+i")
    parser.add_argument("--frame-ms", type=int, default=2,
                        help="bridge --frame-ms (sleep after every simulated frame)")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--decisions", type=int, default=40,
                        help="steady-state observations to time after the first one")
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--ticks", type=int, default=0,
                        help="bridge --ticks (frame cap); 0 = derive from --decisions")
    parser.add_argument("--episode-ticks", type=int, default=3600,
                        help="in --episode mode, the bridge --ticks horizon")
    parser.add_argument("--unlock-ai", action="store_true")
    parser.add_argument("--win-mode", default="", choices=["", "native", "annihilation"],
                        help="bridge --win-mode passthrough")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-instance deadline in seconds")
    parser.add_argument("--stagger-s", type=float, default=0.0,
                        help="start instance i this many seconds after instance 0 "
                             "(0 = all JVMs boot simultaneously)")
    parser.add_argument("--episode", action="store_true",
                        help="drive a full episode with a fixed scripted policy and "
                             "record the state trace (for frame_ms A/B comparison)")
    parser.add_argument("--reply-every", type=int, default=1,
                        help="in --episode mode, send a step every Nth observation")
    parser.add_argument("--max-decisions", type=int, default=0,
                        help="in --episode mode, stop after this many observations "
                             "(0 = run until done/ticks)")
    parser.add_argument("--trace-out", default="",
                        help="in --episode mode, write the per-decision trace here")
    parser.add_argument("--no-actions", action="store_true",
                        help="in --episode mode, never send a step: isolates whether "
                             "frame_ms changes the physics from whether it changes "
                             "when client commands land")
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--mem-sample", action=argparse.BooleanOptionalAction, default=True,
                        help="sample java RSS/CPU via Get-Process while instances>1")
    parser.add_argument("--mem-sample-interval", type=float, default=2.0)
    parser.add_argument("--out", default="", help="also write the JSON here")
    return parser


def human_summary(payload: dict) -> str:
    params = payload.get("params", {})
    summary = payload.get("summary", {})
    lines = [
        f"map={params.get('map')} frame_ms={params.get('frame_ms')} "
        f"instances={params.get('instances')} decisions={params.get('decisions')} "
        f"ticks={params.get('ticks')}",
        f"{'idx':>3} {'port':>5} {'jvm':>7} {'mapload':>8} {'first_obs':>9} "
        f"{'total':>7} {'ms/dec':>7} {'ticks/s':>8} {'p50':>7} {'p95':>7} {'max':>7}  status",
    ]
    for row in payload.get("instances", []):
        lines.append(
            f"{row.get('index'):>3} {row.get('port'):>5} "
            f"{row.get('jvm_boot_s') or row.get('connect_s') or 0:>7.3f} "
            f"{row.get('map_load_s') or 0:>8.3f} "
            f"{row.get('first_obs_s') or 0:>9.3f} {row.get('total_s') or 0:>7.2f} "
            f"{row.get('ms_per_decision') or 0:>7.2f} {row.get('ticks_per_s') or 0:>8.2f} "
            f"{(row.get('obs_interval_ms') or {}).get('p50', 0):>7.1f} "
            f"{(row.get('obs_interval_ms') or {}).get('p95', 0):>7.1f} "
            f"{(row.get('obs_interval_ms') or {}).get('max', 0):>7.1f}  "
            f"{'ok' if row.get('ok') else 'FAIL ' + str(row.get('error') or row.get('notes'))}"
        )
    memory = payload.get("memory") or {}
    lines.append(
        f"wall={summary.get('wall_s')}s total_decisions={summary.get('total_decisions')} "
        f"dec/s(wall)={summary.get('decisions_per_s_wall')} "
        f"dec/s(active)={summary.get('decisions_per_s_active')} "
        f"failed={summary.get('instances_failed')}"
    )
    if memory.get("available"):
        lines.append(
            f"java procs={memory.get('peak_proc_count')} "
            f"rss_total={memory.get('peak_total_rss_mb')}MB "
            f"rss/proc={memory.get('peak_rss_mb_per_proc')} "
            f"cpu={memory.get('cpu_cores_busy')} cores"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ticks <= 0:
        # +1 obs is the baseline, each decision is 10 ticks, plus slack for the
        # map-load warm-up frames the bridge burns before the first decision.
        args.ticks = (args.decisions + 8) * 10 + 400
    if args.episode:
        # A full episode needs the whole horizon, not just the timed window.
        args.ticks = args.episode_ticks
    if not (1 <= args.instances <= 32):
        print("--instances must be within 1..32", file=sys.stderr)
        return 2

    payload = Benchmark(args).run()
    if "error" in payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(human_summary(payload), file=sys.stderr)
    summary = payload.get("summary") or {}
    return 0 if summary.get("instances_failed") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
