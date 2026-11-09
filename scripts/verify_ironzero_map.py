#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the IronZero25 25v25 arena map on the real headless RWPP 1.15 runtime.

What it proves (stage-1 acceptance for ``task-1``):

1. the generated ``IronZero25.tmx`` loads in the real game,
2. the first observation carries exactly 25 own tanks (team 0) and 25 enemy
   tanks (team 1), all non-building units with ``max_hp == 210``,
3. the spawn coordinates match ``configs/maps/IronZero25.meta.json`` and the
   enemy block is the 180-degree rotation of the player block,
4. the simulation advances (>= ``--min-ticks`` ticks, monotonic ticks) without
   crashing or losing units (battle is too far away to start that early).

Windows note: the workspace path contains CJK characters and Java/RWML's ASM
self-attach fails on non-ASCII runtime paths, so this script runs the runtime
from an ASCII mirror (``%TEMP%\\ironzero-rwpp-runtime-1.15`` by default) and
copies the map into that mirror before launching:

    python scripts/verify_ironzero_map.py
    python scripts/verify_ironzero_map.py --runtime "<ascii-runtime>" --ticks 300

The bridge jar is expected to support ``--win-mode annihilation`` (25v25 maps
have no buildings, so the native "no factory = defeated" bookkeeping would end
the episode on tick ~1).  Use ``--win-mode native`` to mirror the plain command
from the task description instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = REPO_ROOT / ".runtime" / "rwpp-runtime-1.15"
DEFAULT_MAP_SOURCE = REPO_ROOT / "configs" / "maps" / "IronZero25.tmx"
DEFAULT_META = REPO_ROOT / "configs" / "maps" / "IronZero25.meta.json"
DEFAULT_REPORT = REPO_ROOT / "configs" / "maps" / "IronZero25.verify.json"
DEFAULT_LOG = REPO_ROOT / "runs" / "_ironzero25_verify_server.log"
MAP_RELATIVE = "maps/ironzero/IronZero25.tmx"
TANK_MAX_HP = 210.0
TANK_HP_TOLERANCE = 1.0
SPAWN_TOLERANCE_PX = 15.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the IronZero25 arena on the real RWPP runtime")
    parser.add_argument("--runtime", default="", help="RWPP runtime directory (ASCII path recommended)")
    parser.add_argument("--java-home", default=r"E:\jdk25copy")
    parser.add_argument("--port", type=int, default=5401)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--frame-ms", type=int, default=2)
    parser.add_argument("--min-ticks", type=int, default=120)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="do not stop at --min-ticks; run until the bridge sends done (full --ticks horizon)",
    )
    parser.add_argument("--connect-timeout", type=float, default=90.0)
    parser.add_argument("--obs-timeout", type=float, default=120.0)
    parser.add_argument("--win-mode", choices=["annihilation", "native"], default="annihilation")
    parser.add_argument("--no-unlock-ai", action="store_true", help="do not pass --unlock-ai")
    parser.add_argument("--map-source", default=str(DEFAULT_MAP_SOURCE))
    parser.add_argument("--map-relative", default=MAP_RELATIVE)
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    return parser


def resolve_runtime(explicit: str) -> tuple[Path, str]:
    """Pick the runtime directory; prefer the ASCII temp mirror when present."""

    candidates: list[tuple[Path, str]] = []
    if explicit:
        candidates.append((Path(explicit), "explicit --runtime"))
    env_runtime = os.environ.get("IRONZERO_ASCII_RUNTIME", "")
    if env_runtime:
        candidates.append((Path(env_runtime), "IRONZERO_ASCII_RUNTIME"))
    candidates.append(
        (Path(tempfile.gettempdir()) / "ironzero-rwpp-runtime-1.15", "%TEMP% ASCII mirror")
    )
    candidates.append((DEFAULT_RUNTIME, "workspace .runtime copy"))
    for path, origin in candidates:
        if (path / "bin" / "rwpp-server.bat").is_file():
            return path, origin
    raise SystemExit(
        "no RWPP runtime found; pass --runtime or create %TEMP%\\ironzero-rwpp-runtime-1.15"
    )


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def install_map(runtime: Path, source: Path, relative: str) -> Path:
    destination = runtime / "assets" / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copyfile(source, destination)
    if not destination.is_file():
        raise SystemExit(f"arena map missing at {destination} (and no source at {source})")
    return destination


def launch_server(
    args: argparse.Namespace, runtime: Path, log_path: Path
) -> tuple[subprocess.Popen, Any]:
    env = dict(os.environ)
    if Path(args.java_home).is_dir():
        env["JAVA_HOME"] = args.java_home
        env["PATH"] = str(Path(args.java_home) / "bin") + os.pathsep + env.get("PATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    command = [
        str(runtime / "bin" / "rwpp-server.bat"),
        "--no-console",
        "--",
        "--bridge-port",
        str(args.port),
        "--ticks",
        str(args.ticks),
        "--map",
        args.map_relative,
        "--frame-ms",
        str(args.frame_ms),
        "--win-mode",
        args.win_mode,
    ]
    if not args.no_unlock_ai:
        command.append("--unlock-ai")
    process = subprocess.Popen(
        command,
        cwd=str(runtime),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def connect(port: int, timeout: float, process: subprocess.Popen) -> socket.socket:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before the bridge port opened (exit={process.returncode})")
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=2)
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"bridge port {port} never opened within {timeout:.0f}s")


def read_messages(
    sock: socket.socket,
    process: subprocess.Popen,
    args: argparse.Namespace,
    report: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    sock.settimeout(5.0)
    buffer = b""
    observations: list[dict[str, Any]] = []
    hello: dict[str, Any] | None = None
    done_message: dict[str, Any] | None = None
    deadline = time.time() + args.obs_timeout
    reached_min_ticks = False
    assertions: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        assertions.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    try:
        while time.time() < deadline:
            try:
                chunk = sock.recv(262144)
            except socket.timeout:
                if process.poll() is not None:
                    break
                continue
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                if not raw_line.strip():
                    continue
                try:
                    message = json.loads(raw_line)
                except json.JSONDecodeError:
                    report.setdefault("invalid_lines", []).append(raw_line[:200].decode("utf-8", "replace"))
                    continue
                kind = message.get("type")
                if kind == "hello":
                    hello = message
                elif kind == "obs":
                    observations.append(message)
                    if len(observations) == 1:
                        check_first_obs(message, meta, record)
                    if (message.get("tick") or 0) >= args.min_ticks:
                        reached_min_ticks = True
                    if message.get("result"):
                        break
                elif kind == "done":
                    done_message = message
            if done_message is not None:
                break
            if reached_min_ticks and not args.full_run:
                break
    finally:
        try:
            sock.sendall(b'{"type":"shutdown"}\n')
        except OSError:
            pass

    report["hello"] = hello
    report["done"] = done_message
    report["assertions"] = assertions
    report["observation_count"] = len(observations)

    if observations:
        first = observations[0]
        last = observations[-1]
        ticks = [int(obs.get("tick", 0)) for obs in observations]
        report["first_obs"] = {"tick": first.get("tick"), "own": len(first.get("own_entities", [])),
                               "enemy": len(first.get("visible_enemy_entities", [])),
                               "result": first.get("result")}
        report["last_obs"] = {"tick": last.get("tick"), "own": len(last.get("own_entities", [])),
                              "enemy": len(last.get("visible_enemy_entities", [])),
                              "result": last.get("result")}
        report["obs_ticks"] = ticks

    # ------------------------------------------------------------------
    # Assertions shared by every observation stream (after the per-obs first check).
    # ------------------------------------------------------------------
    own_counts = [len(obs.get("own_entities", [])) for obs in observations]
    enemy_counts = [len(obs.get("visible_enemy_entities", [])) for obs in observations]
    ticks = [int(obs.get("tick", 0)) for obs in observations]
    record(
        "observations_received",
        len(observations) >= 2,
        f"obs_count={len(observations)}",
    )
    if observations:
        record(
            "own_entities_25",
            all(count == 25 for count in own_counts),
            f"counts={sorted(set(own_counts))}",
        )
        record(
            "enemy_entities_25",
            all(count == 25 for count in enemy_counts),
            f"counts={sorted(set(enemy_counts))}",
        )
        record(
            "ticks_monotonic",
            all(later >= earlier for earlier, later in zip(ticks, ticks[1:])),
            f"ticks={ticks[:3]}...{ticks[-3:]}",
        )
        required_tick = (args.ticks - 10) if args.full_run else args.min_ticks
        record(
            "ticks_advanced",
            bool(ticks) and ticks[-1] >= required_tick,
            f"last_tick={ticks[-1] if ticks else None} required>={required_tick}",
        )
        record(
            "no_early_termination",
            all(not obs.get("terminated") for obs in observations),
            "no terminated obs frames while the episode is still running"
            + (" (annihilation mode keeps the building-less arena live)" if args.win_mode == "annihilation" else " (native mode may end early on building-less maps)"),
        )
        if args.full_run:
            record(
                "ran_full_horizon",
                bool(ticks) and ticks[-1] >= args.ticks - 10,
                f"last_tick={ticks[-1] if ticks else None} ticks={args.ticks}",
            )
            record(
                "done_received",
                done_message is not None,
                f"done={done_message}",
            )
            if done_message is not None:
                record(
                    "done_timeout_at_tick_cap",
                    done_message.get("result") == "timeout"
                    and int(done_message.get("tick", 0)) == args.ticks,
                    f"done.tick={done_message.get('tick')} done.result={done_message.get('result')}",
                )


def _nearest_distance(point: tuple[float, float], candidates: list[tuple[float, float]]) -> float:
    return min(((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5 for cx, cy in candidates)


def check_first_obs(message: dict[str, Any], meta: dict[str, Any], record) -> None:
    own = message.get("own_entities", [])
    enemy = message.get("visible_enemy_entities", [])
    record("first_obs_own_count_25", len(own) == 25, f"own={len(own)}")
    record("first_obs_enemy_count_25", len(enemy) == 25, f"enemy={len(enemy)}")

    hp_ok = all(
        abs(float(entity.get("max_hp", 0.0)) - TANK_MAX_HP) <= TANK_HP_TOLERANCE
        for entity in own + enemy
    )
    hps = sorted({round(float(entity.get("max_hp", 0.0)), 2) for entity in own + enemy})
    record("tank_max_hp_210", hp_ok, f"max_hp_values={hps}")

    unit_flags_ok = all(
        not entity.get("is_building")
        and not entity.get("is_factory")
        and not entity.get("is_builder")
        and int(entity.get("entity_type_id", -1)) == 1
        for entity in own + enemy
    )
    record("all_units_are_mobile_units", unit_flags_ok, "no buildings/factories/builders in the arena")

    team_ok = all(int(entity.get("team_id", -1)) == 0 for entity in own) and all(
        int(entity.get("team_id", -1)) == 1 for entity in enemy
    )
    record("team_ids_0_and_1", team_ok, "own=team0, visible_enemy=team1")

    # Spawn coordinates must match the generated meta (180-degree rotation pair).
    team0_expected = [tuple(point) for point in meta["spawn"]["team_0"]["positions_px"]]
    team1_expected = [tuple(point) for point in meta["spawn"]["team_1"]["positions_px"]]
    own_positions = [(float(entity["x"]), float(entity["y"])) for entity in own]
    enemy_positions = [(float(entity["x"]), float(entity["y"])) for entity in enemy]
    own_error = max(_nearest_distance(point, own_positions) for point in team0_expected)
    enemy_error = max(_nearest_distance(point, enemy_positions) for point in team1_expected)
    record(
        "team0_spawn_coordinates",
        own_error <= SPAWN_TOLERANCE_PX,
        f"max deviation from meta={own_error:.2f}px",
    )
    record(
        "team1_spawn_coordinates",
        enemy_error <= SPAWN_TOLERANCE_PX,
        f"max deviation from meta={enemy_error:.2f}px",
    )
    symmetry_error = max(
        _nearest_distance(
            (meta["size"]["width_px"] - px, meta["size"]["height_px"] - py), enemy_positions
        )
        for px, py in own_positions
    )
    record(
        "spawn_180_degree_symmetry",
        symmetry_error <= SPAWN_TOLERANCE_PX,
        f"max |enemy - rotate(own)| nearest deviation={symmetry_error:.2f}px",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time()
    runtime, runtime_origin = resolve_runtime(args.runtime)
    report: dict[str, Any] = {
        "map_relative": args.map_relative,
        "runtime": str(runtime),
        "runtime_origin": runtime_origin,
        "win_mode": args.win_mode,
        "unlock_ai": not args.no_unlock_ai,
        "args": {
            "port": args.port,
            "ticks": args.ticks,
            "frame_ms": args.frame_ms,
            "min_ticks": args.min_ticks,
            "full_run": args.full_run,
        },
        "ok": False,
    }

    if not port_is_free(args.port):
        report["error"] = f"port {args.port} is already in use"
        write_report(args.out, report)
        print(f"ERROR: {report['error']}", file=sys.stderr)
        return 2

    runtimes_non_ascii = any(ord(char) > 127 for char in str(runtime))
    if runtimes_non_ascii:
        print(
            "WARNING: runtime path contains non-ASCII characters; Java RWML self-attach may fail. "
            "Use the %TEMP% ASCII mirror.",
            file=sys.stderr,
        )

    meta_path = Path(args.meta)
    if not meta_path.is_file():
        report["error"] = f"meta json missing: {meta_path}"
        write_report(args.out, report)
        print(f"ERROR: {report['error']}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    map_source = Path(args.map_source)
    installed = install_map(runtime, map_source, args.map_relative)
    report["installed_map"] = str(installed)
    report["map_bytes"] = installed.stat().st_size
    report["map_sha256"] = hashlib.sha256(installed.read_bytes()).hexdigest()
    print(f"runtime        : {runtime} ({runtime_origin})")
    print(f"map installed  : {installed} ({installed.stat().st_size} bytes)")

    log_path = Path(args.log)
    process, server_log = launch_server(args, runtime, log_path)
    report["server_log"] = str(log_path)
    obs_count = 0
    try:
        sock = connect(args.port, args.connect_timeout, process)
        print(f"bridge connected on 127.0.0.1:{args.port}")
        with sock:
            read_messages(sock, process, args, report, meta)
            obs_count = report.get("observation_count", 0)
    except RuntimeError as failure:
        report["error"] = str(failure)
        print(f"ERROR: {failure}", file=sys.stderr)
    finally:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        server_log.close()

    assertions = report.get("assertions", [])
    failed = [item for item in assertions if not item["ok"]]
    report["failed_assertions"] = failed
    report["server_exit_code"] = process.returncode
    report["duration_s"] = round(time.time() - started_at, 2)
    report["ok"] = not failed and not report.get("error") and obs_count >= 2
    write_report(args.out, report)

    print()
    for item in assertions:
        print(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']}: {item['detail']}")
    first = report.get("first_obs") or {}
    last = report.get("last_obs") or {}
    print(
        f"\nfirst obs: tick={first.get('tick')} own={first.get('own')} enemy={first.get('enemy')} "
        f"result={first.get('result')}"
    )
    print(
        f"last obs : tick={last.get('tick')} own={last.get('own')} enemy={last.get('enemy')} "
        f"result={last.get('result')}"
    )
    print(f"report   : {Path(args.out)}")
    print(f"verdict  : {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


def write_report(path: str, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
