# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the economy actions on a stock map.

Starts one real headless game, then drives the three economy actions through the
bridge and asserts that each one comes back ``accepted=true``:

1. ``build_extractor`` -- the bridge must resolve the policy's ``slot_id`` to a
   real ore pool (``Map.resourcePoolPoints``). This is the regression that
   mattered: the bridge used to scan ``Unit.unitList`` for resource *entities*,
   found none, fell back to map coordinate (0,0) and every build was rejected.
2. ``produce_unit`` -- ``land_factory`` production via the factory's
   ``SpecialAction`` / build queue.
3. ``upgrade_building`` -- extractor T1 -> T2 via the building's
   ``getBuildActionId()``.

Also checks that the observation now carries a non-empty ``resource_slots``
array, because the policy needs that list to choose a mining spot.

Exit codes: 0 = every action accepted, 3 = resource slots missing (map has no
ore), 4 = an action was rejected, 2 = the game/bridge never came up.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ironzero.protocol import parse_message  # noqa: E402

DEFAULT_RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
DEFAULT_JAVA_HOME = Path(r"E:\jdk25copy")
DEFAULT_MAP = "maps/skirmish/[p2]Lake (2p).tmx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Economy action smoke test")
    parser.add_argument("--port", type=int, default=5135)
    # A builder needs ~15 s of game time (900+ ticks) to walk to a nearby ore
    # pool and finish the structure, and the test also waits for production and
    # an upgrade, so the budget is comfortably larger than one build.
    parser.add_argument("--ticks", type=int, default=30000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--java-home", default=str(DEFAULT_JAVA_HOME))
    parser.add_argument("--frame-ms", type=int, default=1)
    return parser


class Bridge:
    """Tiny NDJSON client: send ``step`` messages, read obs/receipt/done."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.decision = 0
        self.last_obs: dict | None = None
        self.receipts: dict[str, list[dict]] = {}
        self.done = False

    def pump(self, timeout: float) -> list[dict]:
        """Read whatever is available for up to ``timeout`` seconds."""
        messages: list[dict] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            if b"\n" not in self.buffer:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.sock.settimeout(max(0.2, min(remaining, 5.0)))
                try:
                    chunk = self.sock.recv(1 << 16)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                self.buffer += chunk
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    parse_message(line.decode("utf-8"))
                    message = json.loads(line)
                except Exception:  # noqa: BLE001 - a bad line must not kill the test
                    continue
                self._absorb(message)
                messages.append(message)
                if message.get("type") == "obs":
                    # One decision per observation: stop reading ahead.
                    return messages
        return messages

    def _absorb(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "obs":
            self.last_obs = message
        elif kind == "receipt":
            self.receipts.setdefault(str(message.get("decision_id")), []).append(message)
        elif kind == "done":
            self.done = True

    def step(self, actions: list[dict], timeout: float = 30.0) -> list[dict]:
        """Send one decision and read until the next observation.

        The bridge streams receipts *before* the next ``obs`` for the same
        decision, so reading just past the next observation is enough to collect
        them; the extra drain covers a bridge that flushes receipts late.
        """
        self.decision += 1
        decision_id = f"d-smoke-{self.decision}"
        payload = {
            "protocol": "ironzero/1",
            "type": "step",
            "message_id": f"m-{decision_id}",
            "episode_id": "rwpp-bridge",
            "tick": (self.last_obs or {}).get("tick", 0),
            "decision_id": decision_id,
            "actions": actions,
        }
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        self.pump(timeout)
        if decision_id not in self.receipts and not self.done:
            # Receipts can trail the observation by a frame; give them one more.
            self.pump(5.0)
        return self.receipts.get(decision_id, [])

    def wait_obs(self, timeout: float = 60.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.pump(5.0)
            if self.last_obs is not None or self.done:
                return self.last_obs
        return self.last_obs

    def settle(self, seconds: float) -> None:
        """Let the game run without issuing decisions (builder walks, builds)."""
        deadline = time.time() + seconds
        while time.time() < deadline and not self.done:
            self.pump(2.0)


def own_units(obs: dict) -> list[dict]:
    return list(obs.get("own_entities") or [])


def find_builder(obs: dict) -> dict | None:
    for unit in own_units(obs):
        if unit.get("is_builder"):
            return unit
    return None


def report(label: str, receipts: list[dict]) -> bool:
    if not receipts:
        print(f"  {label:<24} NO RECEIPT")
        return False
    accepted = [r for r in receipts if r.get("accepted")]
    if accepted:
        print(f"  {label:<24} accepted=true")
        return True
    reasons = sorted({str(r.get("rejected_reason")) for r in receipts})
    print(f"  {label:<24} accepted=false  reason={','.join(reasons)}")
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Path(args.runtime)
    java_home = Path(args.java_home)
    launcher = runtime / "bin" / "rwpp-server.bat"
    if not launcher.is_file():
        print(f"runtime launcher missing: {launcher}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_runs_econ_smoke.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(launcher),
            "--no-console",
            "--",
            "--bridge-port",
            str(args.port),
            "--ticks",
            str(args.ticks),
            "--map",
            args.map,
            "--frame-ms",
            str(args.frame_ms),
        ],
        cwd=str(runtime),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    sock: socket.socket | None = None
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", args.port), timeout=2.0)
                break
            except OSError:
                time.sleep(0.5)
        if sock is None:
            print(f"bridge port {args.port} never opened; see {log_path}", file=sys.stderr)
            return 2
        sock.settimeout(5.0)

        bridge = Bridge(sock)
        obs = bridge.wait_obs(90.0)
        if obs is None:
            print("no first observation; see", log_path, file=sys.stderr)
            return 2

        slots = obs.get("resource_slots") or []
        global_state = obs.get("global_state") or {}
        print(f"map            = {args.map}")
        print(f"own units      = {len(own_units(obs))}")
        print(f"resource_slots = {len(slots)}")
        print(f"funds          = {global_state.get('funds')}")
        print(f"income         = {global_state.get('income')}")
        if slots:
            print(f"  first slot   = ({slots[0].get('x')}, {slots[0].get('y')}) occupied={slots[0].get('occupied')}")

        if not slots:
            print("\nRESOURCE SLOTS MISSING: the bridge found no ore pools on this map.")
            print("The economy actions cannot succeed without them.")
            return 3

        ok = True
        print("\nactions:")

        # ---- 1. build_extractor -------------------------------------------
        # Build range is limited (the builder must walk to the pool and the
        # engine rejects anything out of range), so try the pools nearest to our
        # own base first: on a stock skirmish map some ore sits across the map.
        base = next(
            (unit for unit in own_units(obs) if unit.get("is_factory")),
            (own_units(obs) or [{"x": 0.0, "y": 0.0}])[0],
        )
        nearby = sorted(
            slots,
            key=lambda slot: (float(slot["x"]) - float(base["x"])) ** 2
            + (float(slot["y"]) - float(base["y"])) ** 2,
        )
        built = False
        for rank, slot in enumerate(nearby[:4], start=1):
            if find_builder(obs) is None:
                print("  (no own builder alive; cannot test build_extractor)")
                break
            distance = (
                (float(slot["x"]) - float(base["x"])) ** 2
                + (float(slot["y"]) - float(base["y"])) ** 2
            ) ** 0.5
            receipts = bridge.step(
                [{"kind": "build_extractor", "unit_group": 4,
                  "slot_id": int(slot["slot_id"])}]
            )
            if report(f"build_extractor(slot{slot['slot_id']}, d={distance:.0f})", receipts):
                built = True
                break
            obs = bridge.last_obs or obs
            time.sleep(1.0)
        ok = ok and built

        # Give the builder time to walk over and finish the structure: the order
        # is accepted immediately but the extractor only appears after the build
        # completes, and the upgrade test needs that extractor to exist.
        if built:
            for _ in range(40):
                if bridge.done:
                    break
                bridge.settle(3.0)
                obs = bridge.last_obs or obs
                if any(
                    unit.get("is_building")
                    and "extractor" in str(unit.get("unit_type", "")).lower()
                    for unit in own_units(obs)
                ):
                    break
        obs = bridge.last_obs or obs
        print(f"  after build: building_count={(obs.get('global_state') or {}).get('building_count')}"
              f" own_units={len(own_units(obs))}")

        # ---- 2. produce_unit ----------------------------------------------
        # A map-placed commandCenter only queues ``builder``; ``tank`` needs a
        # landFactory. Try the types a base can plausibly make and accept the
        # first one the factory's own menu offers -- the point is to prove the
        # production path reaches the engine, not to force a specific unit.
        factories = [
            unit for unit in own_units(obs)
            if unit.get("is_factory") or unit.get("is_factory_building")
        ]
        if factories:
            factory_id = int(factories[0].get("entity_id", -1))
            produced = False
            for unit_type in ("builder", "tank"):
                receipts = bridge.step(
                    [{"kind": "produce_unit", "unit_group": 4,
                      "factory_entity_id": factory_id, "unit_type": unit_type}]
                )
                if report(f"produce_unit({unit_type})", receipts):
                    produced = True
                    break
            ok = produced and ok
        else:
            print("  produce_unit            SKIPPED (no own factory)")
            ok = False

        # ---- 3. upgrade_building ------------------------------------------
        extractors = [
            unit for unit in own_units(obs)
            if unit.get("is_building") and "extractor" in str(unit.get("unit_type", "")).lower()
        ]
        if extractors:
            building_id = int(extractors[0].get("entity_id", -1))
            receipts = bridge.step(
                [{"kind": "upgrade_building", "unit_group": 4,
                  "building_entity_id": building_id}]
            )
            ok = report("upgrade_building", receipts) and ok
        else:
            print("  upgrade_building        SKIPPED (no own extractor yet)")
            ok = False

        print()
        if ok:
            print("ECONOMY SMOKE PASS")
            return 0
        print("ECONOMY SMOKE FAIL")
        return 4
    finally:
        if sock is not None:
            try:
                sock.sendall(b'{"type":"shutdown"}\n')
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
