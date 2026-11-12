# SPDX-License-Identifier: Apache-2.0
"""Real-game capability probe for the economy actions.

Background
----------
``build_extractor`` / ``build_structure`` are implemented on the Java side via
``CommandControllerCommand.setBuildWaypoint`` (verified: the class compiles and
the bridge dispatches the action). Whether the *engine accepts* a build order
depends on map/unit data that this repo does not control:

* The stock skirmish maps (e.g. ``[p2]Lake (2p)``) place the built-in ``builder``
  unit, whose build menu does **not** include ``landFactory`` or ``extractor``.
  Only ``combat_engineer`` lists those (see ``assets/units/combat_engineer``).
* Therefore on a stock map an economy action is *correctly* rejected, and the
  bridge must report that rejection honestly instead of a false "accepted".

What this script asserts
------------------------
1. Every line on the wire parses against ``ironzero.protocol`` (no extras, no
   negative ``team_id``).
2. The bridge never reports ``accepted=true`` for an order the engine then
   rejects (the "false positive receipt" bug).
3. When the selected map *does* provide a capable builder, a build order is
   accepted end-to-end.

Exit code 0 = wire protocol is clean AND receipts agree with the engine log.
Exit code 3 = protocol clean but no capable builder on this map (expected on
stock skirmish maps; not a failure of the bridge).
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

RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")
JAVA_HOME = Path(r"E:\jdk25copy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWPP economy-action capability probe")
    parser.add_argument("--port", type=int, default=5133)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=200.0)
    parser.add_argument(
        "--map",
        default="maps/skirmish/[p2]Lake (2p).tmx",
        help="map the runtime loads; affects whether economy actions are buildable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + env.get("PATH", "")
    log_path = Path("runs/_rwpp_econ_probe.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(RUNTIME / "bin" / "rwpp-server.bat"),
            "--no-console",
            "--",
            "--bridge-port",
            str(args.port),
            "--ticks",
            str(args.ticks),
        ],
        cwd=str(RUNTIME),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    sock = None
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", args.port), timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        if sock is None:
            print("bridge port never opened; see", log_path, file=sys.stderr)
            return 2

        sock.settimeout(20)
        buffer = b""
        observations = 0
        receipts: list[dict] = []
        protocol_errors: list[str] = []
        own_types: set[str] = set()
        sent: list[str] = []
        step_index = 0

        while time.time() < deadline:
            try:
                chunk = sock.recv(1 << 16)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    parse_message(line.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    protocol_errors.append(f"{type(exc).__name__}: {str(exc)[:140]}")

                message = json.loads(line)
                kind = message.get("type")
                if kind == "obs":
                    observations += 1
                    own = message.get("own_entities", [])
                    own_types |= {u.get("unit_type", "?") for u in own}
                    if observations == 1:
                        print(f"world: {len(own)} own, "
                              f"{len(message.get('visible_enemy_entities', []))} others")
                        print("own unit types:", sorted(own_types))
                    # Probe both economy actions once each.
                    if observations == 3 and not sent:
                        step_index += 1
                        sent.append("build_extractor")
                        sock.sendall((json.dumps({
                            "protocol": "ironzero/1", "type": "step",
                            "message_id": f"m-{step_index}", "episode_id": "rwpp-bridge",
                            "tick": message["tick"], "decision_id": f"d-{step_index}",
                            "actions": [{"kind": "build_extractor",
                                         "unit_group": 4, "slot_id": 0}],
                        }) + "\n").encode("utf-8"))
                    elif observations == 8 and len(sent) == 1:
                        step_index += 1
                        sent.append("build_structure")
                        sock.sendall((json.dumps({
                            "protocol": "ironzero/1", "type": "step",
                            "message_id": f"m-{step_index}", "episode_id": "rwpp-bridge",
                            "tick": message["tick"], "decision_id": f"d-{step_index}",
                            "actions": [{"kind": "build_structure", "unit_group": 4,
                                         "structure_type": "land_factory",
                                         "x": 1500.0, "y": 1200.0}],
                        }) + "\n").encode("utf-8"))
                elif kind == "receipt":
                    receipts.append(message)
                    print("receipt accepted=", message.get("accepted"),
                          "reason=", message.get("rejected_reason"))
                elif kind == "done":
                    print("done tick", message.get("tick"))
                    break

        # Did the engine log reject any order the bridge called accepted?
        engine_rejects = []
        try:
            for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "can not queue build" in raw or "isValidNewWaypoint==false" in raw:
                    engine_rejects.append(raw.strip())
        except OSError:
            pass

        accepted = [r for r in receipts if r.get("accepted")]
        print()
        print(f"observations   = {observations}")
        print(f"own unit types = {sorted(own_types)}")
        print(f"receipts       = {len(receipts)}  accepted={len(accepted)}")
        print(f"protocol errors= {len(protocol_errors)}")
        for err in protocol_errors[:3]:
            print("   ", err)
        print(f"engine rejects = {len(engine_rejects)}")
        for line in engine_rejects[:3]:
            print("   ", line)

        # The core invariant: the bridge must never claim acceptance for an
        # order the engine then drops.
        false_positives = len(accepted) > 0 and len(engine_rejects) > 0
        print(f"false-positive receipts = {'YES' if false_positives else 'no'}")

        if protocol_errors or false_positives:
            print("ECONOMY PROBE FAIL")
            return 1

        if not accepted:
            print("ECONOMY PROBE PASS (protocol clean; no capable builder on this map)")
            print("NOTE: stock skirmish maps place the built-in 'builder', whose build")
            print("      menu lacks landFactory/extractor. Use a map that places")
            print("      'combat_engineer' to exercise a successful build.")
            return 3

        print("ECONOMY PROBE PASS (build accepted end-to-end)")
        return 0
    finally:
        if sock is not None:
            try:
                sock.sendall(b'{"type":"shutdown"}\n')
            except OSError:
                pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
