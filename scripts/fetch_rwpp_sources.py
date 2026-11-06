# SPDX-License-Identifier: Apache-2.0
"""Pre-download RWPP/RWML sources and game-lib artifacts for the compat build.

The repo build script uses curl, which cannot pass this machine's TLS proxy;
Python's urllib works, so this helper fills the ``downloads`` directory that
``scripts/build_rwpp_compat.ps1`` expects, then the script can skip downloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"E:\qwq\强化学习笔记\铁锈战争")
BUILD = Path(r"E:\IronZeroBuild")
DOWNLOADS = BUILD / "downloads"
MANIFEST = json.loads((ROOT / "vendor" / "rwpp" / "pinned-versions.json").read_text(encoding="utf-8"))


def opener():
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()


def fetch(url: str, target: Path, *, sha256: str | None = None) -> bool:
    if target.exists() and (not sha256 or hashlib.sha256(target.read_bytes()).hexdigest().upper() == sha256.upper()):
        print(f"cached: {target.name} ({target.stat().st_size/1e6:.1f} MB)", flush=True)
        return True
    print(f"download: {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ironzero-build/1.0"})
    started = time.time()
    try:
        with opener().open(request, timeout=120) as response, target.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED {url}: {exc}", file=sys.stderr, flush=True)
        if target.exists():
            target.unlink()
        return False
    elapsed = time.time() - started
    if sha256:
        digest = hashlib.sha256(target.read_bytes()).hexdigest().upper()
        if digest != sha256.upper():
            print(f"checksum mismatch: {digest} != {sha256}", file=sys.stderr, flush=True)
            return False
    print(f"ok: {target.name} ({target.stat().st_size/1e6:.1f} MB, {elapsed:.0f}s)", flush=True)
    return True


def main() -> int:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    rwpp_commit = MANIFEST["rwpp"]["commit"]
    rwml_commit = MANIFEST["rwml"]["commit"]
    gl_version = MANIFEST["game_lib"]["version"]
    gl_sha = MANIFEST["game_lib"]["sha256"]
    maven = f"https://repo1.maven.org/maven2/io/github/rwpp-team/game-lib/{gl_version}"
    jobs = [
        (f"https://codeload.github.com/RWPP-Team/RWPP/tar.gz/{rwpp_commit}", DOWNLOADS / f"RWPP-{rwpp_commit}.tar.gz", None),
        (f"https://codeload.github.com/RWPP-Team/RWML/tar.gz/{rwml_commit}", DOWNLOADS / f"RWML-{rwml_commit}.tar.gz", None),
        (f"{maven}/game-lib-{gl_version}.jar", DOWNLOADS / f"game-lib-{gl_version}.jar", gl_sha),
        (f"{maven}/game-lib-{gl_version}-sources.jar", DOWNLOADS / f"game-lib-{gl_version}-sources.jar", None),
        (f"{maven}/game-lib-{gl_version}.pom", DOWNLOADS / f"game-lib-{gl_version}.pom", None),
    ]
    failures = 0
    for url, target, sha in jobs:
        if not fetch(url, target, sha256=sha):
            failures += 1
    print(f"done, failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
