# SPDX-License-Identifier: Apache-2.0
"""Download and unpack a portable JDK 25 for the RWPP runtime build (local only)."""
from __future__ import annotations

import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://aka.ms/download-jdk/microsoft-jdk-25-windows-x64.zip"
TARGET_DIR = Path(r"E:\IronZeroBuild")
ARCHIVE = TARGET_DIR / "jdk25.zip"
DEST = Path(r"E:\qwq\强化学习笔记\铁锈战争\.runtime\jdk25")


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})) if proxy else urllib.request.build_opener()
    request = urllib.request.Request(URL, headers={"User-Agent": "ironzero-build/1.0"})
    print(f"downloading {URL}", flush=True)
    started = time.time()
    with opener.open(request, timeout=120) as response, ARCHIVE.open("wb") as handle:
        if response.status != 200:
            raise SystemExit(f"HTTP {response.status}")
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total and downloaded % (20 << 20) < (1 << 20):
                print(f"{downloaded/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    print(f"downloaded {ARCHIVE} ({ARCHIVE.stat().st_size/1e6:.1f} MB) in {time.time()-started:.0f}s", flush=True)
    if DEST.exists():
        import shutil

        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(DEST)
    print(f"extracted to {DEST}", flush=True)
    for candidate in DEST.rglob("bin/java.exe"):
        print("java.exe:", candidate, flush=True)
        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
