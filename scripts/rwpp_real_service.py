# SPDX-License-Identifier: Apache-2.0
"""兼容 shim：实现已迁移至 :mod:`rwpp_service`（``scripts/rwpp_service/``）。

入口保持不变：

    python scripts/rwpp_real_service.py --port 8300 --max-parallel 4

脚本目录不是包，必须先把 ``scripts/`` 放进 ``sys.path`` 才能 import
``rwpp_service``。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rwpp_service.server import main  # noqa: E402

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
