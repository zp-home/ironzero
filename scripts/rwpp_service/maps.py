# SPDX-License-Identifier: Apache-2.0
"""原版 2p 遭遇战地图目录（供 Web 表单与 ``/maps`` 只读接口使用）。

为什么需要这个模块
------------------
Web 表单原来把 ``map_width`` / ``map_height`` 写死成 ``2700x2500``（Artillery
的尺寸）。观测层用地图像素尺寸归一化每一个单位坐标，所以在一张
``4000x3200`` 的地图上跑 ``2700x2500`` 会把所有坐标压进左三分之一，而且**不会
报任何错** —— 界面上只是"模型学不会"。``Lake`` 也一直被写成 ``2600x2500``，
y 方向差 100 px，归一化从头就偏。

唯一可信来源是 :mod:`ironzero.rwpp.maps`（另一处维护的注册表），它把九张原版
``[p2]`` 地图的像素尺寸写成常量。这里做两件事：

1. **优先**读那个注册表（``map_specs()``）；
2. 读不到时退回 :func:`_fallback_map_catalog` 的内置 9 行常量 —— 任务不允许
   "等别人建好再开工"，而且宿主机服务导入 ``ironzero`` 会连带拉起 torch，
   一个只读的地图清单不该有那个代价。

注册表是**按文件路径直接加载**的（``importlib`` + ``spec_from_file_location``），
不是 ``import ironzero.rwpp.maps``：后者会先执行 ``ironzero/__init__`` 和
``ironzero/rwpp/__init__``，把 env/torch 全都拖进来。``maps.py`` 自己就是纯标准
库模块，直接按路径执行既快又不会因为包内别处的改动而 import 失败。

查找规则
--------
``Lake``、``[p2]Lake (2p).tmx``、``maps/skirmish/[p2]Lake (2p).tmx``、
``Dire_Straight`` 与 ``Dire Straight`` 必须都能解析到同一条记录 —— 表单、CLI
参数和资源清单各自用的写法不一样，让调用方去翻译只会多一处出错的地方。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from .paths import REPO_ROOT

__all__ = [
    "DEFAULT_ENEMY_AI_DIFFICULTY",
    "DEFAULT_MAP_NAME",
    "DIFFICULTY_CHOICES",
    "lookup_map",
    "map_catalog",
    "maps_payload",
    "resolve_map_dimensions",
]

#: 表单默认地图。``Lake`` 是历史默认图，也是唯一有大量实测数据的 2p 图。
DEFAULT_MAP_NAME = "Lake"

#: 引擎的六档 AI 难度（-2 非常简单 … 3 炼狱）。难度不是装饰：它直接决定
#: ``Team.getAIDifficultyIncomeMultiplier()``（x0.40 / x0.70 / x1.00 / x1.40 /
#: x1.80 / x3.20）和 AI 进攻前攒到多大的军队。
DIFFICULTY_CHOICES: tuple[int, ...] = (-2, -1, 0, 1, 2, 3)

#: 未显式指定时的对手难度。
#:
#: ``1``（困难，x1.40 经济）而不是 ``None``：``None`` 会让引擎去读全局
#: ``SettingsEngine.aiDifficulty``，那个值默认是 **0**（正常，x1.00，没有任何加成），
#: 于是"对手难度"这一栏在界面上看着有值、实际跑的是最容易的一档。训练需要一个
#: 真的会进攻的对手，否则策略学到的只是"对面不动"。
DEFAULT_ENEMY_AI_DIFFICULTY = 1

#: 注册表模块的缓存。``None`` = 还没试过；``False`` = 试过且不可用
#: （文件不存在或执行失败），避免每次请求都重新失败一遍。
_REGISTRY: object = None


def _fallback_map_catalog() -> tuple[dict, ...]:
    """内置的九张原版 2p 地图常量（``.tmx`` 头部实测值）。

    这也是**显示名**的唯一来源。:mod:`ironzero.rwpp.maps` 的 ``MapSpec.name``
    取自 ``.tmx`` 文件名主干（``Dire_Straight``、``Two_cold_sides``），而这九个
    名字是项目通篇文档、任务书和运营沟通里用的写法（带空格）。用文件名做主键
    会让下拉框显示 ``Two_cold_sides``，和所有既有记录对不上。

    每张图的像素尺寸与水域占比是 ``.tmx`` 头部 / Ground 层的实测值；
    :func:`map_catalog` 会用注册表里的最新测量值覆盖它们（注册表优先），但名字
    保持这里的样子。
    """

    rows = (
        ("Beach landing", "maps/skirmish/[p2]Beach landing (2p) [by hxyy].tmx", 4000.0, 3200.0, 0.6082),
        ("Big Island", "maps/skirmish/[p2]Big Island (2p).tmx", 3600.0, 3600.0, 0.5067),
        ("Dire Straight", "maps/skirmish/[p2]Dire_Straight (2p) [by uber].tmx", 2200.0, 2200.0, 0.4644),
        ("Fire Bridge", "maps/skirmish/[p2]Fire Bridge (2p) [by uber].tmx", 2400.0, 2400.0, 0.2233),
        ("Hills", "maps/skirmish/[p2]Hills_(2p)_[By Tstis & KPSS].tmx", 2600.0, 2600.0, 0.0),
        ("Ice Island", "maps/skirmish/[p2]Ice Island (2p).tmx", 2900.0, 2900.0, 0.5156),
        ("Lake", "maps/skirmish/[p2]Lake (2p).tmx", 2600.0, 2600.0, 0.2502),
        ("Small Island", "maps/skirmish/[p2]Small_Island (2p).tmx", 2200.0, 2200.0, 0.6379),
        ("Two cold sides", "maps/skirmish/[p2]Two_cold_sides (2p).tmx", 2700.0, 2500.0, 0.6159),
    )
    return tuple(
        {
            "name": name,
            "map_path": relative,
            "map_width": width,
            "map_height": height,
            "water_ratio": water,
        }
        for name, relative, width, height, water in rows
    )


def _load_registry():
    """按文件路径加载 ``src/ironzero/rwpp/maps.py``，失败返回 ``None``。

    不 ``import ironzero.rwpp.maps``：那会先执行包的两个 ``__init__``，把 env 和
    torch 一起拖进宿主服务的请求线程里。注册表本身是纯标准库模块，按路径执行
    语义完全相同而代价小得多。任何异常都只意味着"退回内置常量"。
    """

    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY if _REGISTRY is not False else None
    path = REPO_ROOT / "src" / "ironzero" / "rwpp" / "maps.py"
    try:
        if not path.is_file():
            _REGISTRY = False
            return None
        spec = importlib.util.spec_from_file_location(
            "ironzero_rwpp_maps_registry", path
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            _REGISTRY = False
            return None
        module = importlib.util.module_from_spec(spec)
        # ``exec_module`` 必须在 ``sys.modules`` 里登记过模块名：``maps.py`` 用了
        # ``@dataclass(frozen=True, slots=True)``，数据类在解析注解时要回查
        # ``sys.modules[cls.__module__]``。失败时把这个半成品移除，不留脏状态。
        previous = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
            raise
    except Exception:  # noqa: BLE001 - 任何加载失败都退回内置常量
        _REGISTRY = False
        return None
    _REGISTRY = module
    return module


def _registry_map_catalog() -> tuple[dict, ...]:
    """从 :mod:`ironzero.rwpp.maps` 读取目录；不可用时返回空元组。"""

    module = _load_registry()
    if module is None:
        return ()
    try:
        specs = module.map_specs()
    except Exception:  # noqa: BLE001 - 注册表坏了也只是退回常量
        return ()
    entries: list[dict] = []
    for spec in specs:
        try:
            entries.append(
                {
                    "name": str(spec.name),
                    "map_path": str(spec.relative),
                    "map_width": float(spec.width_px),
                    "map_height": float(spec.height_px),
                    "water_ratio": float(spec.water_ratio),
                }
            )
        except (AttributeError, TypeError, ValueError):
            # 接口变了的条目跳过，其余照常 —— 一张图不该让整个下拉框空掉。
            continue
    return tuple(entries)


def map_catalog() -> tuple[dict, ...]:
    """九张地图的目录：仓库注册表的**测量值** + 本模块的**显示名**。

    合并规则是有意这样分工的：

    * ``map_width`` / ``map_height`` / ``water_ratio`` 以
      :mod:`ironzero.rwpp.maps` 为准 —— 那是有人真的去解析 ``.tmx`` 并测量
      水域占比的地方，它更新了就该跟着更新；
    * ``name`` 以 :func:`_fallback_map_catalog` 为准 —— 注册表的名字是文件名
      主干（``Two_cold_sides``），而文档、任务书和运营沟通一直写
      ``Two cold sides``。让下拉框显示另一种拼法只会让"我们说的那张图"变得
      需要翻译。

    注册表里出现了本模块没有的图（例如以后有人加了第 10 张）时，整条追加进去，
    这样一个新图不会因为这里漏改就悄悄从界面上消失。注册表不可用时退化为纯内置
    常量。
    """

    entries: list[dict] = [dict(entry) for entry in _fallback_map_catalog()]
    registry = _registry_map_catalog()
    if not registry:
        return tuple(entries)
    by_path = {_normalise_key(entry["map_path"]): entry for entry in entries}
    for measured in registry:
        target = by_path.get(_normalise_key(measured["map_path"]))
        if target is None:
            entries.append(dict(measured))
            by_path[_normalise_key(measured["map_path"])] = entries[-1]
            continue
        target["map_width"] = measured["map_width"]
        target["map_height"] = measured["map_height"]
        target["water_ratio"] = measured["water_ratio"]
    return tuple(entries)


def _normalise_key(value) -> str:
    """把各种写法压成同一个可比较的键（见模块文档）。"""

    text = str(value or "").strip().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    if text.lower().endswith(".tmx"):
        text = text[:-4]
    text = re.sub(r"^\[p\d+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*p\s*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def lookup_map(name_or_path) -> dict | None:
    """按名字或任意形式的路径查一条目录记录，查不到返回 ``None``。"""

    key = _normalise_key(name_or_path)
    if not key:
        return None
    for entry in map_catalog():
        if _normalise_key(entry["name"]) == key:
            return entry
        if _normalise_key(entry["map_path"]) == key:
            return entry
    return None


def resolve_map_dimensions(*candidates, fallback=(2600.0, 2600.0)) -> tuple[float, float]:
    """按候选名/路径依次查找，返回 ``(width_px, height_px)``。

    ``fallback`` 只在所有候选都不在目录里时使用（例如 ``IronZero25`` 这类竞技场
    图，它们不在这九张原版图里，尺寸由调用方给的历史默认值决定）。
    """

    for candidate in candidates:
        entry = lookup_map(candidate)
        if entry is not None:
            return (float(entry["map_width"]), float(entry["map_height"]))
    return (float(fallback[0]), float(fallback[1]))


def maps_payload() -> dict:
    """``GET /maps`` 的响应体。

    只读、无副作用、不碰文件系统以外的任何东西，因此表单每次打开都可以拿它
    当作唯一的地图清单来源。
    """

    entries = [dict(entry) for entry in map_catalog()]
    names = {entry["name"] for entry in entries}
    default = DEFAULT_MAP_NAME if DEFAULT_MAP_NAME in names else (
        entries[0]["name"] if entries else ""
    )
    return {
        "maps": entries,
        "default": default,
        "difficulty_choices": list(DIFFICULTY_CHOICES),
        "default_enemy_ai_difficulty": DEFAULT_ENEMY_AI_DIFFICULTY,
    }


def _reset_cache() -> None:
    """测试用：清掉注册表加载缓存，让下一次调用重新探测。"""

    global _REGISTRY
    _REGISTRY = None
