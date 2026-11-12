# `ironzero.terrain` — 地形栅格与确定性 A* 寻路

> 本包由原 `src/ironzero/terrain.py`（463 行）拆分而来。
> **只搬结构，不改行为**：A* 的邻居顺序、开放集 tie-break、代价函数、节点预算
> 与拆分前逐字节一致。旧路径 `ironzero.terrain` 仍然可用——它现在是本包
> （`ironzero/terrain/__init__.py`）本身。
>
> ⚠️ **命名冲突提示**：拆分前 `src/ironzero/terrain.py` 是文件，目标 `src/ironzero/terrain/`
> 是目录，二者在同一个 import 名上冲突。正确顺序是**先建包、再删文件**，
> 中间态若导入了旧 `.pyc` 会拿到过期实现（记得清 `__pycache__/terrain.cpython-*.pyc`）。

## 职责

平台后端解析真实 `.tmx` 地图（见 `web/backend/maps.py`），并给每个推理任务写一份紧凑的
`map.json`：

```json
{
  "id": "<map id>", "name": "[p2]Lake (2p)",
  "width": 130, "height": 130,
  "tile_width": 20, "tile_height": 20,
  "width_px": 2600, "height_px": 2600,
  "blocked": [0, 0, 1, "..."]
}
```

`TerrainGrid` 把它变成**不可变的碰撞结构**，提供世界（像素）坐标查询，以及一个
确定性的 4 邻接 A*。**纯标准库、完全确定性**：同一栅格 + 同一查询永远得到同一条路径，
引擎靠这一点保证 episode 可复现。

## 模块

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `__init__.py` | 42 | re-export `TerrainGrid` / `DEFAULT_MAX_NODES` / 两个 mixin / 工具函数 |
| `grid.py` | 165 | `TerrainGrid` 本体：`from_payload` / `from_file` / 坐标变换 / 阻挡判定 / `tile_count` / `land_ratio` |
| `passable.py` | 98 | `PassableMixin`：`nearest_passable_world` / `_nearest_passable_in_box` / `_inside_box` |
| `pathing.py` | 159 | `PathingMixin`：`find_path`（A*）/ `_compress_waypoints` / `path_length`；`DEFAULT_MAX_NODES = 30000` |
| `utils.py` | 36 | `_positive_int` / `_clamp` / `_clamp_int` / `_clamp_bounds`（无状态） |

## 为什么是 mixin，而不是拆成独立函数

`TerrainGrid` 是 `@dataclass(frozen=True, slots=True)`。原本单类 414 行，其中 A* 独占一大块。
**对外 API 必须完全不变**（`tests/test_terrain.py` 449 行覆盖 spawn / movement / regression，
且引擎直接调 `grid.find_path(...)`、`grid.nearest_passable_world(...)`、`grid.blocked`），
所以采用**无状态 mixin**组合：

```python
@dataclass(frozen=True, slots=True)
class TerrainGrid(PassableMixin, PathingMixin):
    map_id: str
    ...
    blocked: tuple[bool, ...]
```

两个 mixin 都声明 `__slots__ = ()`，只读取 `grid.py` 里声明的字段并调用其公开辅助方法，
**不新增任何实例属性**——因此 dataclass 的字段顺序、`frozen`/`slots` 语义、
`dataclasses.fields()`、`==`、`hash()` 与 `repr` 全部保持原样。这也是等价变换的前提：
原类没有 `super()`、没有 `@property` 之外的自定义描述符，mixin 拆分是安全的。

## 数据流

```
  map.json / TerrainGrid 实例
            │
            ▼
   TerrainGrid.from_payload(...)    ← 也接受 TerrainGrid 实例本身（直接返回）
   TerrainGrid.from_file(path)      ← 容忍 payload 嵌套在 "terrain" / "map" 下，
                                      缺 tile 数时由 width_px / height_px 反推
            │
            ▼
   ┌────────────────────────┬──────────────────────────────┐
   │  grid.py               │  passable.py                 │  pathing.py
   │  坐标变换 & 阻挡判定    │  最近可通行点（环形扫描）      │  确定性 A*
   │  world_to_tile         │  nearest_passable_world      │  find_path
   │  tile_center           │    ├ bounds 约束在半场        │  _compress_waypoints
   │  is_blocked_tile       │    ├ 已在陆地则原样返回        │  path_length
   │  is_blocked_world      │    └ 全阻挡图则原地不动        │
   │  tile_count/land_ratio │                              │
   └────────────────────────┴──────────────────────────────┘
            │                        │                       │
            ▼                        ▼                       ▼
    mock_engine 的出生点吸附、    观测通道 3 的地形先验     引擎移动：世界坐标航点
    移动碰撞、入水吸附           (observation/terrain_channel)  (_repath_terrain)
```

### 消费方（谁依赖本包）

- `ironzero.mock_engine` — 出生点吸附到岸边、逐步移动的碰撞与滑动、入水目标吸附；
- `ironzero.observation.terrain_channel` — `TerrainGrid.from_file` + 降采样成观测通道 3；
- `ironzero.training.record` / 平台后端 — 从 `map.json` 载入。

本包**不认识** `observation/`、`mock_engine`、`numpy`——依赖方向是单向的。

## 关键行为约定（改动即返工）

1. **A* 确定性**：开放集 tie-break 是 `(f, h, tile_y, tile_x)`，4 邻居顺序是
   上/下/左/右。测试 `test_a_star_is_deterministic` 与引擎的可复现 episode 都依赖它。
2. **节点预算**：`DEFAULT_MAX_NODES = 30000` 通过 `find_path(max_nodes=...)` 传入；
   超预算返回 `None`（不是抛异常）。130×130 = 16900 格的湖泊图远低于此上限。
3. **`exact_goal`**：为 `True` 时终点用请求的精确世界坐标（当地面可通行），
   否则用 tile 中心。`test_a_star_exact_goal_keeps_requested_point` 锁这一点。
4. **航点压缩**：连续共线的 tile 中心合并成一段，只保留拐点；
   跟随者只需走到下一个转弯，而不是每个格心。
5. **`nearest_passable_world` 语义**：请求点本身可通行就**原样返回**（保住精确目标）；
   `bounds` 是可选的世界坐标闭区间盒，用于把出生点约束在地图正确的一侧；
   有界区域内找不到可通行格时回退到全图；全图都不可通行时返回钳制后的原地坐标。
   环形扫描的 tie-break 是"先低 y 后低 x"。
6. **越界即阻挡**：`is_blocked_tile` 对越界 tile 返回 `True`；
   `is_blocked_world` 对 `NaN` / `inf` 也返回 `True`。

## 自测

```powershell
python -m pytest -q tests/test_terrain.py          # 26 passed（spawn / movement / regression）
python -m pytest -q                                # 全量 209 passed
python -X utf8 -c "import sys; sys.path.insert(0,'src'); from ironzero.terrain import TerrainGrid; print(TerrainGrid.__module__)"
```
