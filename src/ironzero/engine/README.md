# `ironzero.engine` —— 确定性 micro-duel 引擎

本包由原 `src/ironzero/mock_engine.py`（1225 行）拆分而来，**只搬结构、不改行为**。
`MicroDuelEngine` 通过 **mixin 组合** 装配：原类没有 `super()`、没有 `__slots__`、没有
自定义 `@property`，因此拆分是严格的等价变换（拆分前后对同一 `(config, opponent, seed,
terrain)` 的 `snapshot()` / `observation()` / receipts / reward 逐字节一致）。

旧 import 路径 `ironzero.mock_engine` 保留为纯 re-export shim。

## 1. 模块清单

| 模块 | 行数 | 职责 |
| --- | ---: | --- |
| `constants.py` | 55 | `ORDER_NAMES` / `ORDER_IDS` / `ACTION_KINDS`（位置即动作 id）/ `*_TYPE_ID` / `TERRAIN_*` 调参 / 8 个 `ErrorCode` 同名字符串 |
| `entities.py` | 55 | `@dataclass(slots=True)` 的 `Unit`（坦克 + 建筑共用）与 `MockResourceSlot` |
| `config.py` | 38 | `make_terrain_config()`：按真实地图像素尺寸重算 `map_width/height`、`spawn_margin`、`episode_ticks` |
| `util.py` | 8 | `_clamp()`（三角形收敛写法，NaN/inf 行为与原实现一致） |
| `spawn.py` | 157 | `SpawnMixin`：`reset()` 全量状态初始化 + `_spawn_terrain_units()`（贴岸出生） |
| `actions.py` | 233 | `ActionMixin`：`apply_actions()` / `_apply_one()`，含 5 种经济动作与 5 种军事动作的校验分支 |
| `movement.py` | 251 | `MovementMixin`：`_move` / `_move_terrain` / `_assign_group_path` / `_repath_terrain` / `_path_stale` / `_reset_unit_path` / `_step_toward` / `_try_offset` / `_distance` |
| `combat.py` | 93 | `CombatMixin`：`_select_target` / `_visible_to` / `_damage` / `_check_done` |
| `economy.py` | 66 | `EconomyMixin`：`_economic_tick()`（收入 + 工厂生产队列，原 `_tick` 的第一个 `if` 块）+ `_calculate_income()` |
| `opponents.py` | 96 | `OpponentMixin`：`scripted_actions()`（也作为 scripted 对手的 team-0/team-1 共用实现）与 `_opponent_actions()` |
| `observation.py` | 78 | `ObservationMixin`：`observation()` / `_entity_payload()`，输出 `ironzero/1` 形状的裸 payload |
| `reward.py` | 81 | `RewardMixin`：`_potential()` / `_reward_after_window()`（势能差 + 经济分解 + 终局奖励 + 闲置资金惩罚） |
| `engine.py` | 175 | `class MicroDuelEngine(*Mixins)`：`__init__` / `alive_units` / `unit_by_id` / `_tick` / `advance` / `step_decision` / `snapshot` / `evaluate` |
| `adapter.py` | 41 | `MicroDuelAdapter`：env 与 TCP bridge 共用的游戏侧适配器 |
| `__init__.py` | 88 | re-export 全部旧公开名 + `__all__` |

单文件上限要求：默认 ≤ 300 行，硬上限 450 —— 本包最大 251 行。

## 2. mixin 职责与组合顺序

```python
class MicroDuelEngine(
    SpawnMixin,        # reset / _spawn_terrain_units
    ActionMixin,       # apply_actions / _apply_one
    MovementMixin,     # _move / _move_terrain / A* 路径管理 / _distance
    CombatMixin,       # _select_target / _visible_to / _damage / _check_done
    EconomyMixin,      # _economic_tick / _calculate_income
    OpponentMixin,     # scripted_actions / _opponent_actions
    ObservationMixin,  # observation / _entity_payload
    RewardMixin,       # _potential / _reward_after_window
): ...
```

**为什么顺序可以随便排？** 原 `MicroDuelEngine` 没有任何方法调用 `super()`，也没有
`__slots__`/`@property`/`__init_subclass__`，因此各 mixin 的方法表严格不重叠，MRO 只影响
属性查找的先后，对行为没有影响。现在的顺序仅按「一局的执行顺序」排列，便于阅读。

**约定（维护者必读）**：

- mixin 内的方法只通过 `self` 访问状态，**不新增实例属性**。实例属性全部由
  `MicroDuelEngine.__init__` 与 `SpawnMixin.reset()` 建立。
- 因此跨 mixin 的调用（如 `MovementMixin._move` 调 `CombatMixin._distance`）在运行时
  一定成立；mypy 单独检查某个 mixin 文件时会报未声明属性，这是该模式的已知代价。
- 下面这些名字是**测试与外部代码的硬依赖**，拆分后必须继续存在：
  `_opponent_actions`、`_pending_reward`、`_apply_one`、`_move`、`_move_terrain`、
  `_assign_group_path`、`_repath_terrain`、`_path_stale`、`_reset_unit_path`、
  `_step_toward`、`_try_offset`、`_distance`、`_select_target`、`_visible_to`、
  `_damage`、`_check_done`、`_calculate_income`、`_potential`、`_reward_after_window`、
  `_entity_payload`、`_spawn_terrain_units`。

## 3. 调用顺序（一局的生命周期）

```
MicroDuelEngine.__init__          engine.py
  ├─ make_terrain_config()        config.py   （仅 terrain 模式重算地图尺寸）
  └─ self.reset(seed=0)           spawn.py

RwppDuelEnv / bridge 每步:
  adapter.reset(seed=...)         adapter.py → SpawnMixin.reset
      ├─ 重建 funds / resource_slots / stats / economic_events
      ├─ _spawn_terrain_units() 或 arena 网格出生
      └─ _phi_prev = _potential(); _pending_reward = 0.0

  adapter.step(actions)           adapter.py → MicroDuelEngine.step_decision
      ├─ self.events = []
      ├─ apply_actions(actions, team)         actions.py → _apply_one ×N（≤64）
      ├─ _opponent_actions(team)              opponents.py
      ├─ apply_actions(opponent_actions, 1)   actions.py
      ├─ advance()                            engine.py
      │     └─ _tick() × decision_interval    engine.py
      │           ├─ _economic_tick()         economy.py
      │           ├─ for unit in alive_units():
      │           │     ├─ _select_target()   combat.py
      │           │     ├─ _distance/_damage  combat.py
      │           │     └─ _move()            movement.py
      │           │           └─ terrain ? _move_terrain → _path_stale /
      │           │                _repath_terrain / _step_toward / _try_offset
      │           └─ _check_done()            combat.py
      │     └─ _pending_reward = _reward_after_window()   reward.py
      ├─ observation(team)                    observation.py
      └─ 合并 reward / terminated / truncated / receipts / events 后返回
```

`advance()` 是**唯一**改写 `_pending_reward` 的地方；`_reward_after_window()` 同时负责把
`economic_events` 清零，因此「每个决策窗口一次」的语义必须保持。

## 4. 数据流

```
            ┌──────────────┐
 action ──▶ │ ActionMixin  │ ── 校验通过 ──▶ 直接改 self.units / self.funds / slots
 dict[]     │ _apply_one   │ ── 拒绝 ─────▶ 返回 ErrorCode 字符串（STALE_TICK 等）
            └──────────────┘
                    │
                    ▼ mutate in place
            ┌──────────────────────────────────────────┐
            │ 共享可变状态（全部挂在 engine 实例上）      │
            │  tick / done / result / units / funds /   │
            │  resource_slots / stats / events /        │
            │  economic_events / _phi_prev / _rng       │
            └──────────────────────────────────────────┘
                    │                          │
      ┌─────────────┴──────────┐   ┌───────────┴────────────┐
      ▼                        ▼   ▼                        ▼
 MovementMixin            CombatMixin              RewardMixin
 (改 unit.x/y/path_*)   (改 hp/alive/stats/events) (_potential/_phi_prev)
                                                          │
                                    ObservationMixin ◀────┘
                                    (只读快照：不修改任何状态)
```

关键不变量：

- **arena 模式（`terrain is None`）绝不触碰** `TERRAIN_*` 常量与 `unit.path*` 字段，
  走的是原始直线代码路径；terrain 模式才启用 A* 与水域碰撞。
- `ObservationMixin` 与 `MicroDuelEngine.snapshot/evaluate` 是纯只读的。
- `Unit` 用 `@dataclass(slots=True)`，字段顺序即构造顺序，**不要重排字段**（`Unit(...)`
  有大量位置无关但依赖默认值的行为）。

## 5. 兼容 shim

`src/ironzero/mock_engine.py`（80 行）是纯 re-export：

```python
from .engine import *                    # noqa: F401,F403
from .engine import ACTION_KINDS, MicroDuelEngine, ...   # 显式名单，保证 IDE/静态分析可见
__all__ = [...]                          # 兜底
```

它另外 `import math / random / dataclass / field / replace / Any / Iterable / Sequence`，
只是因为原模块曾把这些名字泄漏到命名空间里。shim 内**不得**写任何实现逻辑。

## 6. 验证方式

```powershell
cd E:\qwq\强化学习笔记\铁锈战争

# 相关子集
python -m pytest -q tests/test_mock_engine.py tests/test_terrain.py tests/test_env.py tests/test_fog.py tests/test_economy.py

# 导入契约
python -X utf8 -c "import sys; sys.path.insert(0,'src'); from ironzero.mock_engine import MicroDuelEngine, ACTION_KINDS, EXTRACTOR_TYPE_ID, MicroDuelAdapter, make_terrain_config, ORDER_IDS, BUILDER_TYPE_ID; print('ok')"

# 全量（基线 209 passed）
python -m pytest -q
```

拆分时另做过一次「逐字节差分」验证：对同一组 `(config, opponent, seed, terrain)` 分别跑
旧实现与新包，比对 `observation()`、`snapshot()`、`evaluate()`、`receipts`、
`latest_reward_components`、`funds`、`_pending_reward`，覆盖 scripted/idle/random、
开/关经济、三档 `vision_radius`、terrain 模式 4 个 seed —— 全部一致。
