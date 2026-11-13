# `ironzero.observation` — 观测编码与动作掩码

> 本包由原 `src/ironzero/observations.py`（393 行）拆分而来。
> **只搬结构，不改行为**：所有数值、归一化分母、掩码分支与拆分前逐字节一致。
> 旧路径 `ironzero.observations` 仍然可用（薄 shim，仅 re-export）。

## 职责

把游戏侧适配器（mock engine 或未来的 RWPP 桥）产出的**原始观测 payload**
（`dict`）转换成策略网络消费的**扁平 `float32` 向量**，并从同一份 payload
派生各动作头的 **bool 掩码**。两者必须共用一套布局约定，否则掩码与观测会错位，
所以它们放在同一个包里、由同一份 `MicroDuelConfig` 驱动。

## 模块

| 文件 | 行数 | 内容 |
| --- | ---: | --- |
| `__init__.py` | 44 | 唯一对外出口：re-export 全部公开名 + `__all__` |
| `encoding.py` | 110 | `encode_observation()`：payload → 扁平向量 |
| `features.py` | 86 | `_entity_features` / `_unit_group` / `_cell` / `_nearest` / `GROUP_DIM` |
| `terrain_channel.py` | 100 | `terrain_passable_channel` / `_resolve_terrain_path` / `_terrain_passable_channel` / `_downsample_passable` / `REPO_ROOT` |
| `masks.py` | 90 | `build_masks` / `masks_as_lists` / `flat_mask_vector` |

## 数据流

```
                    raw state dict（own_entities / visible_enemy_entities /
                    global_state / resource_slots / tick）
                          │
        ┌─────────────────┼──────────────────────────────┐
        │                 │                              │
   encode_observation   build_masks                 masks_as_lists
        │                 │                              │
        │   features.py ──┤  （实体特征 / 单位分组 /       │
        │   逐实体特征     │    网格坐标 / 最近敌人）      │
        │                 │                              │
        │   terrain_channel.py                           │
        │   （读 TerrainGrid → 降采样 → 通道 3）          │
        ▼                 ▼                              ▼
  float32[flat_obs_dim]  {"kind","group",      ironzero/1 消息里的
                          "x","y","target"}     action_mask / unit_group_mask
        │                                       / target_mask / position_mask
        ▼
  PPO / 策略网络（observation 与 mask 必须同源于一次 payload）
```

### 观测向量的四段布局

偏移量按 `MicroDuelConfig` 声明累加，`encode_observation` 末尾有
`assert offset == cfg.flat_obs_dim` 兜底：

1. **global 段**（`global_dim`，默认 15）
   时间进度、双方兵力、双方血量、双方质心、质心差、
   常数 1、`funds / funds_scale`、`income / income_scale`、已占矿点比例。
2. **己方实体段**（`max_entities × entity_features`）
   按 slot 索引，空位补零；每个单位 10 维，含占位掩码、绝对坐标、
   HP 比、分组、命令、相对质心位移、最近敌人方向与距离。
3. **敌方实体段**（同上尺寸）
   与己方同构，但 `nearest` 传 `None`，方向/距离相对己方质心。
4. **粗粒度控制网格**（`map_channels × map_grid × map_grid`）
   - 通道 0：己方单位计数
   - 通道 1：敌方单位计数
   - 通道 2：`clip(通道0 - 通道1, -1, 1)`
   - 通道 3：静态地形可通行先验（仅 `map_channels > 3` 且配置了 `terrain_path`）

   **只有 0..2 三个"计数"通道**除以 `units_per_team` 归一化；通道 3 本身已在
   `[0, 1]`，不再缩放。
5. **标识块 identity**（`2 × max_entities × type_features`，默认 `type_features = 9`）
   己方一段 + 敌方一段，同 slot 顺序，回答「这是什么、能不能打」。
   位于**向量最末尾**（`type_block_offset` 之后），所以前面所有列的偏移不变。
   每个 slot 9 维：

   | 维 | 含义 |
   |---: | --- |
   | 0 | occupancy（空位补 0，与几何块一致） |
   | 1 | `is_building` 建筑 |
   | 2 | `is_factory` 可产兵工厂 |
   | 3 | `is_builder` 工程车 |
   | 4 | `is_air_unit` 空中单位 |
   | 5 | `build_progress` 在建进度（1.0 = 已完工） |
   | 6 | 类型哈希 `[0, 1)`（区分其余类型） |
   | 7 | `can_attack` 能不能打（0/1） |
   | 8 | `attack_range / 1000`，clamp 到 `[0, 1]` |

   > **为什么有 7/8 维**：前 7 维能说清「这是什么」，但说不清「靠近它会不会死」。
   > 策略只能靠第 6 维类型哈希从奖励里归纳「哈希 0.236 的东西冲上去会死」，
   > 样本效率极低，实测表现为**建造者造完矿场后直接冲向敌方基地送死**。
   > 引擎侧由 `IronZeroBridge.entityJson()` 的 `Unit.canAttack()` /
   > `OrderableUnit.getMaxAttackRange()` 提供（异常降级为 `false` / `0`）。
   >
   > **归一化分母是固定常数 `ATTACK_RANGE_SCALE = 1000.0`**，不是 `cfg.map_width`：
   > 地图宽度属于场景，用它做分母会让同一辆坦克在 1000px 微竞技场读作 0.130、
   > 在 Lake(2600px) 读作 0.050，特征含义随地图漂移、checkpoint 无法迁移。
   >
   > **7→9 是尾部追加**：每个 slot 的 stride 从 7 变 9，但**第 0..6 维在 slot 内的
   > 偏移与语义完全不变**。老 checkpoint 的 trunk 列迁移见
   > `training.ppo._column_source`（注意 stride 变化时它会放弃重映射、退化为
   > 前缀拷贝，见该函数 docstring）。


### 地形通道的失败语义（关键约定）

`terrain_passable_channel()` 的返回类型是 `np.ndarray | None`：

- 配置没有地形通道（`map_channels <= 3` 或 `terrain_path` 为空）→ 返回 `None`，**不告警**；
- 路径解析不到文件 → 发 `RuntimeWarning`（"terrain file not found ..."），返回 `None`；
- 文件存在但解析失败 → 发 `RuntimeWarning`（"terrain file could not be parsed ..."），返回 `None`。

返回 `None` 时调用方把通道 3 留零。**观测维度永不因文件缺失而改变**——维度在
`MicroDuelConfig` 构造时就固定了。`_terrain_passable_channel` 带
`lru_cache(maxsize=16)`，同一 `(path, resolution)` 只解析一次；测试依赖这条缓存
语义，不要去掉。

## 掩码约定

`build_masks` 的输出顺序严格对应 `MicroDuelConfig.action_nvec`：
`kind(K) → group(G) → x(P) → y(P) → target(T)`，其中 `K == len(ACTION_KINDS)`。

- **kind 头**：战术命令恒可选；经济类动作（`build_extractor` / `build_structure` /
  `produce_unit` / `upgrade_building` / `repair_target`）需要"有活着的小兵 + 付得起"。
  若整头全为 False，回退为前 5 个战术命令——**绝不发出全 False 的头**。
- **group 头**：payload 带 `group_id` 时按实际分组置位；只有 `entity_id` 时
  **全部 `groups` 个槽位置位**（真机桥不传 `group_id`，而游戏侧按
  `unit.id % groups` 推导；收窄到 group 0 会让 75% 的部队失去控制）。
  最后一槽恒为 True，表示"广播到所有存活分组"。
- **target 头**：前 `min(len(enemies), target_slots - 1)` 个槽位可选，最后一槽恒为
  True 表示"无目标"，保证采样永远合法。

## 依赖方向

```
observation/ ──> terrain/        （TerrainGrid.from_file，仅在读地形时）
             ──> micro_config    （布局常量与归一化分母）
             ──> mock_engine     （ACTION_KINDS / BUILDER_TYPE_ID / ORDER_IDS）
```

禁止反向依赖：`terrain/` 不认识 `observation/`，`micro_config` 不认识两者。

## 自测

```powershell
python -m pytest -q tests/test_env.py tests/test_fog.py tests/test_terrain.py
python -m pytest -q tests/test_rwpp_25v25.py tests/test_training_pipeline.py
python -X utf8 -c "import sys; sys.path.insert(0,'src'); from ironzero.observations import build_masks, encode_observation, flat_mask_vector, masks_as_lists; print('ok')"
```
