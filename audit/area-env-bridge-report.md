# 审计报告：真机环境 / 桥接 / 协议 / 托管服务

审计范围：`src/ironzero/rwpp_env.py`、`bridge.py`、`tcp_adapter.py`、`protocol.py`、
`src/ironzero_bridge/java/ironzero/IronZeroBridge.java`、mock 侧（`mock_engine.py`、`mock_bridge.py`、
`mock_server.py`、`env.py`、`observations.py`、`replay.py`）、`scripts/rwpp_real_service.py`、
桥接测试（`test_bridge_tcp.py`、`test_protocol.py`、`test_rwpp_env.py`、`test_rwpp_25v25.py`、`test_mock_engine.py`）。

> 本报告只读代码 + 实测（pytest、直接用 Python 复算掩码/解码、校验 Java 实际发出的 NDJSON 能否被
> `protocol.parse_message` 接受、抽取已安装 jar 的字符串常量、读取历史真机录制流）。**未修改任何源码。**
> 凡本文标注「实测」的结论均有可复现命令；无法确认处标「待确认」。

---

## 1. 一句话结论

Python 侧定义了 10 种动作、2 套解码器、3 套时间常数、5 套血量归一化、5 套胜负判定，而 Java 桥只实现了
其中 5 种动作、1 套时间常数、0 套掩码，且两侧的 NDJSON 契约（`ReceiptMessage` 的 `extra="forbid"`）当场对不上；
`observations.py` 里被占位注释替换掉的 group 兜底逻辑、以及桥里硬编码的 `income=0.0`，使「经济 + 多组微操」
这条主线在真机上**要么静默无效、要么直接抛异常**，而现有测试只覆盖到 mock，真机路径基本没有回归网。

---

## 2. Python 动作空间 ↔ Java 桥接能力 双向对照表（最重要）

### 2.1 三份「动作种类」清单互相不一致

| 来源 | 内容 | 行号 |
|---|---|---|
| `protocol.ActionKind`（StrEnum，协议真源） | 10 项：move_group, attack_move, attack_target, stop, hold, build_extractor, build_structure, produce_unit, upgrade_building, repair_target | `protocol.py:27-37` |
| `protocol.Action` 判别联合（pydantic） | 10 个模型，`extra="forbid"`，各带必填字段 | `protocol.py:93-163` |
| `mock_engine.ACTION_KINDS`（tuple，**位置即动作 id**） | 同样的 10 项，顺序与上面一致 | `mock_engine.py:34-45` |
| `MicroDuelConfig.action_nvec[0]`（策略头宽度） | **7**（注释写 "build_mine, build_factory"，与实际 kind 名不符） | `micro_config.py:101-108` |
| Java `applyStep` 分支 | **5** | `IronZeroBridge.java:603-657` |

`action_nvec[0]=7` 与 `ACTION_KINDS` 长度 10 直接冲突：策略只能采样到索引 0..6，
而两个解码器都用 `% len(ACTION_KINDS)`（`env.py:179`、`rwpp_env.py:335`）做取模，索引 7/8/9 永远取不到。

实测（`PYTHONPATH=src`）：

```
ACTION_KINDS: [(0,'move_group'),(1,'attack_move'),(2,'attack_target'),(3,'stop'),(4,'hold'),
               (5,'build_extractor'),(6,'build_structure'),(7,'produce_unit'),
               (8,'upgrade_building'),(9,'repair_target')]
nvec[0] = 7
kind idx 7 -> [{'kind': 'produce_unit', 'unit_group': 0}]      # 协议非法：缺 unit_type
kind idx 8 -> [{'kind': 'upgrade_building', 'unit_group': 0}]  # 协议非法：缺 building_entity_id
kind idx 9 -> [{'kind': 'repair_target', 'unit_group': 0}]     # 协议非法：缺 target_entity_id
produce_unit action INVALID per protocol: step.actions.0.produce_unit.unit_type
```

即：**索引 7/8/9 由策略不可达；即使被外部注入，产出的动作也不满足 `protocol.py` 自身的契约。**

### 2.2 完整双向对照表（10 种动作 × Python 定义 × Python 解码 × Java 分支）

| # | Python ActionKind（行号） | 必填字段（protocol.py） | Python 解码器产生的载荷 | Java dispatch 分支 | Java 实际行为 |
|---|---|---|---|---|---|
| 0 | `move_group` (p:28) | x, y | `{kind, unit_group, x, y}`（env.py:187-191 / rwpp_env.py:342-344） | `IronZeroBridge.java:619-628` | ✅ 实现。`createCommandForTeam` + `setMoveWaypoint` |
| 1 | `attack_move` (p:29) | x, y | 同上 | `:619-628` | ✅ 实现。`setAttackMoveWaypoint` |
| 2 | `attack_target` (p:30) | target_entity_id | `{kind, unit_group, target_entity_id}`（rwpp_env.py:350-358）；env.py:193-201 | `:629-636` | ⚠️ 部分。`findUnitById`（`:709-720`）在**全图**找目标，不校验可见性/敌我；找不到则 `command==null` → `rejected++`（无原因码） |
| 3 | `stop` (p:31) | — | **被改写成 hold**：`_hold_action` 永远发 `{"kind":"hold"}`（rwpp_env.py:359-368） | `:637-641` → `holdGround(group, false)` | ⚠️ 语义丢失：`stop` 在真机与 `hold` 完全等价，`stop` 分支的 `guardArea=false` 永远走不到区别 |
| 4 | `hold` (p:32) | — | `{kind:"hold", unit_group}` | `:637-641` → `holdGround(group, true)` | ✅ 实现（逐单位一条命令 + `AttackMode.guardArea`） |
| 5 | `build_extractor` (p:33) | slot_id, [builder_entity_id] | rwpp_env.py:345-346 发 `{kind, slot_id}`（**无 unit_group/builder**） | **无分支** | ❌ **被丢弃**：落到 `:643-647` 的 `command==null` → `rejected++`，无 `rejected_reason`、无 `error` 消息 |
| 6 | `build_structure` (p:34) | structure_type, x, y | rwpp_env.py:347-349 发 `{kind, structure_type:"land_factory", x, y}` | **无分支** | ❌ **被丢弃**（同上） |
| 7 | `produce_unit` (p:35) | unit_type, [factory_entity_id] | 策略不可达（nvec=7）；外部注入时只产出 `{kind, unit_group}` → 协议非法 | **无分支** | ❌ **被丢弃** |
| 8 | `upgrade_building` (p:36) | building_entity_id | 策略不可达；外部注入 → 协议非法 | **无分支** | ❌ **被丢弃** |
| 9 | `repair_target` (p:37) | target_entity_id, [builder_entity_id] | 策略不可达；外部注入 → 协议非法 | **无分支** | ❌ **被丢弃** |

**Python 有 / Java 无：5 项** —— `build_extractor`(`protocol.py:33`)、`build_structure`(`:34`)、
`produce_unit`(`:35`)、`upgrade_building`(`:36`)、`repair_target`(`:37`)。
它们在 `IronZeroBridge.java:619-641` 的 if/else 链里没有对应分支，`command` 保持 `null`，
计入 `rejected`（`:645-646`），**但 Java 既不回 `error` 也不回单条 `receipt`**，Python 侧只能看到一个不透明的
`rejected_count`（而这个字段本身还破坏了协议，见 §7.1）。

**Java 有 / Python 无：0 项（在 `applyStep` 内）**。但**消息级**存在反向缺口：

| 消息类型 | Python 发送方 | Java 处理方 | 结论 |
|---|---|---|---|
| `step` | `rwpp_env.py:615`、`record_rwpp_replay.py:245` | `IronZeroBridge.java:593-595` | ✅ |
| `shutdown` | `rwpp_env.py:950` | `:589-592` | ✅（但 Python 发的是裸 `{"type":"shutdown"}`，**不是** `ShutdownMessage`，缺 `protocol`/`message_id`，见 §5） |
| `reset` | `tcp_adapter.py:80-88`、`protocol.py:239-245` | **无** | ❌ Java 侧无 reset 概念，整局靠重启进程；`ResetRequest` 只服务 mock |
| `hello` | `bridge.py:158-164` | **无**（Java 只在启动时主动 `sendHello`，`:226-231`） | ❌ Java 不读客户端 hello，握手是单向的 |
| `heartbeat` | `protocol.py:318-321`、`mock_bridge.py:146` | **无** | ❌ 真机无心跳，`POLICY_TIMEOUT`/`BRIDGE_DISCONNECTED` 永远发不出 |
| `error` | —（Java 从不发） | — | ❌ Java 解析失败直接 `continue`（`:579-583`），任何非法动作都被静默吞掉 |
| `receipt` | —（Java 发，`:650-656`） | `rwpp_env._await_observation:937-939` 只 append 不做校验 | ⚠️ 见 §7.1，用 `protocol` 校验必失败 |

### 2.3 两套 Python 解码器已经分叉

| 能力 | `env.py:decode_actions`（mock 路径） | `rwpp_env.py:decode_action`（真机路径） |
|---|---|---|
| kind→动作映射 | `% len(ACTION_KINDS)`（`:179`）| `% len(ACTION_KINDS)`（`:335`） |
| 广播组 | 展开成 `cfg.groups` 条动作（`:191`，`_target_groups`） | 单条 `unit_group=G`（`:344`），依赖 Java `GROUP_BROADCAST` |
| `build_extractor` | **只发 `{kind, unit_group}`，缺 `slot_id`** → mock `_apply_one:341-343` 判 `INVALID_MESSAGE` | 发 `{kind, slot_id}`（`:346`） |
| `build_structure` | 不发 `structure_type/x/y` → mock 判 `INVALID_MESSAGE` | 发（`:349`） |
| `attack_target` 越界回退 | `stop`（`:201`） | `hold`（`:359`） |
| 雾下目标索引 | 用 `state["target_ids"]`（未雾化） | 用 `obs_state`（雾化，`:603`） |

实测 mock 路径（`MicroDuelEnv(seed=0)`，`funds=1000` ≥ 两档成本，掩码打开 kind 5/6）：

```
kind 5 {'kind':'build_extractor','unit_group':0} -> receipts [{'accepted': False, 'rejected_reason': 'INVALID_MESSAGE'}]
kind 6 {'kind':'build_structure','unit_group':0} -> receipts [{'accepted': False, 'rejected_reason': 'INVALID_MESSAGE'}]
```

即：**经济动作在 mock 里也是坏的**，而 `tests/test_env.py` 的 `test_masked_random_actions_are_legal`
（`:146`）恰好只断言 kind ∈ 5 种旧动作，才让这个缺陷长期潜伏（该用例现因 `kind` 掩码变成 7 元素而失败，见 §3）。

---

## 3. 功能完成度表格

| 功能 | Python 侧 | Java 侧 | mock 侧 | 完成度 | 证据 |
|---|---|---|---|---|---|
| 移动 / 攻击移动 | ✅ | ✅ | ✅ | 完成 | `protocol.py:93-103` / `IronZeroBridge.java:619-628` / `mock_engine.py:461-479` |
| 攻击目标 | ✅ | ⚠️ 不校验可见性 | ✅ | 可用但有语义缺口 | `IronZeroBridge.java:629-636,709-720` |
| stop / hold | ✅ 定义 | ✅ | ✅ | 完成 | `IronZeroBridge.java:637-641,682-707` |
| 经济建造 5 种 | ✅ 定义 + 解码 | ❌ 丢弃 | ✅ 模拟 | **真机 0%** | §2.2 |
| 多组控制（4 组 + 广播） | ⚠️ 掩码回归 | ✅ `id % 4` | ✅ `unit.group` | **条件性回归** | §4.1 |
| 观测掩码（`ActionMasks`） | 自算，忽略桥的 | 发空数组 | 发真值 | 契约装饰化 | `IronZeroBridge.java:289` / `observations.py:294-332` |
| 迷雾（vision_radius） | ✅ 观测级过滤 | ❌ 全量流 | ✅ 引擎级 | 两条不同实现 | `rwpp_env.py:371-397` / `mock_engine.py:625-631` |
| 终局胜负 | ✅ 3 层回退 | ✅ native + annihilation | ✅ | 5 套实现分叉 | §8.4 |
| 地形通道 | ✅ | 不涉及 | ✅ A* | 完成 | `observations.py:120-204` |
| 回放 | ✅ | 不涉及 | ✅ | 两条写入路径 | `replay.py` / `record_rwpp_replay.py` / `play_rwpp_policy.py` |
| 心跳 / 断线检测 | 定义 | ❌ | ✅ | **真机 0%** | `protocol.py:318-321` |
| 错误回执 | 定义 13 种码 | ❌ 一种都不发 | 发 4 种 | **真机 0%** | §2.2 |
| LLM/规则战略层 | ✅ 只改奖励 | 不涉及 | ✅ | 完成（设计如此） | `rwpp_env.py:823-882` |

---

## 4. 死代码清单

### 4.1 死代码 · group 掩码（含回归）

- `observations.py:314` `# ... (keep existing group logic)`、`:322` `# ... rest of the function remains similar`
  —— 占位注释，替换掉了 HEAD 中的兜底分支（`git show HEAD:src/ironzero/observations.py` 可见
  `group_ids_known` / `elif own: group_mask[:cfg.groups] = True` 及其"否则 75% 部队不可指挥"的注释）。
- 现逻辑 `observations.py:313-317`：只用 `unit.get("group_id") or 0`，桥不给 `group_id` 时所有单位塌成 group 0。
- 实测（真实历史真机录制流 `runs/_rwpp_bridge_smoke.jsonl`，2026-11-06，20 帧 obs，实体字段只有
  `entity_id/entity_type_id/hp/is_alive/is_building/is_visible/max_hp/team_id/x/y`，**无 group_id**）：

  ```
  own=2 ids=[23, 24]  group mask = [True, False, False, False, True]
  => 4 组里只有 group0 可指挥，group1/2/3 被 mask（= 75% 组失效）
  ```
- 需要说明的**条件性**：当前工作树 Java 已在 `entityJson` 输出 `group_id = unit.id % GROUP_COUNT`
  （`IronZeroBridge.java:311`，为未提交改动），实测 25 个单位 id=1..25 时 residue 覆盖 0..3 →
  `group mask = [True,True,True,True,True]`。**所以"75% 不可指挥"只在缺少 `group_id` 的桥上成立**
  （旧 jar / 未重编 runtime / 任何第三方桥）。当前仓库内 `.runtime/bridge-dist/ironzero-bridge.jar`
  与 `E:\IronZeroBuild\rwpp-runtime-1.15\extensions\ironzero-bridge.jar` 均为 2026-11-11 09:20:37 的
  新构建（抽取 class 字符串确认含 `group_id`/`enemy_teams`/`win-mode`），因此**本机现在不会复现**
  —— 但兜底逻辑被删是事实，且单测仍在守护它，见 §3 失败用例。
- 更严重的是**观测与掩码自相矛盾**：`observations._unit_group`（`:252-271`）明确"桥不给 group_id 时用
  `entity_id % groups` 推断"，而 `build_masks` 不做同样推断。实测：

  ```
  mask group      : [True False False False True]
  obs group feat  : [1, 1]        # 策略看到"单位属于 group 1"
  => 策略认为自己能指挥 group1，掩码却禁止 —— 学到的策略必然被掩码纠正成 group0
  ```

- Java 侧 `GROUP_COUNT=4`（`:55`）是硬编码，而 Python `cfg.groups` 可配（`micro_config.py:44`）。
  `cfg.groups=2` 时 Python 组头宽 3，Java 仍按 `id % 4` 过滤（`:671`、`:695`）→ group 头与命令选择错位。**待确认**（当前没有 groups≠4 的真机配置）。

### 4.2 死代码 · 未被引用的定义

| 位置 | 符号 | 说明 |
|---|---|---|
| `protocol.py:407-428` | `class SequenceGuard` | 顺序/重复决策守卫，**生产代码零调用**，仅 `test_protocol.py:129` 用 → `DUPLICATE_DECISION`(`:43`) 实际永不触发 |
| `protocol.py:59-62` | `DoneResult.DEFEAT/DRAW/TIMEOUT/FAILURE` | 全仓仅 `VICTORY` 被测试引用（`test_protocol.py:169`） |
| `protocol.py:67-68` | `HeartbeatStatus.PAUSED/STOPPING` | 无引用 |
| `protocol.py:47,51,52,54` | `TARGET_INVALID` / `POLICY_TIMEOUT` / `BRIDGE_DISCONNECTED` / `UNSUPPORTED_VERSION` | 全仓从不构造 |
| `protocol.py:185-187` | `ActionMasks.build_mask/produce_mask/slot_mask` | `masks_as_lists`（`observations.py:335-345`）从不填；Java 也不发 |
| `bridge.py:336` | `load_message_line` | 零引用 |
| `bridge.py:342` | `to_json_line` | 零引用 |
| `bridge.py:125-191` + `tcp_adapter.py:28-140` | `BridgeClient` / `TcpDuelAdapter` / `NdjsonConnection` | **只被 `tests/test_bridge_tcp.py` 使用**；真机走 `rwpp_env._LineReader`（`:254-293`）自己实现一套 NDJSON 解析 → 同一协议两套客户端 |
| `mock_bridge.py:13-14,21,23` | `BridgeClosed/BridgeError/BridgeTimeout`、`ACTION_KINDS`、`canonical_json` | 未使用 import |
| `rwpp_env.py:26,31` | `sys`、`Iterable` | 未使用 import |
| `rwpp_env.py:513,965-970` | `self._log_handle` | 只在 `__init__` 置 `None`，全仓再无赋值 → `_shutdown_game` 的关闭分支是死分支 |
| `rwpp_env.py:528-529` | `_BoxSpace` / `_MultiDiscreteSpace` | 仅自我赋值，训练/评测从不读 `observation_space`/`action_space` |
| `rwpp_env.py:50,1006` | `DECISION_TICKS = 10` | 导出但零引用；真正的 10 在 Java（`DECISION_INTERVAL`），而 `micro_config.decision_interval=30` → 三方不一致（§8.2） |
| `observations.py:24` | `GROUP_DIM = 4` | 零引用 |
| `env.py:156-157` | `flat_action_mask` | 零引用 |
| `env.py:248` | `__all__` 里的 `"uuid"` | 把 import 当导出泄漏 |
| `mock_engine.py:48` | `BUILDER_TYPE_ID = 2` | 零引用（经济系统里没有 builder 实体） |
| `mock_engine.py:988` | `_ = member` | 为消 unused 变量的赋值，本身是垃圾 |
| `replay.py:138` | `load_replay` | 零引用 |
| `IronZeroBridge.java:49` | `FRAME_PERIOD_MILLIS = 16` | 零引用（真正用的是 `frameMs` 字段，默认 `:66`） |
| `IronZeroBridge.java:144-146` | `if (frames % 30 == 0 && writer == null) break;` | 看门狗：`writer` 只在 `writeLine` 的 IOException 里置 null（`:731`），且下一轮 `sendObservation` 已经提前 return；该 break 实际不可达 |
| `src/ironzero/training/curriculum.py:1-17` | `CurriculumScheduler` | 新增未跟踪文件，全仓零引用；`if update_count < 1000` 与 `else` **返回同一个 config**，第二个分支是纯粹死代码 |
| `src/ironzero/test_tutorial.py:1-25` | `test_tutorial_env` | 未跟踪的调试脚本放进 `src/ironzero/` 包内（pytest 不会收集成 test_*? 实际文件名 `test_tutorial.py` 会被 `testpaths=tests` 排除，但 `from ironzero.test_tutorial import *` 会带上 `print`）；`kind=5 (build_extractor)`（`:18`）用的正是 §2.2 里 Java 会丢弃的动作 |

### 4.3 死代码 · mock 与真机重复实现且已分叉

| 能力 | mock 实现 | 真机实现 | 分叉点 |
|---|---|---|---|
| 单帧推进 | `mock_engine.advance/_tick`（`:516-601`） | Java `updateFrameInternal`（`:122`） | tick 语义不同：mock 1 decision = 30 tick，Java 1 decision = 10 tick |
| 动作校验 | `_apply_one`（`:335-511`）返回 8 种 reason | Java `applyStep`（`:603-657`）仅聚合计数 | 真机无 reason 码 |
| 经济 | `build_extractor/build_structure/produce_unit/upgrade_building`（`:340-444`） | **不存在** | 整块功能只在 mock 存在 |
| 迷雾 | 引擎级 `_visible_to`（`:625-631`） | 观测级 `apply_vision`（`rwpp_env.py:371-397`） | 真机是"事后过滤"，AI 仍然全知 |
| 胜负 | `_check_done`（`:896-908`） | Java `detectGameResult`（`:346-418`） | 5 套判定见 §8.4 |
| 血量归一化 | `_potential` 用 `units_per_team*unit_hp`（`:1077-1082`） | `hp_fractions` 用开局 ref（`rwpp_env.py:416-435`） | 同名指标不可比 |
| 观测字段 | 有 `order_type/attack_target_id/group_id/cooldown_frac/build_progress/production_queue_length`（`:1054-1072`） | Java 只发 `unit_type/is_building/is_factory/is_builder/is_operational/group_id`（`:297-313`） | `observations._entity_features` 读 `order_type`（`:220`）→ 真机永远是 `idle`，该特征恒 0 |
| 资源槽 | `resource_slots`（`:1038-1049`） | **不发** | 导致 `rwpp_env.py:346` 恒取 slot 0（§7.4） |

`mock_engine.py` 里 7 处 `if hasattr(self, "economic_events")`（`:369,401,441,586,882,1091,1130`）
是"给未来/给真机预留"的防御式写法，而 `reset()` 无条件创建该属性（`:200-209`）→ 7 个恒真分支，属半成品痕迹。

---

## 5. 垃圾代码清单

| 位置 | 问题 |
|---|---|
| `observations.py:305,312,318` | 三处只有空白的行（尾随空格），配合占位注释像是 IDE 半途保存 |
| `observations.py:314,322` | `# ... (keep existing group logic)` / `# ... rest of the function remains similar` —— 直接写进生产函数的伪代码注释 |
| `rwpp_env.py:702` | 单行 400+ 字符，把 `"income"` 与 `"bonus_build"` 两个奖励分量塞进一行；魔法数 `5.0`（建成奖励）与 `/10.0`（income 缩放）无出处 |
| `rwpp_env.py:759-764` | 魔法数 `0.08`（血量优势阈值）出现两次，`*0.8`（超时奖励折扣）出现两次，均无命名常量 |
| `rwpp_env.py:752-757` | `is_micro_battle = win_mode=="annihilation" or "ironzero" in map or "ironzero" in map_name or self.micro.units_per_team > 0` —— 最后一项恒真（`units_per_team` 恒 >0），使整个条件恒 `True`，`:767-768` 的 `else: self.last_result = "timeout"` **不可达** |
| `rwpp_env.py:346` | `slot_id = target_index % len(state.get("resource_slots", [{}]))`：真机不发 `resource_slots` → 永远 `% 1 = 0`；若显式传空列表则 `ZeroDivisionError`（**实测**） |
| `rwpp_env.py:950` | `sendall(b'{"type":"shutdown"}\n')` —— 手写裸 JSON，绕过了 `protocol.ShutdownMessage`（缺 `protocol`/`message_id`，若对端是 `BridgeServer` 会被 `parse_message` 判 `INVALID_MESSAGE`） |
| `rwpp_env.py:962` | `except Exception:  # noqa: BLE001` 把 wait 失败统一当"还活着"然后 `kill()`，丢失超时/OSError 区别 |
| `rwpp_env.py:784-785`、`857-858` | `except Exception:  # noqa: BLE001` 两处静默吞异常（strategy 失败即降级），且都只写注释不记日志 |
| `rwpp_env.py:176-201` | 模块级可变全局 `FRAME_MS_WARNED` 做"只警告一次"节流；同文件另有 `print` 式诊断（`effective_frame_ms:196-200`） |
| `rwpp_env.py:218,250` | `open(log_path,...)` 后立刻 `handle.close()`，注释解释"子进程自己持有句柄"；但 Windows 上 `stdout=handle` 的句柄由 Popen 复制，父进程提前关存在竞态风险。**待确认** |
| `mock_engine.py:1099-1104,1110` | `getattr(cfg, "extractor_reward", 0.5)` 等 6 处 `getattr(config, ..., default)`：字段**确实存在**（`micro_config.py:90-95`），默认值是纯冗余，且默认值和真实字段值不同步时会产生静默分歧 |
| `mock_engine.py:216,260,277-282` | 生怪布局里 `72.0/18.0/10.0/0.06/0.25/0.75` 一堆裸数字，且 `_spawn_terrain_units` 与 `reset` 的 arena 分支是**复制粘贴**（`:212-241` vs `:244-298`，差异仅 `nearest_passable_world` 与 bounds） |
| `mock_engine.py:968` | `kind = self._rng.choice(ACTION_KINDS)`：随机对手会抽到 `build_extractor` 等经济动作，随后套用 `{"kind":kind,"unit_group":group}`（`:987`）→ 必然 `INVALID_MESSAGE`，随机对手有 ~50% 动作是空转 |
| `mock_bridge.py:161-162` | `elif isinstance(message, ErrorMessage): return` —— 收到错误消息什么都不做，静默 |
| `bridge.py:256-257` | `except BridgeError: pass` 吞掉整条会话的传输错误，`self._error` 不记录 |
| `bridge.py:282-283` | 只把 `ValueError` 转成协议错误，其他异常直接冒泡到 `_serve` 的 `BaseException` 分支 |
| `tcp_adapter.py:136-137` | 收到 `ErrorMessage` 直接 `raise BridgeError`，`recoverable=True` 的语义被丢弃 |
| `rwpp_real_service.py:808-809` | `except Exception: pass`，`ensure_arena_map` 失败被完全吞掉 → 后面 `terrain` 落到硬编码 `IronZero25.terrain.json`（`:812`），可能出现"地图是 10v10、地形是 25v25"的静默错配 |
| `rwpp_real_service.py:50-51,117` | `ACTIVE_PROCESSES[job_id] = []` 只清空不删键；`STOPPED_JOBS` 只增不减；`JOBS` 从不淘汰 → 长跑服务内存单调增长（**泄漏**） |
| `rwpp_real_service.py:72-73,92-95` | `_run_managed_proc` 在 `JOBS_LOCK` 内维护列表，但 `stop_job`（`:98-118`）持锁 `kill()` → 持锁执行 IO，`kill` 阻塞时全局串行化 |
| `rwpp_real_service.py:85-90` | `except Exception:` 分支重新 `kill` 并 `return -1, ""`，把子进程 stderr 全部丢弃 |
| `rwpp_real_service.py:225,282` | 用 `stdout[-1200:]` / `[-1500:]` 两个不同魔法长度截尾日志 |
| `rwpp_real_service.py:895,931` | `frame_ms` 由 `_start_ppo` 透传，但 `frame_ms` 在 `:1000` 也被记录；`_start`（`/infer`）路径 `:959` 读 `frame_ms` 却**从未传给 `_run_player`/`_run_recorder` 的 vision 校验**（vision 在整个 `/infer` 路径完全丢失，见 §8.3） |
| `record_rwpp_replay.py:56` | `parser.add_argument("bridge_args", nargs="*")` —— **位置参数无法接收 `--` 开头的参数**，实测 `--ticks 100 --win-mode annihilation` → `SystemExit 2: unrecognized arguments: --win-mode`。即该"透传"特性完全不可用 |
| `record_rwpp_replay.py:294-295` | `"vision_radius": 260, "fog": False` —— 硬编码 260 却声明无雾，自相矛盾；`:71-79` 是 `play_rwpp_policy.py:62-69` 的复制粘贴（unit_type 推断表一字不差） |
| `record_rwpp_replay.py:364-366` | `process.wait(timeout=60)` 后 `process.kill()`，但 `finally` 里 `log.close()` 在 kill 之前，Windows 上偶发句柄占用。**待确认** |
| `play_rwpp_policy.py:200-203` | 超时公式 `ticks * (frame_ms + 6.0) / 1000 + 120.0` 与 `rwpp_real_service._episode_timeout`（`:138-139`，`+90.0`）**是同一公式的两份拷贝、常量不同**（6.0/120 vs 6.0/90） |
| `IronZeroBridge.java:172` | `System.out.println("[ironzero] arguments=" + args)` 调试残留：把整条启动参数打印进游戏日志（无害但属残留） |
| `IronZeroBridge.java:246-270, 428-447, 684-707` | 同一段"遍历 `Unit.unitList` → 跳过 `removed \|\| hp<=0` → 过滤 `team.teamId != playerTeam.teamId`"复制了 **5 次**（`sendObservation`、`tally`、`addOwnOrderableUnits`、`holdGround`、`countUnits`），其中 `:668` 与 `:692` 的团队过滤、`:671` 与 `:695` 的组过滤逐字重复 |
| `IronZeroBridge.java:273-293` | 用手写字符串拼 JSON（`+ "\"hp\":" + fmt(...)`），既无转义也无 schema 校验；`unit_type` 直接拼 `getInternalName()`（`:304`，未转义 → 单位名含 `"` 即破坏 JSON） |
| `IronZeroBridge.java:302-303,765-767` | 每个字段一次 `String.format(Locale.ROOT,"%.3f")`，每帧 25×2 实体 × 7 字段 ≈ 350 次格式化，热路径无必要 |
| `IronZeroBridge.java:579-583` | `catch (RuntimeException failure) { continue; }` —— 非法 JSON 行动静默丢弃，既不计数也不回执 |
| `IronZeroBridge.java:730-732` | `catch (IOException) { writer = null; }` —— 写失败后把 writer 置空，之后 `sendObservation` 开头 `if (writer == null) return;`（`:234-236`）静默停发，主循环继续跑到 tick 上限 |
| `web/backend/app.py:86 vs :102` | `RwppRecordRequest.vision_radius=0.0`（`:102`）在 `/infer` 路径从不使用（见 §8.3），却与 PPO 的 260.0 默认值（`:118`）并排摆放，读代码者极易误判 |

---

## 6. 半成品清单

| # | 半成品 | 证据 | 后果 |
|---|---|---|---|
| 1 | **5 种经济动作只有协议 + mock，没有宿主** | `protocol.py:33-37` / `mock_engine.py:340-444` / `strategy/macro_controller.py:55-123`（会产出这 5 种）/ Java 无分支 `IronZeroBridge.java:619-641` | `MacroController` 只在 `training/vec_macro_micro.py:94`、`evaluate_macro_micro.py:67` 里被用于 **mock**；真机训练（`vec_rwpp.py`）根本不构造 MacroController → "宏微分层"在真机不存在 |
| 2 | `build_extractor` 的 `builder_entity_id`、`build_structure` 的 `builder_entity_id`、`produce_unit` 的 `factory_entity_id` | `protocol.py:121,129,135` 均为可选；两套解码器都不填（`env.py:203` / `rwpp_env.py:345-349`） | 即使 Java 未来补分支，"谁去建"也没有信息 |
| 3 | `ActionMasks` 三个可选掩码 | `protocol.py:185-187` 定义；`observations.masks_as_lists`（`:335-345`）不产出；Java 发 `[]`（`:289`） | 掩码契约是装饰品，真机掩码完全由 Python 本地重算（`rwpp_env.py:549,674`），桥说了不算 |
| 4 | Java 侧错误回执 | `IronZeroBridge.java` 全文无 `"type":"error"` | `ErrorCode` 13 个码在真机永不出现；`POLICY_TIMEOUT`/`BRIDGE_DISCONNECTED` 的断线检测靠 Python socket 超时兜住 |
| 5 | 心跳 | `HeartbeatStatus`（`protocol.py:65-68`）+ `mock_bridge.py:145-152` 有；`rwpp_env` 从不发、Java 从不处理 | 长局（216000 tick）无活性检测，卡死只能等 `obs_timeout` |
| 6 | `hello` 握手方向 | Java 主动 `sendHello`（`:226-231`），但 `readClientCommands` 只认 `shutdown`/`step`（`:589-595`）；`BridgeClient.handshake`（`bridge.py:149-169`）期待对端 `hello` 响应 | 两条握手实现互不兼容；真机 `capabilities` 列表（`:229-230`）是无接收方的广播 |
| 7 | `getattr(config, ..., default)` 兜底 | `mock_engine.py:1099-1104,1110` 6 处 | **核查结论：无字段名拼错**（`extractor_reward/extractor_t2_reward/factory_reward/production_reward/building_loss_penalty/idle_funds_penalty_coef` 均在 `micro_config.py:90-95` 存在），属纯冗余防御；但 `rwpp_env.py` 内 6 处 `getattr`（`:780,852,867-869,982`）也是对**已定义属性**的兜底，`directive.intent/attack_region/rally_point` 在 `strategy/directive.py:78-84` 是必填/有默认的 dataclass 字段 |
| 8 | `_run_recorder` 与 `_run_player` 能力不对等 | `rwpp_real_service.py:172-199`（recorder 命令）**没有** `--win-mode`/`--unlock-ai`；`:267-268`（player 命令）有 | arena 地图 + `policy=scripted/none` 的 `/infer`、以及 `/train` 全部走 recorder → 无建筑地图上 native 判定立即判负（Java 自己注释了这个坑：`IronZeroBridge.java:68-71,128-134`） |
| 9 | `record_rwpp_replay` 的 "extra bridge args" | `:56` + `:149` | 位置参数收不到 `--xxx`（实测 SystemExit 2）→ 该特性从未生效；`eval_rwpp_policy.py:236-239` 想透传，实际靠 `play_rwpp_policy` 而不是 recorder |
| 10 | `/infer` 的 vision/25v25 参数 | `app.py:102` 定义 `vision_radius`；`rwpp_real_service.py:948-1047`（`_start`）**从不读** `vision_radius`；`RwppRecordRequest` 无 `units_per_team` | 推理页无法表达"25v25 + 迷雾"；只能靠 checkpoint 里带出来的 `micro`（`play_rwpp_policy.py:178`） |
| 11 | `curriculum.py` | `:8` `self.stage = 3`，`:12-17` 两个分支返回同一对象 | 课程学习是空壳 |
| 12 | `test_tutorial.py` | `src/ironzero/` 内的 untracked 调试脚本 | 把 `print` 调试代码留在包目录；用到的 `kind=5` 恰是 Java 丢弃的动作 |
| 13 | `local_maps` / `remembered_enemies` | `protocol.py:268-269` 定义；mock 发 `{}`/`[]`（`mock_engine.py:1036-1037`）；Java 发 `{}`/`[]`（`:288`） | 记忆地图/记忆敌人两个 LSTM 友好字段永远空 |
| 14 | `advance()` 的 `_pending_reward` | 只在 `mock_engine` 存在（`:516-522`） | 真机奖励完全由 `rwpp_env` 事后算，mock 与真机的 reward 语义无法对拍 |

---

## 7. 契约断裂清单

### 7.1 线格式（NDJSON）当场对不上 —— 最硬的断裂

`protocol.MessageBase` 是 `extra="forbid"`（`protocol.py:72-75`），但 Java 回执多带两个字段：

```java
// IronZeroBridge.java:650-656
+ "\"accepted_count\":" + accepted + ",\"rejected_count\":" + rejected
```

实测用协议模型解析 Java 实际发出的回执：

```
receipt parse FAILED:
  receipt.accepted_count   Extra inputs are not permitted
  receipt.rejected_count   Extra inputs are not permitted
```

影响：`tcp_adapter._await_observation`（`:121-126`）走 `EnvelopeAdapter` → 必然 `BridgeError`；
`rwpp_env._LineReader.read`（`:290-293`）用裸 `json.loads` 才侥幸躲过。
**即：官方客户端（`bridge.py`/`tcp_adapter.py`）无法与官方 Java 桥通信，只有 `rwpp_env` 的私有实现能。**

同源问题：`DoneMessage.truncated` 在 Java 里与 `terminated` 互斥（`:321`），而 `protocol.DoneMessage`
默认 `terminated: bool = True`、`truncated: bool = False`（`:305-306`）—— Java 发 `terminated:false, truncated:true`
能通过校验（实测 OK），但 `protocol` 的默认值语义与桥的实际语义相反。

### 7.2 观测维度与实体槽

| 契约项 | `micro_config.DEFAULT` | `RwppGameConfig()` | `train_rwpp` CLI | `rwpp_real_service` `/ppo` arena | `rwpp_real_service` `/infer` | 结论 |
|---|---|---|---|---|---|---|
| `units_per_team` | 8 (`:43`) | 8 (`:81`) | `DEFAULT_CONFIG` = 8 (`:66`) | 25（arena 时）(`:797`) | **不可指定** | 断裂 |
| `max_entities` | 8 (`:58`) | 8 → `resolved_max_entities` 抬到 ≥units_per_team (`:108-111`) | 8 (`:68`) | 25 | 不可指定 | 断裂 |
| `target_slots` | 9 (`:59`) | 0 → `max_entities+1` (`:113-119`) | 0 → 自动 (`:70`) | `max_entities+1` | 不可指定 | 尚可 |
| `flat_obs_dim` | 364 | 364（默认） | 364 | **768**（4×8×8 地形） | 由 checkpoint 决定 | 同一次 `/infer` 与 `/ppo` 的 obs 空间可能不同 |
| `map_channels` | 3 (`:61`) | 3 + 地形 (`:145`) | 取决于 `--terrain` | 4（强制地形 `:810-812`） | 由 checkpoint | 断裂 |

**溢出实测**（默认配置 + Artillery 实况 22 自方 / 25 敌方）：

```
obs dim 364, nvec (7,5,8,8,9)
own_count feature   = 2.75   (设计域 [0,1])
enemy_count feature = 3.125
own_hp_frac feature = 2.75   (分母 units_per_team*unit_hp = 8*210 = 1680)
target slots        = 9 个，实际 25 个敌人 → 16 个敌人无目标槽
```

`observations.py:47-63` 的全局块带 `assert g.shape[0] == cfg.global_dim`（`:64`）却**不做 clip**，
`_entity_features`（`:235-249`）也只 clamp 距离不 clamp hp 比例 —— 与 `rwpp_env.hp_fractions`（`:432-435`
明确 clamp 到 [0,1]）口径相反。

### 7.3 时间常数三方不一致

| 常量 | `micro_config` | `rwpp_env` | Java | 影响 |
|---|---|---|---|---|
| 每决策 tick 数 | `decision_interval=30`（`:33`，0.5 s @60Hz） | `DECISION_TICKS=10`（`:50`，注释称"桥的契约"）—— **导出但零引用** | `DECISION_INTERVAL=10`（`:48`） | replay meta 写 30（`replay.py:52`）→ 回放时间轴与实际差 3× |
| tick 率 | `tick_rate=60`（`:32`） | 无 | 60（`game_time = tick/60.0`，`:276`） | 一致 |
| 回合上限 | `episode_ticks=3600`（`:34`） | `RwppGameConfig.episode_ticks=3600`（`:68`） | `maxFrames=900`（`:65`，**`--ticks` 缺省时**） | 少传 `--ticks` 直接 900 |
| 单帧墙钟 | — | `frame_ms=2`（`:76`），`effective_frame_ms` 把 <1 钳到 1（`:179-201`） | `frameMs=16`（`:66`） | `record_rwpp_replay.py:52` 默认 16、`play_rwpp_policy.py:120` 默认 16、`rwpp_real_service.py:170/299/469` 默认 2 |
| `--frame-ms 0` | — | 钳到 1 并警告（`:192-201`） | 直接当 **frame delta** 传 `updateFrameInternal`（`:122`） | `record`/`play` 用 `max(0,...)`（`record:148`、`play:175`）**绕过**了 `effective_frame_ms` → 0 真的会到 Java（`rwpp_env.py:183-187` 的注释明确说这会产生不同战斗） |
| 开局/观测超时 | — | `start_timeout=180 / obs_timeout=120`（`:71-72`，**未提交改动**，HEAD 是 90/60） | `accept` 30 s（`:93,214`） | Java 30 s 内没连上就 `stop`，Python 却肯等 180 s |

### 7.4 血量归一化 5 套口径

| 位置 | 分母 | 是否 clamp |
|---|---|---|
| `observations.py:42,52-53` | `units_per_team * unit_hp`（固定 1680） | ❌（实测 2.75/3.125） |
| `rwpp_env.hp_fractions:429-435` | 开局 ref（`army_hp_reference:455-466`） | ✅ [0,1] |
| `mock_engine.evaluate:1164-1169` | `units_per_team*unit_hp` | ❌ |
| `play_rwpp_policy.state_to_frame`（经 ref 传入，`test_rwpp_scripts.py:145-151` 已验证） | 开局 ref | — |
| `record_rwpp_replay.py:99-111` | **存活单位 max_hp 之和**（分母随死亡缩小） | ❌ |
| `replay.py:105,115-116` | `units_per_team*unit_hp` | ❌ |
| `probe_ironzero25.py:167` | `FULL_TEAM_HP` | ❌ |

同一 episode 的 `own_hp_frac` 在训练页（`rwpp_env`）、推理页（`play_rwpp`）、录制页（`record_rwpp_replay`）、
回放页（`replay`）四个来源可以给出 4 个不同数值。

### 7.5 迷雾半径 6 个默认值

| 位置 | 默认 | 说明 |
|---|---|---|
| `micro_config.py:55` | `0.0`（关） | 契约默认 |
| `rwpp_env.RwppGameConfig:77` | `0.0` | 真机默认关 |
| `train_rwpp.py:81` | `260.0` | 训练默认开 |
| `rwpp_real_service.py:470,775` | `260.0` | `/ppo` 默认开 |
| `rwpp_real_service.RwppRecordRequest`（`app.py:102`） | `0.0`，且**从不透传** | `/infer` 实际取 checkpoint 值 |
| `record_rwpp_replay.py:294` | 硬写 `260` + `"fog": False` | 自相矛盾 |
| `play_rwpp_policy.py:178` | 跟 checkpoint | 唯一"正确"的做法 |
| `evaluate_rwpp.py:53-54` | 若 micro 有雾而 game 没有则**偷偷补齐** | 隐式修正，掩盖了默认值不一致 |

### 7.6 胜负判定 5 套实现

| # | 实现 | 规则 | 行号 |
|---|---|---|---|
| 1 | Java native | `defeatScreenShown`/`Team.defeated`/`wipedOut` → defeat；`victoryScreenShown` → victory；否则"任一敌方队伍存活"清零 → victory | `IronZeroBridge.java:380-417` |
| 2 | Java annihilation | 前 30 tick 免判（`ANNIHILATION_GRACE_TICKS`）；`enemySeen && enemyAlive==0` → victory；`ownAlive==0` → defeat | `:354-379` |
| 3 | `rwpp_env` 桥优先 + 回退 | `bridge_result ∈ {victory,defeat}` → terminated；`None` 时 `enemy_alive==0 and own_alive>0` → victory | `rwpp_env.py:717-736` |
| 4 | `rwpp_env` 超时判定 | `own_alive>enemy_alive && own_frac>=enemy_frac+0.08` → victory×0.8；对称 → defeat×0.8；否则 timeout | `:747-772` |
| 5 | `mock_engine` | `alive_units(1)` 空 → victory；`alive_units(0)` 空 → defeat；`tick>=episode_ticks` → timeout | `mock_engine.py:896-908` |

断裂点：Java annihilation 的 `enemySeen`（`:78,359-362`）是**进程级**状态且只在 `annihilation` 分支更新，
native 分支也复用它（`:408-416`）→ native 模式下"从没见过敌人"会被当成"没消灭完"，永远判不了胜；
`rwpp_env` 的第 4 套（0.08 阈值 + ×0.8 折扣）在 Java 已经给出 `timeout` 时才生效，
而 Java 的 `done.result="timeout"` 会被 `:742-746` 判成 truncated → 于是"同一局"既有桥的 timeout 又有 env 的
victory/defeat，`info["result"]` 与 `info["bridge_result"]` 会不一致（`rwpp_env.py:789-790` 同时上报两者，无仲裁）。

### 7.7 服务层并发 / 超时 / 清理

| 检查项 | 结论 | 证据 |
|---|---|---|
| `JOBS_LOCK` 是否覆盖所有 `JOBS` 写入 | ⚠️ `_update`/`stop_job`/`_append_replay` 都加锁；但 `run_train` 的 worker 在 `:395-412` 手动 `with JOBS_LOCK` 直接改字段，同时又调 `_append_replay`（也会加锁）→ 同一逻辑点两次取锁，`threading.Lock` 非重入但这里不嵌套，**不致命** | `:394-412` |
| 端口槽是否必然归还 | ✅ `run_infer`（`:350-351`）、`run_train` worker（`:413-414`）都是 `finally: PORT_SLOTS.put(port)` | — |
| 端口槽是否可能重复归还 | ⚠️ `stop_job` 只 `kill` 进程不回收端口；被 kill 的 worker 走 `finally` 归还 → 不会重复 put。但 `run_infer` 在 `:301` `get()` 之后若 `_update` 抛异常，`finally` 仍归还 → OK | — |
| `ACTIVE_PROCESSES` 是否会泄漏空列表 | ❌ 只 `procs.remove(proc)`（`:94-95`）与 `= []`（`:117`），**键永不删除** | `:50,73,95,117` |
| `STOPPED_JOBS` 是否清理 | ❌ 只 `add`（`:105`），从不 `discard` | `:51,105` |
| `JOBS` 是否淘汰 | ❌ 无 TTL/上限；`/jobs` 只截取最后 50 条（`:729`），字典本身无限增长 | `:38,729` |
| PPO 进程表清理 | ✅ `finally: PPO_PROCESSES.pop(...)`（`:608`）；但 `stop_job` 在 `:109-111` 读 `PPO_PROCESSES[job_id]` 时未加 `PPO_STOP` 的锁，`PPO_STOP` 是 `set` 无锁 | `:109-111,437-438,606` |
| `stop_job` 持锁 `kill()` | ⚠️ `:112-116` 在 `with JOBS_LOCK` 内 `p.kill()`；`kill` 可能阻塞（Windows TerminateProcess 一般快，**待确认**） | `:101-118` |
| `run_ppo` 轮询是否泄漏文件句柄 | ✅ `handle.close()` 在 `finally`（`:607`）；但异常分支（`:603-604`）之后 `finally` 仍执行 → OK | — |
| 是否可能永不释放的锁 | ❌ 未发现 `Lock.acquire()` 无 `with`/`finally` 的路径；`JOBS_LOCK` 全为 `with` | 全文 |
| 超时是否双层漂移 | ⚠️ 服务 `_episode_timeout = ticks*(frame_ms+6)/1000 + 90`（`:138-139`）、子进程自己再算 `+120`（`record:171`、`play:202`）、外层再 `+60`（`:198,271`） | 三层超时不一致 |

---

## 8. 修复优先级建议

### P0（让"真机主线"从不可用变成可用）

1. **动作空间收敛或补齐二选一** —— 现状是"Python 声明 10 种、策略只能出 7 种、Java 只做 5 种"的三重错位。
   - 立刻改：`micro_config.py:101-108` 的 `action_nvec[0]` 与 `mock_engine.ACTION_KINDS` 长度绑定（用
     `len(ACTION_KINDS)` 而不是字面量 7），并在 `protocol.py` 与 `mock_engine.py` 之间加一条一致性测试
     （目前 `test_macro_micro_pipeline.py:36-38` 只断言"包含"，没断言"相等"）。
   - 在 `IronZeroBridge.applyStep`（`:603-657`）为未识别 kind 补 `else { rejected++; /* 记原因 */ }`，
     至少回一条带 `rejected_reason` 的 `receipt` 或 `error`，不要静默 `rejected++`。
   - 若短期不实现 5 种经济动作：把它们从 `ActionKind`/`ACTION_KINDS` 里移除或标记 `experimental`，
     避免"动作空间 10 种"的假象（`mock_engine` 的经济实现保留给 mock-only 的 `vec_macro_micro`）。
2. **修 `ReceiptMessage` 的线格式** —— 二选一：
   - Java 删掉 `accepted_count`/`rejected_count`（`IronZeroBridge.java:655`），把计数塞进 `details`；
   - 或给 `ReceiptMessage` 显式加这两个字段（`protocol.py:281-296`）。
   现状是 `tests/test_bridge_tcp.py` 能过（mock 不发这两个字段）、真机必然解析失败，属于"测试绿、真机红"。
3. **修 group 掩码** —— 恢复 `observations.build_masks` 的兜底（无 `group_id` 时开放全部 group），
   并让它与 `_unit_group`（`:252-271`）用**同一个**推断函数，消除"obs 说 group1 / mask 禁止 group1"的自相矛盾。
   同时把 `IronZeroBridge.GROUP_COUNT=4`（`:55`）与 `cfg.groups` 对齐（透传一个 `--group-count`，或 Python 侧固定 4）。
4. **让 `_run_recorder` 也转发 `--win-mode`/`--unlock-ai`**（`rwpp_real_service.py:172-199`），
   或按 `arena` 统一在 `_start` 里决定；否则 arena 地图的任何 scripted/`/train` 任务都会立即判负。

### P1（数值契约与可复现性）

5. **统一 `decision_interval`**：删掉 `rwpp_env.DECISION_TICKS`（`:50,1006`）或让它成为唯一真源，
   并让 `replay.py:52` / `play_rwpp_policy` 的帧时间轴与 Java `DECISION_INTERVAL=10` 一致。
6. **统一血量归一化**：`observations.encode_observation:42,52-53` 加 clip，或改用 `army_hp_reference`
   的开局 ref（`rwpp_env.py:438-466`）；同时把 `record_rwpp_replay.py:99-111`、`replay.py:105` 的
   分母换成开局 ref，让训练/推理/录制/回放四页可比。
7. **`--frame-ms 0` 全链路钳到 1**：`record_rwpp_replay.py:148`、`play_rwpp_policy.py:175` 改用
   `rwpp_env.effective_frame_ms`，否则 `rwpp_env.py:183-187` 记录的"0 ms 会改变战斗结果"仍会复发。
8. **修 `record_rwpp_replay.py:56` 的 `nargs="*"`**：改成 `--bridge-arg`（`action="append"`）或
   `parse_known_args`，否则"透传 bridge 参数"永远无效。
9. **`income` 奖励分量**：Java `:281` 硬编码 `"income":0.0` → `rwpp_env.py:702` 的 `income` 项恒 0。
   要么桥真算收入（`player.credits` 增量），要么把该分量删掉/标注为 mock-only，避免"奖励分解图里永远一条直线"。
10. **`bonus_build` 的魔法 5.0 + income 的 `/10.0`**（`rwpp_env.py:702`）改成命名常量放进
    `RwppGameConfig`，与 `kill_bonus/loss_penalty/invalid_penalty` 同源管理。

### P2（清理与防御）

11. 删死代码：`SequenceGuard`、`BridgeClient`/`TcpDuelAdapter`（或让 `rwpp_env` 复用它）、
    `load_message_line`/`to_json_line`、`GROUP_DIM`、`flat_action_mask`、`DECISION_TICKS`、
    `_log_handle`、`FRAME_PERIOD_MILLIS`、`BUILDER_TYPE_ID`、未使用的 `ErrorCode` 值、
    `curriculum.py`、`test_tutorial.py`、`env.py.__all__` 的 `"uuid"`。
12. `rwpp_env.py:346` 的 `% len(resource_slots)` 加空列表保护（真机恒无 `resource_slots`，
    建议直接返回 `hold` 或跳过该 kind）。
13. `rwpp_real_service.py` 的 `ACTIVE_PROCESSES`/`STOPPED_JOBS`/`JOBS` 加淘汰（保留最近 N 个终态任务）。
14. Java `entityJson`（`:297-313`）改用 JSON 库序列化（`unit_type` 未转义，单位名含引号即破坏整帧）；
    `fmt()` 的逐字段 `String.format` 换成复用的 `StringBuilder`。
15. 补真机级测试：目前 `test_bridge_tcp.py` 只跑 mock，`test_rwpp_env.py` 用自造 `FakeBridge`
    （它发的回执**没有** `accepted_count`，所以永远测不出 §7.1）。建议用 `.runtime/bridge-dist/ironzero-bridge.jar`
    的真实常量（或 `IronZeroBridgeTest.java`）生成"金样 NDJSON"喂给 `parse_message`。

---

## 附录 A：本次 pytest 结果与失败用例定位

`python -m pytest -q` → **11 failed, 190 passed**（14.5 s）。与本审计范围相关的 6 条：

| 用例 | 失败断言 | 根因 |
|---|---|---|
| `tests/test_rwpp_25v25.py:138` `test_25v25_observation_and_action_contract` | `(7,5,8,8,26) != (5,5,8,8,26)` | `micro_config.py:105` 把 nvec[0] 从 5 改成 7（未提交改动），测试仍锁 5 |
| `tests/test_rwpp_25v25.py:178` `test_masks_without_group_id_expose_all_groups` | `masks["group"].all()` 为假 | **就是 §4.1 的回归**：`observations.py:315-317` 删掉了无 `group_id` 时开放全部 group 的兜底 |
| `tests/test_rwpp_25v25.py:381` `test_game_config_builds_25v25_terrain_micro` | `(7,...) != (5,...)` | 同 nvec[0] |
| `tests/test_rwpp_25v25.py:502` `test_ironzero25_flags_build_the_25v25_terrain_contract` | 同上 | 同上（`train_rwpp.build_game` 带出的 `micro.action_nvec`） |
| `tests/test_rwpp_25v25.py:662` `test_terrain_checkpoint_rebuilds_the_768_dim_policy` | 同上 | 同上（checkpoint 里 `asdict(micro)` 的 nvec） |
| `tests/test_mock_engine.py:143` `test_custom_config_keeps_shapes_consistent` | `7 != 5` | 同 nvec[0] |

范围外的 5 条（供交叉参考，非本区域）：`test_env.py:40,56,146`（kind 掩码 7 vs 5、`build_structure` 越界被允许）、
`test_training_pipeline.py:47,101`（`policy._masked` 维度 5 vs 7 直接 `RuntimeError`）。

**结论**：`action_nvec[0]=7` + `kind_mask=np.ones(7)` 这两个未提交改动，把 `observations`↔`policy`↔`micro_config`
三处的动作维度契约同时打破，而且 `policy.py:23-29` 的 `ACTIVE_HEADS`（5 行）与 `_active_matrix`（`:108-117`，
只处理 0..4）**没有随之一改** —— 若 kind 采样到 5/6，`_active_matrix` 给出的 `active[:,2]=kinds<=1` 为 False，
位置头与目标头不参与 log-prob，等于"建造动作没有空间位置梯度"。这是下一个必须修的洞。

## 附录 B：本次核查的 `getattr(config, ..., default)` 清单（任务要求项）

| 位置 | 表达式 | 字段是否存在 | 判定 |
|---|---|---|---|
| `mock_engine.py:1099` | `getattr(cfg, "extractor_reward", 0.5)` | ✅ `micro_config.py:90` | 冗余兜底 |
| `mock_engine.py:1100` | `getattr(cfg, "extractor_t2_reward", 0.8)` | ✅ `:91` | 冗余兜底 |
| `mock_engine.py:1102` | `getattr(cfg, "factory_reward", 0.6)` | ✅ `:92` | 冗余兜底 |
| `mock_engine.py:1103` | `getattr(cfg, "production_reward", 0.4)` | ✅ `:93` | 冗余兜底 |
| `mock_engine.py:1104` | `getattr(cfg, "building_loss_penalty", 1.0)` | ✅ `:94` | 冗余兜底 |
| `mock_engine.py:1110` | `getattr(cfg, "idle_funds_penalty_coef", 0.0001)` | ✅ `:95` | 冗余兜底 |
| `mock_engine.py:882` | `getattr(target, "is_building", False)` | ✅ `Unit.is_building` (`:97`) | 冗余 |
| `rwpp_env.py:780` | `getattr(self.directive, "to_payload", None)` | ✅ `directive.py:175` | 防御（directive 可能是任意 strategy 对象，合理） |
| `rwpp_env.py:852` | `getattr(self.strategy, "propose", None)` | ✅ `rule_strategy`/`llm_strategy` 都有 | 合理 |
| `rwpp_env.py:867-869` | `getattr(directive, "attack_region"/"rally_point"/"intent", ...)` | ✅ `directive.py:78-80` | 冗余 |
| `rwpp_env.py:982` | `getattr(value, "shape", None)` | numpy 数组 | 合理 |
| `rwpp_env.py:198` 等 | 无 `getattr(config, ...)` 用法 | — | — |

**核查结论：没有字段名拼错导致的"永远走 default"。** 全部 6 处 mock 兜底都对得上真实字段；
风险不在拼错，而在"默认值与真实字段值可能不同步"（例如把 `extractor_reward` 改成 0.9 而某处仍用 0.5）。
