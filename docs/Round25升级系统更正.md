# Round 25：推翻 rounds 19-20 的错误结论——升级系统从未损坏

## 重大更正

Rounds 19-20 断定“升级被扣款但队列项被引擎丢弃”。**这个结论是错的**，因为它
建立在一条根本不派发的路径上。

### 证据 1：`_send_direct_order` 对 upgrade 完全无效

`scripts/probe_upgrade_paths.py` 对比两条派发路径（同种子、同经济、资金 3372）：

| 派发方式 | 结果 |
|---|---|
| `env._send_direct_order({kind: upgrade_building, ...})` | **资金只增不减**（3376→5543），进度恒 0，无任何扣款 |
| `env.step(action)` with kind=`upgrade_building` | **扣款 1400**（3372→1981），进度 0→0.183→0.375→…→0.951 |

Rounds 19-20 的所有“手工派发”实验都走的第一条路径。它不发包，所以“队列项被
丢弃”实际上是“订单从未发出”。

### 证据 2：升级确实完成，收入真实上涨

`scripts/probe_upgrade_functional.py` 双盲对照（同种子、同经济，仅差是否派发升级）：

| 配置 | 矿场类型 | tech_level | 收入 |
|---|---|---|---|
| 对照（不升级）| `extractorT1` | 1 | **26.0** |
| 派发升级 | **`extractorT2`** | **2** | **30.0** |

**升级系统工作正常。**

### 证据 3：重复派发会重启升级

`queue_progress` 从 0.951 掉回 0.086，正是“重复订单重启任务”的经典特征。
已修复：新增 `_upgrade_in_flight` 闩锁，升级进行中时课程阶段不再重复派发。

## 发现 2：建筑被包含进移动编组，其队列被取消

`addOwnOrderableUnits(command, unitGroup, excludeBuilders)` 只排除 `isBuilder()`，
**没有排除其他建筑**。矿场（建筑但非 builder）因此进入 broadcast 组（group 4），
每次 `attack_move` 都重新指派它。

已修复：`excludeBuilders` 现在同时排除 `isBuilding()`。建筑本来就不能移动，
给它下移动指令没有任何收益，只会取消它的建造队列。

## 发现 3（未解决）：升级与扩张争用唯一的建造者

修复后课程阶段能派发升级（`issued=1~2/局`），但矿场仍停在 T1。追踪显示
`queue_progress` 恒为 0.0——但矿场的这个字段本来恒为 0（已知引擎怪癖，
见 `IronZeroBridge.canUpgrade` 注释：extractor 的 buildProgress 永远是 0.0）。

功能测试之所以成功，是因为那时建造者**完全空闲**（无其他建造任务）。而课程
中 `_issue_late_expansion` 每决策都在给建造者派发建矿场订单，建造者永远没空
去施工升级。

这是下一个明确的修复点：升级阶段需要让扩张阶段让出建造者，或改用独立的
建造者去执行升级。

## 本轮修复清单
1. `_upgrade_in_flight` 闩锁（防止重复派发重启升级）— `src/ironzero/rwpp/env.py`
2. 建筑排除出移动/攻击编组 — `src/ironzero_bridge/java/ironzero/IronZeroBridge.java`
3. 三个诊断探针：`probe_upgrade_paths.py` / `probe_upgrade_functional.py` /
   `probe_upgrade_trace.py` / `probe_upgrade_curriculum.py`
