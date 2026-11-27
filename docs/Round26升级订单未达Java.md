# Round 26：升级订单已上线但 Java 侧未进入分支（精确定位）

## 结论先行

本轮把问题从"升级不工作"精确缩小到"**订单在 Python 侧已构造并发出，但 Java 侧
没有进入 `upgrade_building` 分支**"。

## 证据链

### 1. 订单确实在发送的 payload 里
`scripts/probe_upgrade_spy.py` 包装了 `stepping.send_step`，捕获真正上网的
action 列表：
```
SENT step=325 tick=3251
  upgrade=[{'kind': 'upgrade_building', 'unit_group': 4, 'building_entity_id': 25}]
  all_acts=[{'kind': 'move_group', 'unit_group': 0, 'x': 24.0, 'y': 24.0},
            {'kind': 'upgrade_building', 'unit_group': 4, 'building_entity_id': 25}]
```
订单格式正确，`building_entity_id=25` 指向真实存在的 `extractorT1`。

### 2. Java 侧完全没有日志
`upgradeBuilding()` 的每一个提前返回分支现在都会打印一行
（`NOT_FOUND` / `NOT_ORDERABLE` / `NOT_OWN_TEAM` / `NO_REAL_ACTION` /
`getSpecialAction threw` / `NO_SPECIAL_ACTION`），加上原有的
`upgrade dispatched` 和 `upgrade rejected`。

实测 bridge 日志（`runs/_rwpp_ppo/game-31670-*.log`）**一行 upgrade 相关输出
都没有**，日志正常结束于 `bridge finished frames=4000`。

结论：`upgradeBuilding(long)` 从未被调用，或 `applyStep` 的分支判断没有命中
`"upgrade_building"`。

### 3. 已排除的原因
- **不是** `Extractor` 类型问题：`javap` 确认
  `Extractor extends Factory`，`instanceof Factory` 必然为真。
- **不是** 价格问题：`canAffordForUnit` 失败会打印 `upgrade rejected ... price=`。
- **不是** 派发路径问题：本轮已证明 `_send_direct_order` 不发包（见 Round 25），
  而 `step(action)` 路径确实发包（本探针捕获到了）。
- **不是** `executeSpecialAction` 的问题：功能测试证明它能完成
  `extractorT1 -> extractorT2`（收入 26 -> 30）。

## 本轮修复（保留）

1. `IronZeroBridge.upgradeBuilding` 每个提前返回分支加诊断日志。
2. `queueLength(factory)` 辅助方法（替代原先编译失败的
   `factory.buildQueue.buildQueue.size` 内联访问 —— 该字段是包级私有，
   导致 javac 报错，而之前的构建脚本用 `Select-String "error:"` 匹配**中文**
   错误输出，永远匹配不到，于是**静默沿用了旧 jar**）。
3. `_curriculum_actions` 中：同一 dispatch 里刚下过 upgrade 时跳过 production，
   避免生产把钱花掉导致升级施工无钱可用。
4. 构建脚本必须用 `powershell -ExecutionPolicy Bypass -File` 调用；
   `Select-String "error:"` 对中文 JDK 输出无效，必须检查 `$LASTEXITCODE`。

## 下一步（明确）

在 `applyStep` 的循环里加一行打印每个 `kind` 和 `actions` 长度，确认：
- `actions` 数组长度是否为 2（而不是 1）；
- 第二个元素的 `kind` 字符串实际是什么。

这是唯一还没观测到的环节。
