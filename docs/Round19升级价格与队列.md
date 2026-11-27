# Round 19：T2 升级价格确定为 1400，但队列项被引擎丢弃

## 发现 1：真实升级价格是 1400（已修复）
Round 18 卡在"升级合法但买不起，价格未知"。本轮通过 bridge 调用引擎自身的
`SpecialAction.getPriceValue()` 得到确切答案：
```
[ironzero] upgrade rejected: building=25 available=true affordable=false price=1400
```
`MicroDuelConfig.extractor_upgrade_cost` 从 1200 改为 **1400**（此前低估 200，
导致所有基于它的储备门槛都少留了钱）。

## 发现 2：扣款成功但队列项被丢弃（未解决）
修复价格后，升级被接受并**确实扣款 1400**：
```
step=406 tick=4071 ISSUED#1 funds=5.0      <- 从 1405 扣到 5.0，正好 1400
```
但矿场状态始终是：
```
('extractorT1', tech_level=1, queue_length=0, order_type='idle')
```
即**钱扣了，队列项没了**。也尝试了与 `produceUnit` 相同的入口
`Factory.executeSpecialAction()`（`produceUnit` 的注释明确说
`CommandControllerCommand` 路径会让队列项停在 progress -1.0），结果相同。

引擎侧 `Extractor extends Factory`，且 `Extractor` 有
`upgradeToLevel2Action` / `upgradeToLevel3Action` 静态字段和
`completeQueueItem()` / `setTechLevel()` 方法，说明升级确实存在，只是
`executeSpecialAction` 之后队列项被丢弃（可能因为升级需要建造者到场施工，
而 `FactoryBuildQueue` 对升级类项目的处理路径不同）。

## 发现 3：早期 rush 无优势（已排除）
假设"在敌方 T2 成型前速攻"（唯一快速胜利发生在 tick 4717）。实测
attack_at=2/3/4（8 种子）全部 0/8，峰值兵力 18-24：
```
attack_at=2: wins=0/8 peak=[22,23,19,24,22,18,21,18]
attack_at=3: wins=0/8 peak=[22,23,23,15,18,22,16,22]
attack_at=4: wins=0/8 peak=[22,23,22,23,22,22,22,21]
```
早攻不改变结果，且峰值兵力反而更高——说明兵力不是限制因素。

## 当前状态总结（19 轮）
- 难度 1 / Lake 胜率：**0-12.5%**（历史最佳 1/8）
- 难度 0 / Lake 胜率：**50%**（证明战术框架有效）
- 已修复 8 个真实工程 Bug：种子转发、菜单死锁、资金争用、第二工厂停滞、
  多建造者、升级阶段三重锁、升级价格
- 唯一未攻克的深层问题：T2 升级的队列项被引擎丢弃

## 下一步
T2 升级需要引擎层面的进一步逆向（查看 `FactoryBuildQueue.queueOrCancel`
对 `SpecialActionBuild` 的处理，或 `completeQueueItem` 的触发条件）。
这是唯一能对冲敌方 T2 空军的路径，值得继续投入。
