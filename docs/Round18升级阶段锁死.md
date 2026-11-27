# Round 18：升级阶段的三重锁死与 T2 价格之谜

## 发现 1：RushStrategy 硬编码 build_order 移除了 upgrade 阶段（已修复）
所有探针脚本使用 `scripts/probe_lake_winrate.py` 的 `RushStrategy`，它把
`build_order` 硬编码为 `("extractor","factory","production","expansion","attack")`
——**不含 upgrade**。实测：
```
reset 前 stage_names: ('upgrade','extractor','expansion','factory','production','defence','attack')
reset 后 stage_names: ('extractor','factory','production','expansion','attack')
allows('upgrade'): False
_issue_upgrades 在 400 决策中被调用 0 次
```
所以 17 轮以来 `economy_max_upgrades` 即使设成 4 也完全无效。已修复：
`max_upgrades > 0` 时把 `"upgrade"` 插到 build_order 最前面。

## 发现 2：升级阶段的资金门槛永远不可达（已修复）
`need = max(credit_reserve*0.5, extractor_upgrade_cost*1.1) = 1320`，而生产阶段
每决策把资金抽到 700 储备，实测资金在 700-1054 间震荡，从未到 1320。
已改为 `need = max(reserve*0.5, extractor_upgrade_cost)` = 1200。

## 发现 3：生产阶段抽干升级预算（已修复）
即使门槛降到 1200，生产阶段每决策都把钱花到 700，升级只在军队饱和后
（tick ~17000 / 19674）才凑够钱，来不及回本。已加入两阶段逻辑：
军队 ≥1 之后，生产阶段额外保留升级价格作为下限。

## 发现 4（未解决，核心障碍）：引擎拒绝升级，因为实际价格远高于 1200
修复以上三点后，升级订单稳定发出（每 seed 1-4 单），但**建筑从未离开 T1**。
在 bridge 中加入诊断后得到确切答案：
```
[ironzero] upgrade rejected: building=25 available=true affordable=false
```
**`available=true`（升级合法）但 `affordable=false`（买不起）。**

价格二分实测（`funds_at_first_issue`）：
| credit_reserve | 首次发单时资金 | 结果 |
|---|---|---|
| 700 | 1206 | affordable=false |
| 1500 | 1207 | affordable=false |
| 3500 | 无军队，升级阶段被阻塞 | — |
| 手动在 872 资金派单 | 872 | affordable=false |

即引擎的 T1→T2 矿场升级价格 **高于 1207**，而 Python 侧
`MicroDuelConfig.extractor_upgrade_cost = 1200.0` 是低估的。由于升级价格
未在观测 payload 中暴露（extractor 实体的 34 个字段里没有价格），只能靠
bridge 调用引擎 API 查询或逐步抬高储备实测。

## 结论
T2 科技路线在代码层面已经打通（三个阶段都修好了），但**卡在价格未知**。
这是下一步唯一需要解决的具体问题：确定引擎真实的 upgrade 价格，然后让
生产阶段的储备下限与之匹配。这是 18 轮以来最有希望的方向——因为 T2 是
唯一能对冲敌方 T2 空军的手段。

## 已提交的代码改动
- `scripts/probe_lake_winrate.py`：RushStrategy 支持 upgrade 阶段
- `src/ironzero/rwpp/env.py`：升级资金门槛 1320→1200；生产阶段两阶段储备
- `src/ironzero_bridge/java/ironzero/IronZeroBridge.java`：升级拒绝诊断日志
- 153 个测试通过
