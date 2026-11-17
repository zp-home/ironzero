# 内置 AI 不进攻：根因与修复

> 结论先行：**桥梁的问题，不是引擎的问题。**
> `--unlock-ai` 除了解锁 `aiLimitedUse` 之外，还顺手把 `AI.aggressiveMode = true` 打开了。
> 这个标志在引擎里**不是"更凶"，而是"不干活"**：它会让 AI 停止建造、停止防守。
> 于是被"解锁"的电脑对手永远只有指挥中心能产出的 builder + scout，
> 造不出工厂/坦克/炮兵，也就没有可以拿去进攻的部队 —— 在正常回合上限内，
> 它一次都不会打到我们。

> **一个诚实的限定**（见 §5）：如果放任跑到远超回合上限的 6 万 tick，
> 这些 scout 靠数量堆积最终也会漏上来把基地磨掉。但那是"漏水淹死"，
> 不是 AI 在打仗；在 20056 tick 的真实回合里它一次都接触不到。
>
> 还须注意：这个症状只出现在**需要现场造兵**的地图上。预置大军 + 无经济的
> `basicAI` 剧本图（如 25v25 竞技场）上 `aggressiveMode` 几乎无害，AI 照样冲锋 ——
> 见 §3.1。

## 1. 症状

`--unlock-ai` 打开时，对局在整段时间里都是"我方挂机 → 敌方囤几个小兵 → 超时"：

| 观测（Lake 2p，20056 tick = 321 s，全程不发送任何动作） | `--unlock-ai` 开 | `--unlock-ai` 关 |
|---|---:|---:|
| 敌方单位类型总数 | **3**（commandCenter, builder, scout） | **12** |
| 最近接敌距离 | 1291 px | 781 px |
| 是否出现过工厂 | **从未** | 第 1671 tick 就有 landFactory |
| 是否出现过坦克/炮兵 | **从未** | 有 |

> 注：`--unlock-ai` 这个开关同时改了两件事，所以这张表只是**相关性**。
> 到底是哪一半坏了，由 §3 的单变量实验分离出来。

顺带纠正一个长期误判：Lake 地图其实**只放了 4 个单位**（双方各一个指挥中心 + 一个 builder），
所以 `unlock-ai: unlockedUnits=2` 本来就是对的 —— 那正是敌方的 2 个单位，不是漏解锁了 8 个。
之前文档里"敌方 2→22"的那些增量，是 AI 自己在整局里造出来的；
而 `aggressiveMode` 开着时它照样能造到 22+ —— 只是全部是 scout（见 §5 的 60000 tick 实测）。

`docs/观测与奖励改造清单.md` 里记录的 39000 tick 实测
（`own=1 enemy=22 bridge_result=None`）就是这张表第一列的直接后果：
对面那 22 个兵全是 scout，没有坦克也没有炮兵，所以我们不动作也不会被迅速打穿；
我方也从来没去打 —— 于是引擎整段时间都不判胜负（详见 §5）。

## 2. 根因：`aggressiveMode` 是"剧本角色"开关，不是"进攻性"开关

引擎里全仓库只有 6 处引用 `aggressiveMode`，其中 3 处是读写序列化，真正生效的只有 3 处：

| 位置 | 行为 |
|---|---|
| `AI.java:1246-1267` `getAttackModeForUnit` | `aggressiveMode` → `AttackMode.aggressive` |
| `Base.java:229-232` `chooseBuildingToBuild` | **`aggressiveMode` 为真时直接 `return null`** |
| `Base.java:1245-1252` `updateBuildLogic` | **`aggressiveMode` 为真时跳过 `defendBase()`** |

第一处看起来是"更凶"，其实**方向恰好相反**：`AttackMode.aggressive` 只把自动索敌半径
`getTargetSearchRange` 加 **180 px**，而默认的 `outOfRange` 加的是 **250 px**
（`OrderableUnit.java:3688-3696`）。也就是说这个标志**把索敌半径缩小了 70 px**。
后两处则是**彻底的禁用**：

- `chooseBuildingToBuild()` 返回 `null` ⇒ `Base.updateBuildLogic()` 永远拿不到要建的建筑
  ⇒ 不造矿场、不造工厂、不造炮塔；
- 跳过 `defendBase()` ⇒ 基地不再对靠近的敌人下令。

于是 AI 只能靠 `Base.buildUnits()` 走"指挥中心菜单"那一档。
Lake 的指挥中心只能产 builder 和 scout（`scout.ini: builtFrom_1_name: commandCenter`），
所以整局就只有这两样 —— 与实测的 3 种类型完全吻合。

`TriggerEngine.java:341` 是引擎自己设置这个标志的唯一地方，条件是地图声明了 `basicAI` 属性。
也就是说它是给**地图作者做剧本**用的，语义是"这个队伍只用你发给它的兵打仗"
（不给它经济，也不让它造东西），本来就不是普通遭遇战的进攻开关。
桥梁把它当成"让 AI 打我"，是因为名字看起来像。

还有一点容易误判：`aggressiveMode` **并不阻止** `AIUnitGroup` 成立或行军
（`AI.java:1890` 那道闸门纯粹是单位计数的算术，flag 不参与）。
它是靠**把单位饿死**来间接阻止进攻：没有工厂就没有战斗单位，
防御小组凑不满 8 个，攻击小组的闸门永远打不开。
所以在"地图预置大军 + 无经济"的 `basicAI` 剧本图上，
开了这个标志的 AI 反而仍然会成组冲锋 —— 那正是它被设计出来的用途。

## 3. 因果验证（三步，全部只用观测，不改引擎）

同一张 Lake、同样的 tick 数、同样不发送任何动作，只改变一个变量：

1. **原始桥梁**：`--unlock-ai` 开 → 3 种单位 / 1291 px；关 → 12 种 / 781 px。
   （只是相关性，因为开关同时改了 `aiLimitedUse` 和 `aggressiveMode` 两件事。）
2. **只改一行**：复制桥源码，把 `unlockAiTeams()` 里的 `((AI) team).aggressiveMode = true;`
   注释掉，其余完全不动，编译进临时 runtime 的副本。
   结果：`--unlock-ai` 开 → **13 种单位 / 164 px**，工厂、矿场、坦克全部出现。
   ⇒ `aiLimitedUse` 是对的，坏的是 `aggressiveMode`。
3. **难度对照**（`aggressiveMode` 关掉、难度分别 0/2/3）：
   敌兵数 31 → 49 → 104，最近距离 781 → 470 → 103 px，接触逐级增强且方向单调。
   ⇒ 解锁后的 AI 是可用的对手，难度旋钮也确实在起作用。

### 3.1 一个反例，反而把机理讲得更清楚

25v25 竞技场（IronZero25）**不是**能区分两种配置的实验台。同样跑 12000 tick：

| 配置 | 结果 | 最近接敌 | 我方 HP 收尾 |
|---|---|---:|---:|
| `--unlock-ai`（原桥梁，`aggressiveMode` 开） | timeout | 124 px | 4170 → 1900 |
| `--unlock-ai` + 不设 `aggressiveMode` | **defeat** | 106 px | 4170 → 10 |
| 不开 `--unlock-ai` | timeout | 1052 px（等于初始距离，全程 0 位移） | 4170 |

**第一行也打了。** 原因：`aggressiveMode` 不阻止 `AIUnitGroup` 成立和行军
（`AI.java:1890` 那道闸门是纯单位计数，flag 不参与），它只是**不造东西**。
而竞技场有 25 辆预置坦克、不需要经济 —— 该建的本来就没什么可建，
所以这个标志在这里几乎无害，AI 照样成组冲锋。
（这张地图还自带 `basicAI=true`，引擎在 `TriggerEngine.java:341` 会自己把
`aggressiveMode` 打开 —— 也就是说竞技场上"原桥梁"和"补丁"其实**都是**开着这个标志在跑。
两者的差别只剩下"补丁把引擎设的那次清掉了吗"这一个细节，本来就很小。）

真正被这个标志毁掉的是**需要现场造兵的地图**：Lake 只给双方各 2 个单位，
战斗单位必须靠工厂产出。而 `aggressiveMode` 恰恰把工厂这条路掐死了。
这就是为什么 Lake 上差别是"3 种单位 vs 10 种单位"，而竞技场上几乎没差别。

> 第三行（不开 `--unlock-ai`）依然是有价值的对照：全程 0 位移、距离恒等于初始值，
> 说明 `aiLimitedUse` 不清就真的是**完全挂机**，这一半修复是必需且有效的。

## 4. 修复

`src/ironzero_bridge/java/ironzero/IronZeroBridge.java`，三处删掉 `aggressiveMode`：

- `unlockAiTeams()` —— 只清 `aiLimitedUse`，不再碰 `aggressiveMode`；
- `installAiPilot()` —— `--ai-pilot` 的 teacher 同样不设；
  否则 BC 数据集里的"老师"整局只造 builder 和 scout
  （这正是之前录出来的 `hold: 13057 / attack_move: 1519 / move_group: 489` 那种分布的来源）；
- 类顶注释同步更正，写明为什么不能设。

修复后用真实源码重新编译、在临时 runtime 副本里跑真机验证：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| **Lake 20056 tick**（需要造兵的地图） | 3 种单位，最近 1291 px，工厂**从未**出现 | **10 种单位**，最近 780 px，第 1671 tick 就有 landFactory |
| Lake 60000 tick | 3 种单位，输 | 32 种单位，输（但是合成部队） |
| IronZero25 12000 tick（预置大军，不需要造兵） | timeout | defeat（两者都打得很凶，差别小 —— 见 §3.1） |

`python -m pytest tests -q` → **506 passed**。

> 生产 runtime `E:\IronZeroBuild\rwpp-runtime-1.15\extensions\ironzero-bridge.jar`
> 全程未被修改（SHA256 前后一致 `23A8C8DB…`）。
> 所有实验都在 `%TEMP%` 下的 runtime 副本里进行，需要时用
> `scripts\build_rwpp_extension.ps1` 把修复正式编译进去。

## 5. 一个诚实的限定：6 万 tick 与 39000 tick 实测的对应

如果故意把回合上限拉到 60000 tick（约 16 分钟游戏时间）并保持不动作，
开着 `aggressiveMode` 的 AI 最终**也会**把基地打掉：

| 60000 tick，不动作 | 结果 | 最近接敌 | 单位类型数 |
|---|---|---:|---:|
| `aggressiveMode` 开（原桥梁） | defeat | 92 px | **3** |
| `aggressiveMode` 关（补丁） | defeat | 102 px | **32** |

两者都会输，但方式完全不同：原桥梁是**几十个 scout 一滴一滴漏上来磨掉基地**
（3 种类型、没有坦克没有炮兵，纯靠数量堆），补丁后是**真正的合成部队推进**
（32 种类型，含 mechFactory、battleShip、mechArtillery）。

所以准确的说法不是"永不进攻"，而是：

> **在真实回合上限（20056 tick）内，被 `aggressiveMode` 关掉建造能力的 AI
> 攒不出能打到我们的部队，也不会组织进攻。** 我这两次 20056 tick 的实测里，
> 最近的一次也还在 1000 px 外（1291 / 1711 px）。
> 放到 60000 tick，它靠 scout 人海才会勉强摸到基地，且永远打不出像样的仗。
>
> 这与 `docs/观测与奖励改造清单.md` 里那个 39000 tick 实测
> （`own=1 enemy=22 bridge_result=None result=None`）是吻合的：
> 我这两次 60000 tick 实测里，scout 群第一次摸到基地是在 25000~29000 tick；
> 39000 tick 已经越过了那个点，所以那时 scout 确实在啃基地（我方掉了 1 个单位），
> 只是啃得很慢、也没打穿，引擎因此整段时间都不判胜负。

同理，修复后 Lake 在 20056 tick 内仍然是 `timeout` —— 但原因变了：
以前是"对面没有兵"，现在是"对面在建经济、扩张、逐步接敌，只是 321 s 内还没打到基地"。
这时 `timeout` 是一个诚实的结论（我方确实没在允许时间内赢），而不是引擎缺陷的遮羞布。

因此 `episode_ticks` 的推导公式（`对角线/66 px/s × 6 × sqrt(队数-1)`）在修复后值得复核：
它当初是照着"对面不进攻"的节奏定的，现在对面会真的打过来，可能需要更长。

## 6. 复现

```powershell
# 原始行为（原桥梁）
python runs/_scratch/ai_probe/probe_ai_motion.py --tag lake_unlock --port 5611 `
  --ticks 20056 --extra "--unlock-ai --win-mode native"

# 难度对照
python runs/_scratch/ai_probe/probe_ai_motion.py --tag lake_d3 --port 5619 `
  --ticks 20056 --extra "--win-mode native --enemy-ai-difficulty 3"

# 用打过补丁的 runtime 副本（设置 IZ_RUNTIME 指向副本）
$env:IZ_RUNTIME='<temp>\rwpp'
python runs/_scratch/ai_probe/probe_ai_motion.py --tag fix_lake --port 5660 `
  --ticks 20056 --extra "--unlock-ai --win-mode native"
```

每个 probe 都会写出 `<tag>.json`（含 `trace`、`min_enemy_closest_to_us`、
`enemy_types`）和对应的游戏日志 `<port>.log`，用于复核。
