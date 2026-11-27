# Round 15 发现：针对性攻击产能（enemy production targeting）

## 背景
14 轮以来，所有探针都让军队攻击敌方 **commandCenter**（4000 HP，位于防空火力
中心）。但 `objective` 判定的真实条件是
`enemies.mainBuildings == 0 || !enemies.hasCapability`，而 `mainBuildings` 只统计
**非炮塔、非矿场的工厂建筑**（IronZeroBridge.capabilityOf，第 2494-2500 行）。
`hasCapability = factories > 0 || builders > 0`。

所以胜利可以通过摧毁敌方 landFactory（1200 HP）/ airFactory（1000 HP）和建造者
达成，**不必摧毁 4000 HP 的指挥中心，也不必赢得空战**。这是一条此前从未
测试过的路径。

## 实现
`scripts/probe_target_production.py` 每决策用 `attack_target`（broadcast group=4）
指向最近的敌方工厂建筑，回退顺序为：工厂 → 建造者 → 任意建筑。

## 结果：兵力峰值几乎翻倍

| 配置 | 兵力峰值 | 胜率 |
|---|---|---|
| 攻击 commandCenter（此前 14 轮）| 13-19 | 0/32 |
| **攻击敌方工厂（本轮）** | **24-30** | 0/8 |

针对性攻击让军队存活率大幅提升（peak 24-30 vs 13-19），说明**攻击正确的目标
本身就是一种有效的战术**——军队不再白白冲进指挥中心的防空火力网。

## 但仍未达成 objective：两个具体障碍

### 障碍 1：军队被拦截在距离工厂 411-1167px 处
`probe_reach.py` / `probe_concentrate.py` 实测军队最近接近距离：
```
seed 71000: closest=536    seed 71002: closest=411
seed 71003: closest=801    seed 71006: closest=1000
```
军队在 y≈314 处与敌方主力交战，无法推进到工厂所在的 y≈210。压制 curriculum
的 attack 阶段（`opening_attack=False`）后 closest 从 1167 改善到 411-536，
证明攻击指令确实在干扰集中火力，但仍不够。

### 障碍 2：敌方产能持续增长
末态敌方工厂数量为 **4-5 座**（landFactory x2 + airFactory x2 + seaFactory），
而我们只有 2 座。即使单座工厂被打到 25 HP（probe_reach 实测 id=31 hp=25.0），
敌方也能迅速补建。

### 难度 0 对比（同样针对性攻击）
```
难度 1: 0/8
难度 0: 1/8（seed 71006，tick 9104，peak 10）
```
难度 0 也仅 1/8，说明针对性攻击本身还不够成熟，不是单纯的难度加成问题。

## 结论
针对性攻击是**正确的方向**（兵力峰值翻倍），但需要：
1. 更强的集中火力——当前军队分散在多个目标间
2. 突破敌方主力拦截才能触及工厂
3. 更快的产能以对抗敌方 4-5 座工厂

这是下一轮的方向，而非"难度 1 不可达"的最终结论。
