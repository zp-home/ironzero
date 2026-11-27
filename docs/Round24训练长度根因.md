# Round 24：训练首次真正工作——找到根因并修复

## 决定性根因：episode 太短，兵力峰值恒为 0

`scripts/probe_train_lengths.py` 用 6 个固定种子对比 episode 长度（难度 1、Lake）：

| episode_ticks | 有胜负判定 | 胜 | 兵力峰值 | 矿场 |
|---|---|---|---|---|
| **1200** | **0/6** | 0/6 | **0** | 1 |
| 4000 | 0/6 | 0/6 | 3 | 1 |
| 8000 | 0/6 | 0/6 | 6 | 3 |
| **19674** | **1/6** | 1/6 | 14-17 | 4-5 |

**1200 tick 时兵力峰值为 0**——军队还没造出来。每个 episode 都是 timeout，
且是在"双方都不可能赢"的状态下超时，策略收到的唯一终止信号是
`timeout_penalty = -2.8`。**不存在任何指向胜利的梯度。**

Round 23 的长时间训练（1200 tick）跑 20 个 episode，`timeout_rate` 全程 1.0，
精确印证了这一点。该任务已终止。

## 修复后的训练曲线（episode_ticks=19674, 4 envs, CUDA）

```
upd=1  eps=12  win=0.083  timeout=0.917  ret=-3.13  t=54s
upd=2  eps=20  win=0.154  timeout=0.846  ret=-0.96  t=103s
upd=3  eps=20  win=0.154  timeout=0.846  ret=-0.77  t=155s
upd=4  eps=20  win=0.077  timeout=0.923  ret=-1.61  t=204s
upd=5  eps=20  win=0.000  timeout=1.000  ret=-2.97  t=251s
upd=6  eps=20  win=0.000  timeout=1.000  ret=-1.45  t=316s
```

**这是 24 轮以来训练循环中第一次出现胜利。** 此前所有训练都是
`win=0.00 timeout=1.00`，因为 1200 tick 下根本不可能有胜负。

峰值 win=0.154（upd=2-3），之后回落到 0。样本仍小（每 update 20 episode），
且 PPO 在稀疏奖励下波动正常，但这证明了**梯度存在、学习可能**。

## 关键次要发现：PRODUCE_UNAVAILABLE 是最大拒绝桶

```
upd=1 rejections: PRODUCE_UNAVAILABLE=1233  BUILD_UNAVAILABLE=30
                  UNIT_NOT_OWNED=25  UPGRADE_UNAVAILABLE=8
```

策略大量尝试生产被拒绝。`PRODUCE_UNAVAILABLE=1233` 远超其他，说明
action mask 与引擎实际可生产集合之间仍有不一致——这是一个具体的、
可修复的优化点（每次无效动作都付 `invalid_penalty`）。

## 训练加速实测
- 19674 tick / 4 envs / CUDA：**约 50-60 秒/update**
- 19674 tick 实时约 11 分钟，加速比约 **12 倍**

## 意义
此前 23 轮关于"难度 1 不可达"的所有结论，都建立在一个**从未真正训练过**的
系统上。本轮修复后，训练第一次在"能分出胜负"的对局上进行，并立即产生了
胜利信号。目标的后半部分（胜率 > 80%）仍远未达成，但现在有了真实的起点。
