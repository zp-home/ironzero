# Round 16 发现：工厂配额不是瓶颈（纠正 Round 15 结论）

## Round 15 的错误结论
Round 15 的文档称“敌方 4-6 座工厂，我方仅 2 座”，并把它列为失败障碍。本轮
证明这个结论是错误的。

## 实测：我方实际建成 3 座 landFactory

`probe_factory_quota.py` 扫描 economy_max_factories = 2/4/6：
```
quota=2: peak_factories=[2,2,2,2] produced=[29,29,29,29]
quota=4: peak_factories=[2,2,2,2] produced=[25,25,25,25]
quota=6: peak_factories=[2,2,2,2] produced=[25,25,25,25]
```
最初误以为卡在 2 座。但直接查实体坐标（`probe` 统计所有 building）显示：
```
commandCenter id=23 (990,2010)
landFactory   id=27 (1090,2070)
landFactory   id=30 (890,2150)
landFactory   id=31 (750,2210)     <- 第三座确实建成了
factory_exact=(890,2150)
```
此前统计脚本只算 `is_operational` 的工厂，采样时刻第三座尚未完工，所以
误报为 2。**工厂产能已饱和（3 座），dead_sites=0，配额不是瓶颈。**

## 真正的结论保持不变

Round 15 的核心发现仍然成立且更重要：
- 攻击敌方工厂使兵力峰值从 13-19 翻倍到 24-30
- 集中攻击最弱工厂在难度 1 取得 1/8 胜利（seed 71003，tick 4717）
- 军队被拦截在距离工厂 411-1167px 处

**修正后的障碍清单**：
1. 军队在 y≈314 处被敌方主力拦截，无法推进到 y≈210 的工厂
2. 敌方会重建被摧毁的工厂（我方摧毁一座时敌方工厂数反增）
3. 不是产能问题——我方 3 座工厂、29+ 坦克产能已足够

下一轮方向：让军队绕过敌方主力（例如沿湖两栖突进工厂），而不是正面交战。
