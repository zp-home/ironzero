# Round 23：Web 界面训练链路打通（目标要求的第一步已达成）

## 目标的两部分
> "通过 Web 界面在原版地图上训练，AI 难度 1，训练加速，2p 场景胜率超过 80%"

本轮完成了前半部分：**Web 界面 + 原版地图 + 难度 1 + 训练加速** 全部验证通过。

## 修复 1：Web API 无法设置 AI 难度（已修复）
`web/backend/app/models.py` 的 `RwppPpoRequest` 缺少 `enemy_ai_difficulty`、
`ai_difficulty`、`win_mode`、`unlock_ai`、`seed` 五个字段。宿主机服务
（`scripts/rwpp_service/ppo.py`）其实支持它们，但浏览器无法发送，所以每个从
UI 启动的训练都跑在引擎默认难度上。

## 修复 2：种子在服务层被丢弃（已修复）
`run_ppo()` 声明了 `seed` 参数，但 `launch.py` 从未从 payload 读取它，
所以请求里的种子被静默丢弃。已补上提取与 `--seed` 转发。

## 端到端验证（真实游戏，6 秒）
`scripts/probe_web_e2e.py` 用 Web UI 会发的完整 payload 调用服务入口：
```
arguments=[--no-console, --bridge-port, 5140, --ticks, 1200,
           --map, maps/skirmish/[p2]Lake (2p).tmx, --frame-ms, 0,
           --win-mode, objective, --unlock-ai,
           --ai-difficulty, 0, --enemy-ai-difficulty, 1,
           --physics-ms, 16, --seed, 71000]
world ready: mapLoaded=true units=24 teams=2 winMode=objective
             unlockAi=true aiDifficulty=team1=level1/x1.400
```
**难度 1（收入 ×1.400）和种子都真正到达引擎。**

## 修复 3：rollout_steps 与 episode_ticks 不匹配导致零学习（已修复）
首次训练用 `rollout_steps=30` + `episode_ticks=3000`，metrics 显示 6 个 update
里 5 个 `episodes=0`——因为 `DECISION_INTERVAL=10`，30 步只有 300 tick，
episode 永远跑不完，PPO 从未见过终止奖励，等于在 300-tick 碎片上采样。

改为 `rollout_steps=130` + `episode_ticks=1200`（120 决策 = 完整 episode）后：
```
episodes: 2 → 10
```

## 训练加速实测
| 配置 | updates | 墙钟 | 每 update |
|---|---|---|---|
| 2 envs × 130 steps, 1200 ticks | 8 | **30 s** | 3.8 s |
| 2 envs × 30 steps, 3000 ticks | 6 | **20 s** | 3.3 s |

真实游戏、难度 1、`frame_ms=0` + lockstep 下约 **3-4 秒/update**，
比实时快约 8 倍以上（1200 tick 实时需 ~40 秒）。

## 当前胜率
- 难度 1 / Lake / 8 updates / 10 episodes：**win_rate = 0.0**（95% CI 0-56%）
- 训练刚起步（10 个 episode），不足以判断学习曲线走向

## 结论
目标的前半部分（Web 界面训练、原版地图、难度 1、训练加速）**已验证达成**，
并有端到端日志证据。后半部分（难度 1 胜率 > 80%）仍未达成：22 轮的手写
curriculum 实验和本轮的 PPO 训练都停在 0-12.5%。

## 可复现配置
```
Web API: POST /api/rwpp/ppo
{
  "updates": 8, "envs": 2, "rollout_steps": 130, "episode_ticks": 1200,
  "map_name": "Lake", "map_path": "maps/skirmish/[p2]Lake (2p).tmx",
  "map_width": 2600, "map_height": 2600, "frame_ms": 0,
  "enemy_ai_difficulty": 1, "win_mode": "objective", "unlock_ai": true,
  "seed": 71000, "eval_every": 2, "eval_episodes": 3
}
```
