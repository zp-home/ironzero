# 训练栈审计报告：未完成与半成品功能清单

> **项目**：IronZero（铁锈战争强化学习平台）  
> **模块**：`src/ironzero/training/`、相关训练/评估脚本、配置与测试  
> **审计时间**：2026-11-11  

---

## 1. 功能完成度矩阵

| 功能项 | 文档/架构期望 | 代码证据 (file:line) | 实际状态 | 严重度 | 优化建议 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **基础 PPO + GAE 训练管线** | 支持多离散动作头掩码、GAE 优势估计、早停、best 检查点保存 | `ppo.py:24-590` | **已闭环** | 无 | 保留现有实现，基线完整 |
| **经济系统奖励项 (Economy Rewards)** | 矿井、工厂、生产与空闲资金惩罚接入微操/宏观奖励 | `micro_config.py:112-117` | **死代码 / 仅 mock 离线回退** | 高 | 在真实游戏与训练环境中明确经济奖励接入，或标记为实验性字段移除 |
| **宏微分层训练 (Macro-Micro)** | 宏观建造/爆兵 + 微观 PPO 协同训练，打通真实游戏 RTS 全流程 | `vec_macro_micro.py:48-56`, `train_macro_micro.py:31` | **仅 Mock 闭环，未接真机** | 高 | 实现 RWPP 真实环境动作下发（需扩展 Java Bridge 支持建造/生产） |
| **真实游戏并行训练 (VecRwppEnv)** | 多进程并发启动 Headless 真实游戏，收集样本输入 PPO | `vec_rwpp.py:88-139`, `rwpp_ppo.py:22-58` | **已闭环 (Micro 战术)** | 低 | 增强崩溃自愈重试逻辑 |
| **课程学习 (Curriculum Learning)** | 从固定少量单位过渡到复杂地图、迷雾及动态对手 | `README.md:218`, 架构文档 | **未实现 (文档承诺)** | 中 | 新建 `curriculum.py` 管理关卡/兵力递进超参 |
| **自博弈与模型晋级 (Self-Play & Promotion)** | 历史模型池对抗、Elo/胜率评估、模型晋级守擂 | `README.md:229`, 命名方案 | **未实现 (路线图虚标)** | 中 | 路线图明确标记为待办，设计 ModelPool 对抗机制 |
| **内网人机评估模式 (`lan_eval`)** | 隔离内网环境对抗真人玩家或指定内网服务器 | 架构文档 | **仅文档，无代码** | 低 | 从架构宣称中移至远期规划，或提供专有 CLI |
| **行为克隆热启动 (BC Warmstart)** | 规则专家经验蒸馏至 PPO 策略网络 | `bc_warmstart_rwpp.py:68` | **半成品 (CLI 参数脱节)** | 中 | 修复 `--episodes` 标志未生效问题 |
| **超参数一致性 (Rollout / Minibatch)** | 训练步长与采样规模协同配置 | `train_rwpp.py:41`, `ppo.py:28`, `app.py:114` | **参数漂移** | 中 | 统一离线与真机默认值文档基准 |

---

## 2. 半成品与“有头没尾”功能深度分析

### 2.1 经济体系奖励字段为死代码
- **现状**：`micro_config.py` 中声明了 `extractor_reward`、`factory_reward`、`production_reward`、`idle_funds_penalty_coef`。
- **问题**：在 `MicroDuelConfig` 的主干引用中，只有 `mock_engine.py` 用 `getattr` 做了回退兜底，在真机环境 `rwpp_env.py` 和标准训练管线中**完全没有计算或加权逻辑**，用户配置后不产生任何训练效果。

### 2.2 宏微分层架构与真机环境断开
- **现状**：`vec_macro_micro.py` 与 `train_macro_micro.py` 完整实现了基于规则的 `MacroController` 与 PPO `MicroDuelEnv` 的配合。
- **断点**：环境硬编码绑定为 `MicroDuelEnv`（Mock 引擎）。真机 `RwppDuelEnv` 和底层的 Java Bridge 只支持 `move_group`, `attack_move`, `attack_target`, `hold`, `stop`，根本无法执行 `build_extractor`, `build_structure`, `produce_unit`。导致宏微分层只能在离线 Mock 跑通，无法应用于《铁锈战争》本体。

### 2.3 `bc_warmstart_rwpp.py` 参数解析但未生效
- **现状**：命令行参数定义了 `--episodes`（默认 24），但在执行收集循环 `collect()` 时，仅根据 `args.transitions` 截断，`args.episodes` 变量被丢弃，未参与任何循环终止或批次控制逻辑。

---

## 3. 优化与整理建议

1. **统一超参基线**：在 `configs/` 或公共模块中明确定义真机采样（由于游戏模拟开销大，rollout=8~20）与 Mock 采样（rollout=128）的预设模板。
2. **连接/标记宏微管线**：在文档与代码中清晰标注 `train_macro_micro.py` 为 Mock 实验特性；如需进军真机，首要任务是扩展 Java Bridge 指令集。
3. **修复参数与死代码**：清理 `bc_warmstart_rwpp.py` 的 `--episodes` 逻辑，移除或打通 `micro_config.py` 的经济奖励字段。
