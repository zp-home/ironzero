# 推理与 Web 平台审计报告：契约差异与接口半成品

> **项目**：IronZero（铁锈战争强化学习平台）  
> **模块**：`scripts/play_rwpp_policy.py`, `scripts/rwpp_real_service.py`, `web/backend/`, `web/frontend/`, `src/ironzero/strategy/`  
> **审计时间**：2026-11-11  

---

## 1. 功能完成度矩阵

| 功能项 | 期望规格 | 代码证据 (file:line) | 实际状态 | 严重度 | 优化建议 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO 模型实机推理 (`play_rwpp_policy.py`)** | 驱动真机对局、产出 Replay 与沙盘帧流 | `play_rwpp_policy.py:182-255` | **已闭环** | 低 | 增强动态血量基准对齐 |
| **实机服务托管 (`rwpp_real_service.py`)** | 单任务并发锁、作业管理、日志采集与状态监控 | `rwpp_real_service.py:864-880` | **已闭环** | 低 | 结构良好，已成为真机运行事实标准 |
| **FastAPI 后端 Web 平台 (`app.py`)** | 提供训练监控、回放回放、地图检索、真机作业交互 | `app.py:764-850` | **已闭环** | 低 | 保持代理路由稳定 |
| **LLM 战略层 (`llm_strategy.py`)** | DeepSeek/OpenAI 兼容接口输出战略指令 + 规则兜底降级 | `llm_strategy.py`, `train_rwpp.py:161` | **半成品 (训练已接，推理未接)** | 中 | 在 `play_rwpp_policy.py` 和 Web 推理接口中补齐 `--use-llm` 链路 |
| **前端 Demo / Sandbox 模式** | 无后端状态下的离线纯前端演示体验 | `app.js:297KB` | **已闭环** | 低 | 前端内嵌离线 Mock 数据容灾完整 |
| **协议操作 `RepairTargetAction`** | 建筑/单位维修动作下发 | `protocol.py:143`, Java Bridge | **死代码 / 未实现** | 中 | 从协议中废弃或在 Bridge 端补齐指令分发 |
| **平台参数与路径硬编码** | 跨平台路径兼容与配置自适应 | `rwpp_env.py:40-41`, `app.py` | **轻度硬编码** | 中 | 保持环境变量覆盖，但在跨环境部署手册中重点注明 |

---

## 2. 训练与推理观测契约差异清单

在模型从训练走向实机推理或沙盘回放时，存在若干关键观测/环境契约口径漂移风险：

1. **战争迷雾半径 (`vision_radius`)**：
   - `train_rwpp.py` 默认设为 `260.0`（启用局部战争迷雾）。
   - `RwppRecordRequest` (Web端数据录制) 默认值设为了 `0.0`（关闭迷雾/全图可见）。
   - **影响**：如果使用录制配置启动策略对战，可能由于观测中敌方单位全量暴露，导致训练在有限视距下的 Policy 发生特征分布偏移。
2. **血量归一化分母 (`hp_fractions`)**：
   - 训练期间若单位死亡，其引用基准在早期版本未固定起始最大生命值。`play_rwpp_policy.py` 通过 `own_max_ref` 在 `reset()` 时锁定了初始血量总量，修复了部队减员后血量百分比依然为 1.0 的 bug。需确保各处评估脚本口径严格统一。

---

## 3. Web 平台与接口断点分析

1. **LLM 战略层在 Web 推理入口缺失**：
   - Web 后端 `RwppPpoRequest` 支持 `use_llm` 字段，并在训练作业中正确透传给宿主服务。
   - 但在普通的实机推理/对局回放请求 `RwppRecordRequest` 中，并未提供 `use_llm` 配置项，导致用户在 Web 页面“单局推理测试”时无法开启 LLM 指挥官进行对战。
2. **Java Bridge 动作能力不对称**：
   - Python 侧已定义 `build_structure`、`produce_unit`、`repair_target` 等动作。
   - Java 侧桥接插件（NDJSON）仅拦截并处理了移动、攻击、原地保持三类指令，上层扩展动作下发会触发未识别指令或直接被丢弃。

---

## 4. 优化与整理建议

1. **统一推理参数与训练参数**：将 `RwppRecordRequest` 的 `vision_radius` 默认值与 `train_rwpp.py` 的 260.0 对齐。
2. **Web 平台补全战略指挥开关**：在 Web 平台的“对局实测/推理”界面增加“启用 LLM 战略官”复选框，并在后端透传至推理脚本。
3. **协议清理与剪枝**：清理未在 Java Bridge 落地的 Action 类型，避免给后续开发者造成“已支持建造”的误导。
