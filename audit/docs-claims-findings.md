# 架构文档承诺与仓库卫生审计报告

> **项目**：IronZero（铁锈战争强化学习平台）  
> **模块**：架构设计文档、Docs、Schemas、CI/CD 与运维脚本  
> **审计时间**：2026-11-11  

---

## 1. 架构承诺 vs 代码实现对照表

| 架构承诺组件 | 出注文档 | 代码证据 / 现实状态 | 审计判定 | 严重度 | 说明与建议 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO + GAE 战术对抗** | 架构文档 11.2 | `ppo.py`, `train_rwpp.py` | **已实现** | 无 | 核心战术层运行良好 |
| **自博弈框架 (Self-Play)** | 架构文档 11.3 | 无对应代码模块 | **仅文档设计** | 高 | 属于 AlphaZero 核心命名承诺，建议建立最简模型池 (Model Pool) 机制 |
| **动态课程学习 (Curriculum)** | 架构文档 11.3, README | 仅硬编码地图和场景参数 | **未实现** | 中 | 建议编写关卡阶梯配置（如 8v8 → 16v16 → 25v25） |
| **内网人机对战模式 (`lan_eval`)**| 训练推理与演示架构:39 | 无专属独立入口 | **仅文档设计** | 低 | 建议通过 Web 平台的实机对局接口吸收替代 |
| **独立安全监控器 (Safety Supervisor)**| 训练推理与演示架构 | 融合在规则策略降级中 | **部分实现** | 低 | 规则策略层已承担安全兜底职责，文档定义偏重概念化 |
| **高性能传输 (ZeroMQ / Protobuf)**| 早期设计规划 | 采用 TCP Socket + NDJSON | **已废弃/变更方案** | 低 | 当前 NDJSON 简单可靠且满足延迟，建议更新文档结论 |

---

## 2. 协议与 Schema 漂移验证

- **Schema 生成机制**：`tools/generate_schema.py` 直接调用 `src/ironzero/protocol.py:message_schema()` 生成 `schemas/ironzero/1/messages.schema.json`。
- **验证结果**：运行 `tools/validate_schema.py` 显示 7 类标准报文全量通过校验，**不存在字段命名与类型的未同步漂移**。
- **协议语义冗余**：JSON Schema 与 Python 数据模型中保留了 `RepairTargetAction`、`ResourceSlot` 等并未在物理真机 Bridge 实现的定义。

---

## 3. CI/CD 与工程发布门禁缺口

- **现状**：`.github/workflows/` 下仅包含 `pages.yml`，用于静态文档发布。
- **风险**：代码仓库没有任何自动化的 CI 测试流水线（如运行 `pytest`、Schema 校验、代码格式检查）。
- **优化**：增加基础 Github Actions Workflow，执行现有通过率 100% 的 201 个单元测试与 schema 校验工具。

---

## 4. 仓库卫生与孤儿脚本清单

1. **已验证有效并保留的核心工具**：
   - `scripts/rwpp_real_service.py`：宿主服务守护神。
   - `scripts/play_rwpp_policy.py`：实机对战执行器。
   - `scripts/bc_warmstart_rwpp.py`：行为克隆热启动。
   - `scripts/probe_ironzero25.py` / `make_ironzero_map.py`：专属地图构建与探针。
2. **一次性探索/临时脚本**：
   - `scripts/download_jdk25.py`、`scripts/fetch_rwpp_sources.py`：为初始环境安装编写的辅助工具，通常在自动化构建容器外仅需单次使用。
3. **废弃临时文件清理**：
   - 之前会话分析留下的 `audit/_tmp_refs.py` 等分析辅助脚本已可归档或删除。
