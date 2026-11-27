## Round 30 完成：Web 前端现已发送 difficulty 1 / win_mode / seed

### 终于修好了

此前 Web 界面根本不发送 `enemy_ai_difficulty`、`win_mode`、`seed` 这三个字段，导致从浏览器启动的任何训练都跑在引擎默认难度（level 0, x1.00 income）之下，明明目标要求“AI 难度 1”却无法 express。

**三处改动**：

1. **`web/frontend/index.html`** —— 新增三个字段：
   - 敌方 AI 难度下拉框（0正常/1困难/2非常困难/3炼狱/-1简单/-2非常简单）
   - 胜利判定 win_mode 选择框（objective/annihilation/native）
   - 种子 seed 数字输入框（默认 71000）

2. **`web/frontend/js/state.js`** —— 将这三个 ID 加入 `els` 缓存表，供后续 JS 读取

3. **`web/frontend/js/ppoRun.js`** —— payload 现在真正包含：
   - `enemy_ai_difficulty: 1`（困难，收入 ×1.40）
   - `win_mode: "objective"`（摧毁敌方产能即胜）
   - `unlock_ai: true`（解锁内置 AI）
   - `seed: 71000`（ pinned seed）

**契约已在三个层面上闭环**：

| 层级 | 验证结果 |
|---|---|
| 前端源码 (`ppoRun.js`) | ✅ 发送了这四个字段 |
| 后端模型 (`models.py`) | ✅ Pydantic schema 接受这些字段 |
| 服务层 (`launch.py`) | ✅ 解析并转发给 trainer 命令行 |
| 训练命令行 | ✅ `--enemy-ai-difficulty 1 --win-mode objective --unlock-ai --seed 71000` |

### 目标进度 `goal-e05e7888-5612-4856-aa40-d3201b8c5df9`

- **Round**: 30/40 (还剩 10 轮)
- **前半部分已达成**: "通过 Web 界面在原版地图上训练，AI 难度 1，训练加速"
  - Web UI 可以设置 difficulty 1 并真正发送到引擎
  - 训练加速已验证 (~3-4 s/update with lockstep)
- **后半部分仍为挑战**: "2p 场景胜率超过 80%"
  - 即使有 extractor T2 (income 30 对比 26) 与完整的 Web 设置，0/4 胜率仍由空中威胁与主导条件决定
  - 现已明确：是**策略学习问题**而非系统 Bug

### 本轮实际改动

| 文件 | 变更 |
|---|---|
| `web/frontend/index.html` | 新增 difficulty / win_mode / seed 表单 |
| `web/frontend/js/state.js` | 新增三个字段到 els 缓存 |
| `web/frontend/js/ppoRun.js` | payload 真正包含这四个字段 |
| `scripts/probe_web_payload.py` | 三端契约（前端/模型/服务）全部验通过 |
| `docs/Round30前端难度参数就绪.md` | 新增文档记录 |

### 状态概览

| 目标要素 | 状态 |
|---|---|
| Web 界面在原版地图上训练 | ✅ 已完成（可设 difficulty 1） |
| AI 难度 1 | ✅ 已完成（前端可设，后端可接受） |
| 训练加速 | ✅ 已验证 (~3-4 s/update) |
| 2p 场景胜率超过 80% | ❌ 0/4（策略学习问题，系统已就绪） |
| 可复现的训练配置 + 评测证据 | ✅ 已提供 |

### 关键路径总结 (Round 10-30)

| 轮号 | 贡献 |
|---|---|
| Round 10-22 | 累计 16 个系统 Bug (种子、菜单、资金、工厂停滞等) |
| Round 23 | Web 训练链路打通，API字段、种子、rollout_steps 修复 |
| Round 24 | 训练长度根因：episode_ticks=19674 恢复学习信号 |
| Round 25-27 | **Extractor tier-up 突破**：CustomUnit 修复，4/4 T2，income 54.0 |
| Round 28 | 系统就位：Extractor升级可用，Web训练入口就绪 |
| Round 29 | 学习问题焦点：系统已就绪，剩为策略学习 |
| Round 30 | **Web 前端补全**：difficulty 1 / win_mode / seed 可从浏览器设置 |

### 最终判断

- **前半部分 ("Web 界面在原版地图上训练，AI 难度 1，训练加速")** ✅ 已完成
  - 前端现在可以从 UI 设置 difficulty 1，后端真正接收并使用
  - 训练加速 (~3-4 s/update) 已验证
- **后半部分 ("2p 场景胜率超过 80%")** ❌ 仍为策略学习挑战
  - 即使有 extractor T2 (income 30) 与完整 Web 设置，0/4 胜率仍由空中威胁与主导条件决定
  - 10 轮剩余，用于策略和配置调参（produce hoverTanks, counter air, dominance）

**总结**: Round 30 完成了 Web 前端对 difficulty 1 / win_mode / seed 的补全，这是目标的**必要条件**。系统基础（extractor upgrade）已固化，政策学习挑战依然存在但已在明确的技术路线上（produce hoverTanks, counter air units）。目标的前半部分已正式完成，后半部分需要在固定系统上继续策略学习。