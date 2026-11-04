<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# RWPP 角色与适配决策

> 文档状态：架构决策记录
> 作者：周培智
> 日期：2026-11-04
> 关联文档：[RWPP 兼容性验证](./RWPP兼容性验证.md)

## 1. 决策结论

本项目应当：

1. 把 RWPP 作为《铁锈战争》的无头运行宿主和游戏 API。
2. 深度适配 RWPP 的扩展 API、tick 事件、游戏线程调度和 AI Controller。
3. 自己实现 RWRL Bridge、Gymnasium 环境、动作安全层、训练和评测。
4. 对上游版本做固定，不跟随 `main` 自动滚动。
5. 维护少量、可审计、可提交上游的兼容补丁。
6. 不深度分叉 RWPP 客户端、画面系统或游戏引擎。

RWPP 是基础设施，不是强化学习框架。真正属于本项目的是“环境层”和“训练闭环”。

---

## 2. RWPP 在系统中的角色

```text
┌────────────────────────────────────────────┐
│ RWRL Trainer / Evaluator                   │
│ PPO、课程、模型、指标                       │
└──────────────────┬─────────────────────────┘
                   │ Gymnasium
┌──────────────────▼─────────────────────────┐
│ RWRL Environment                           │
│ reset / step / reward / masks              │
└──────────────────┬─────────────────────────┘
                   │ RWRL Bridge / NDJSON
┌──────────────────▼─────────────────────────┐
│ RWRL RWPP Extension                        │
│ observation、action queue、receipt、safety  │
└──────────────────┬─────────────────────────┘
                   │ RWPP API + RWML
┌──────────────────▼─────────────────────────┐
│ RWPP Server                                │
│ headless engine、tick、network、extensions │
└──────────────────┬─────────────────────────┘
                   │ game classes + assets
┌──────────────────▼─────────────────────────┐
│ Rusted Warfare 1.15                        │
└────────────────────────────────────────────┘
```

### 2.1 RWPP Server

负责：

- 启动 JVM 无头游戏进程。
- 加载原版 `assets/`。
- 创建和管理游戏引擎。
- 提供扩展生命周期。
- 提供自定义 `ServerExtension` 运行循环。
- 管理 TCP/KCP 网络和房间。
- 通过 `--ticks` 控制运行帧数。

不负责：

- 强化学习观测编码。
- Gymnasium 接口。
- 奖励函数。
- PPO 训练。
- 动作掩码语义。
- episode 生命周期和模型管理。

### 2.2 RWPP API

负责：

- 游戏、房间、地图、单位、队伍和 AI 访问。
- 原生生命周期与模拟事件。
- `SimulationBeforeTickEvent` / `SimulationAfterTickEvent`。
- `FrameBeginEvent` / `FrameEndEvent`。
- `GameMainDispatcher` 游戏线程调度。
- `AiManager` 和自定义 AI Controller。
- 命令执行与扩展事件。

这是 RWRL 最值得深入适配的层。

### 2.3 RWML

负责：

- 扩展 JAR 的加载和依赖排序。
- Mixin/ASM 类变换。
- 游戏资源与扩展生命周期。
- 服务端扩展宿主环境。

它不负责游戏状态建模、RL 协议或训练。

### 2.4 RWRL 自己负责

- `rwrl/1` Bridge 协议。
- 观测 schema 和可见性过滤。
- 动作 schema、动作掩码和执行前校验。
- `reset/step/reward/terminated/truncated`。
- episode 编排、超时、故障隔离和回放。
- Python 训练、评估、推理和安全层。

---

## 3. 适配深度选择

### 方案 A：只把 RWPP 当黑盒服务端

特点：

- 通过外部网络连接。
- 不在进程内实现 RL 扩展。
- 改动最少。

问题：

- 难以实现确定性 lockstep。
- 难以直接控制 tick。
- 动作时序和状态同步复杂。
- 不适合多实例训练。

结论：不够。

### 方案 B：薄补丁 + RWRL 扩展

特点：

- 固定 RWPP/RWML/game-lib 版本。
- 只维护少量 API 兼容补丁。
- 主要逻辑在 RWRL Extension。
- 通过官方扩展点接入 tick 和游戏线程。

优点：

- 控制力和维护成本平衡最好。
- 上游升级后可重新评估补丁。
- 容易做自动化构建和回归。
- RL 逻辑与上游代码边界清晰。

结论：推荐。

### 方案 C：完整分叉 RWPP

特点：

- 直接复制客户端/服务端并大改。
- 自己维护 UI、网络、游戏 API 和构建系统。

问题：

- 上游更新很难合并。
- AGPL 和第三方许可证义务更复杂。
- 维护成本远超 RL 本身。
- 容易把游戏 API 适配和训练问题混在一起。

结论：除非上游完全不接受必要修复，否则不推荐。

---

## 4. 已验证的基础设施

当前已经完成：

- RWPP 无头服务端实际构建。
- RWML Desktop 从源码本地构建。
- 1.15 客户端 `assets/` 加载。
- RWPP API/Server 扩展加载。
- 默认地图加载。
- 180 tick 真实运行。
- 退出码 0。
- 端口释放。

已验证版本见 [RWPP 兼容性验证](./RWPP兼容性验证.md)。

---

## 5. 强化学习训练前必须补齐的基础设施

### 5.1 可重复构建

必须提供：

```text
锁定版本清单
依赖 SHA-256
一键构建脚本
构建产物 manifest
```

不能依赖人工记忆和本机临时修改。

### 5.2 兼容补丁

当前最小补丁：

```text
game-lib v2-1.15 -> v3-1.15
LocaleEngine.b -> LocaleEngine.translateMapName
```

未来所有补丁必须：

- 独立成 patch 文件。
- 标明目标 commit。
- 附构建和冒烟测试结果。
- 升级后重新验证。

### 5.3 RL 扩展骨架

扩展至少需要：

```text
ServerExtension.runServer
  -> HeadlessEngine.create
  -> pump()
  -> 固定 tick 循环
  -> SimulationBeforeTick: 执行动作
  -> SimulationAfterTick: 采集快照
  -> 发送 Observation
  -> 处理 reset/step/shutdown
```

### 5.4 确定性与 lockstep

必须实测：

- 是否能暂停在决策 tick。
- 是否能稳定推进固定 tick 数。
- 相同操作序列是否可复现。
- 游戏 RNG 是否能固定种子。
- 单位 ID 是否在同一 episode 内稳定。

### 5.5 状态快照

第一版只读 Bridge 至少输出：

```text
tick
controlled_team
funds
income
units
visible_enemies
local map
visibility
action masks
```

所有游戏对象读取必须发生在游戏线程。

### 5.6 动作执行与回执

必须验证：

- 命令进入游戏线程队列。
- 非法动作被拒绝。
- 重复动作被去重。
- 过期动作被拦截。
- 执行结果能在后续 tick 观察到。

### 5.7 进程和 episode 生命周期

首版建议：

```text
一个 episode
  -> 一个 RWPP 进程
  -> 一个 Bridge
  -> 一个端口
  -> 一个运行目录
  -> 结束后完整清理
```

### 5.8 可观测性

至少记录：

- tick。
- 动作延迟。
- Bridge RTT。
- 状态大小。
- 扩展加载耗时。
- 地图加载耗时。
- 退出原因。
- 进程和端口清理结果。

---

## 6. 是否值得深入适配

值得深入适配以下部分：

- `ServerExtension` 自定义运行循环。
- `SimulationBeforeTickEvent` 和 `SimulationAfterTickEvent`。
- `GameMainDispatcher` 线程安全。
- `AiManager` 的 `AUGMENT_NATIVE` / `REPLACE_NATIVE`。
- 地图、单位、队伍和迷雾 API。
- 命令回执和错误传播。
- 进程启动和关闭。

不值得一开始深入修改：

- 客户端 UI。
- 渲染管线。
- 音频。
- 网络房间协议。
- Mod 分发。
- 游戏战斗数值。
- 游戏单位平衡。

判断标准：

> 如果一项改动属于“让游戏更容易被观察和控制”，应做；如果属于“改变游戏规则”，通常不应做。

---

## 7. 上游 PR 策略

### 7.1 可提交的上游修复

PR 1：映射版本兼容

```text
game-lib: v2-1.15 -> v3-1.15
LocaleEngine.b -> LocaleEngine.translateMapName
```

PR 2：构建和发布

- 为 `rwml-desktop:1.0.2` 提供匿名可获取的构件。
- 或者提供明确的源码构建和发布说明。

PR 3：M1 集成测试

- 使用 1.15 测试资源。
- 不提交游戏 assets。
- 启动服务端。
- 推进 180 tick。
- 验证端口释放。

### 7.2 PR 前检查

- 先从 `main` 创建独立分支。
- 一个 PR 只解决一个主题。
- 不提交本机路径、代理配置和游戏资源。
- 不混入 RWRL 业务代码。
- 附上构建日志和冒烟结果。
- 说明对旧映射版本的影响。

### 7.3 当前上游状态

- RWPP 公开仓库，默认分支 `main`。
- 当前没有独立的 `CONTRIBUTING.md`。
- 公开搜索结果中未发现对应的 `game-lib v3` 兼容 PR。
- RWPP Server 是 2026-10 新加入的能力，仍处于快速演进阶段。

因此建议先开 Issue 说明兼容矩阵和复现步骤，再提交小范围 PR。

---

## 8. 许可证边界

### 8.1 RWPP

RWPP 使用 AGPL-3.0。

如果未来公开分发修改后的 RWPP 或与其紧密链接的衍生代码，需要认真履行 AGPL 义务。不要把 RWPP 源码直接复制进本项目 Apache-2.0 目录。

### 8.2 RWML

RWML 1.0.2 使用 RWPP-Team Shield License，不是 OSI 开源许可证。其附加条款允许插件、扩展、适配器和配合使用的工具，但限制竞争性衍生产品的公开发布。

因此：

- 仅把 RWML 作为构建/运行依赖时，风险相对可控。
- 不要公开发布修改后的 RWML 竞争性版本。
- 分发前重新阅读具体 LICENSE。
- 本项目自己的代码许可证不能覆盖 RWML。

### 8.3 建议的目录边界

```text
src/rwrl/                    Apache-2.0
schemas/                     Apache-2.0
docs/                        CC BY 4.0
game-agent/rwpp-extension/   单独许可证，按 RWPP 要求
vendor/rwpp/patches/         单独记录许可证和来源
```

对外发布集成包前必须做许可证审查。

---

## 9. 推荐的基础设施路线

```text
M1.1  固定版本清单和构建脚本
M1.2  自动化创建 runtime 和复制 assets
M1.3  长时运行和端口/进程清理测试
M1.4  RWRL Extension 只读探针
M2    状态快照和 episode 生命周期
M3    动作桥和动作回执
M4    Gymnasium + PPO
```

只有在 M1.1—M1.3 稳定后，才应该开始训练。

---

## 10. 最终决策

**采用 RWPP 作为无头游戏宿主，深入适配其 API 和扩展点；维护最小兼容补丁；不深度分叉游戏客户端。**

一句话概括：

> RWPP 负责“让游戏可运行、可扩展、可 tick 控制”，RWRL 负责“把游戏变成可训练、可复现、可安全推理的强化学习环境”。
