<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 铁锈战争强化学习

作者：**周培智**

一个面向 PC 版《铁锈战争》（Rusted Warfare）的强化学习研究项目规划，目标是把游戏接入为可观测、可动作、可重置、可复现的强化学习环境，并完成从训练到推理的完整闭环。

> 当前状态：架构与协议基线阶段。已有 RWRL v1 协议 schema、协议单元测试和 RWPP 环境检查/冒烟脚本；尚未包含真实游戏桥接程序或训练代码。

> 开源许可：代码 Apache-2.0，文档 CC BY 4.0。详见 [许可证说明](./LICENSING.md)。

## 项目目标

- 读取游戏 tick、单位、资源、建筑、生产队列和战争迷雾状态
- 为强化学习模型提供结构化观测与合法动作空间
- 支持 PPO、多环境并行、课程学习和后续自博弈
- 支持 Java Agent 原版客户端与 RWPP Headless 两种接入方式
- 支持训练完成后的独立推理与安全动作校验
- 探索 LLM 负责战略、RL 负责战术的混合架构

## 推荐架构

```text
游戏层（原版客户端 / RWPP Headless）
        ↓
游戏内 Agent（状态快照与动作队列）
        ↓
Game Bridge（TCP / ZeroMQ / Protobuf）
        ↓
Gymnasium 环境（reset / step / reward）
        ↓
训练与推理系统（PPO / 模型导出 / 安全校验）
```

## 文档

完整技术方案见：

- [铁锈战争强化学习系统架构](./铁锈战争强化学习系统架构.md)
- [训练、推理与演示架构](./训练推理与演示架构.md)
- [文档索引](./docs/README.md)

配套规格与检查表：

- [环境规格说明](./docs/环境规格说明.md)
- [确定性与测试方案](./docs/确定性与测试方案.md)
- [安全与发布检查表](./docs/安全与发布检查表.md)
- [性能与容量基线](./docs/性能与容量基线.md)
- [协议 v1 说明](./docs/协议v1说明.md)
- [RWPP 兼容性验证](./docs/RWPP兼容性验证.md)
- [铁锈战争安装与获取](./docs/游戏安装与获取.md)

## 获取游戏

PC 版通过 Steam 官方渠道获取：

- [Steam 商店页面](https://store.steampowered.com/app/647960/Rusted_Warfare/)
- [安装说明](./docs/游戏安装与获取.md)

本仓库不包含游戏本体、`assets/` 或第三方安装包。

## 开发入口

运行协议测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

重新生成协议 schema：

```powershell
python tools/generate_schema.py
python tools/validate_schema.py
```

检查 RWPP 运行环境：

```powershell
.\scripts\check_environment.ps1 `
  -RuntimeDir "D:\rwpp-runtime" `
  -JavaHome "D:\jdks\jdk-25"
```

如果本机没有 Java 25，可先下载便携运行时：

```powershell
.\scripts\bootstrap_java25.ps1
```

执行 RWPP 无头冒烟测试：

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir "D:\rwpp-runtime" `
  -JavaHome "D:\jdks\jdk-25" `
  -Ticks 180
```

协议模型见 `src/rwrl/protocol.py`，机器可读 schema 见 `schemas/rwrl/1/messages.schema.json`。

文档覆盖：

- 游戏接入路线对比
- Java Agent 与 RWPP Headless 架构
- 状态空间、动作空间和奖励函数设计
- 桥接协议与游戏线程安全
- PPO、多环境、课程学习和评估方案
- 推理部署与 LLM + RL 混合架构
- 分阶段实施计划和风险控制

## 计划路线

- [ ] 只读探针：读取 tick、单位、资源、建筑和迷雾
- [ ] 动作桥：移动、攻击、停止、建造、生产
- [ ] Micro 场景：固定地图、小规模单位对战
- [ ] 经济场景：采矿、建造、生产和资源点
- [ ] 完整 1v1：多地图、战争迷雾和内置 AI 基线
- [ ] 自博弈与模型评估
- [ ] LLM + RL 混合指挥

## 使用边界

本项目仅允许在单机或与公网隔离的受控内网环境中运行。禁止接入官方服务器、公共匹配和公开房间；禁止与未经同意的真人玩家进行多人对战。内网研究房间必须由项目所有者完全控制，所有参与者知情同意，并且不得暴露到公网。

第三方项目、游戏资源和反编译相关内容的使用必须遵守各自许可证与版权要求。

## 许可证

本项目采用分层开源许可证：

- 代码、脚本、测试和 schema：**Apache-2.0**
- 文档：**CC BY 4.0**
- 第三方依赖和参考项目：各自保留原许可证

详见：

- [许可证说明](./LICENSING.md)
- [Apache-2.0 全文](./LICENSE)
- [CC BY 4.0 全文](./docs/LICENSE)
- [第三方声明](./THIRD_PARTY_NOTICES.md)
- [NOTICE](./NOTICE)
- [安全政策](./SECURITY.md)

## 参考项目

- [Rusted Warfare LLM Pilot](https://github.com/Crystalhihihi/rusted-warfare-llm-pilot)
- [RWPP](https://github.com/RWPP-Team/RWPP)
- [RWPP Headless Server](https://github.com/RWPP-Team/RWPP/blob/main/rwpp-server/README.md)
- [Tape RTS mappings](https://github.com/TapeRTS/Tape)
- [Rusted Warfare 官方网站](https://rustedwarfare.com)
