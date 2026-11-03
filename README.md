# 铁锈战争强化学习

作者：**周培智**

一个面向 PC 版《铁锈战争》（Rusted Warfare）的强化学习研究项目规划，目标是把游戏接入为可观测、可动作、可重置、可复现的强化学习环境，并完成从训练到推理的完整闭环。

> 当前状态：架构设计阶段。仓库暂未包含可运行的桥接程序或训练代码。

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

许可证暂未指定。公开发布前请根据项目目标补充合适的开源许可证。

## 参考项目

- [Rusted Warfare LLM Pilot](https://github.com/Crystalhihihi/rusted-warfare-llm-pilot)
- [RWPP](https://github.com/RWPP-Team/RWPP)
- [RWPP Headless Server](https://github.com/RWPP-Team/RWPP/blob/main/rwpp-server/README.md)
- [Tape RTS mappings](https://github.com/TapeRTS/Tape)
- [Rusted Warfare 官方网站](https://rustedwarfare.com)
