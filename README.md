<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# IronZero（铁零）

> 让 AI 在一张还在打仗的棋盘上，学会即时战略。

作者：**周培智**

**IronZero = 《铁锈战争》的 AlphaZero。** 一个面向 PC 版《铁锈战争》（Rusted Warfare）的强化学习研究项目，目标是把游戏接入为可观测、可动作、可重置、可复现的强化学习环境，并完成从训练到推理的完整闭环。

与其他 RL 框架的区别：这里真正稀缺的不是训练算法，而是**一个还在更新、有战争迷雾、按 tick 推进的商业 RTS 环境**——以及在这块活棋盘上可复现的轨迹。

> 当前状态：M1 无头环境验证已通过，本机已确认同一 1.15 Build #28 游戏资源（21.4 MiB assets、84 张地图）。
>
> 平台进度：`ironzero/1` Bridge（NDJSON/TCP）、Gymnasium 兼容环境、动作掩码、确定性微操参考环境、PPO + GAE 训练与评估闭环**已实现并通过 203 项测试**；参考环境按真实 1.15 坦克数值校准（HP 210 / 伤害 25 / 射程 130 / 75 tick 射速 / 66 单位每秒），200 轮 PPO 对脚本 AI 的评估胜率 0% → 100%（随机基线 0%，平均回报 −1.33 → +1.21）。训练/推理 Web 平台（Docker + 监控大屏 + 对局沙盘 + 回放/录制）已可运行，沙盘支持真实地图地形模拟（A* 绕水/碰撞）、迷雾与道具层。
>
> 真机进度：已从零重建 RWPP 1.15 无头运行时（180 tick 冒烟通过）；IronZero Java 扩展把真实游戏世界状态（tick/单位/队伍，10 tick 一帧）经回环 NDJSON 输出，并**支持动作注入**（move / attack_move / attack_target 经 `CommandController` 执行，回执 accepted）；`scripts/record_rwpp_replay.py` 可驱动真机单位并把回放落进平台，沙盘按真实 Lake 地图播放。**平台一键真机模式（容器内 Java + runtime）待接**，详见 [平台搭建方案 5.4](./docs/平台搭建方案.md)。

> 开源许可：代码 Apache-2.0，文档 CC BY 4.0。详见 [许可证说明](./LICENSING.md)。

## 项目目标

- 读取游戏 tick、单位、资源、建筑、生产队列和战争迷雾状态
- 为强化学习模型提供结构化观测与合法动作空间
- 支持 PPO、多环境并行；课程学习与自博弈尚未实现（见下方路线图）
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

## 目录结构

代码按「一层一个职责」组织，单文件默认 ≤300 行。旧 import 路径全部保留为薄 shim，升级不会破坏外部引用。

```text
src/ironzero/
  protocol.py          # ironzero/1 消息模型（Pydantic）+ schema 校验
  micro_config.py      # 观测/动作契约的单一真相源
  bridge.py            # NDJSON/TCP 桥接客户端与服务端
  tcp_adapter.py       # 桥接 → 游戏适配层
  mock_server.py       # 离线 mock 游戏服务入口
  env.py               # Gymnasium 兼容环境（离线）
  assets.py            # 游戏资源数值解析（unit.ini / .tmx）
  replay.py            # ironzero.replay/1 帧格式

  engine/              # 确定性参考引擎（原 mock_engine.py，mixin 组合）
    constants.py         # 动作种类、实体类型 ID、调参、错误码
    entities.py          # Unit / MockResourceSlot
    config.py            # make_terrain_config
    spawn.py             # 出生点与地形布阵
    actions.py           # 动作校验与执行
    movement.py          # 移动、A* 路径跟随、卡住恢复
    combat.py            # 选目标、视野、伤害、胜负判定
    economy.py           # 采矿、建造、生产队列
    observation.py       # 引擎侧原始观测载荷
    reward.py            # 潜势与奖励窗口
    opponents.py         # 脚本 AI 与内置对手
    engine.py            # MicroDuelEngine 编排（step/advance/evaluate）
    adapter.py           # MicroDuelAdapter

  observation/         # 观测编码（原 observations.py）
    encoding.py          # encode_observation
    terrain_channel.py   # 地形可通行通道与降采样
    features.py          # 实体特征、分组、栅格投影
    masks.py             # 动作掩码与扁平化

  terrain/             # 地形与寻路（原 terrain.py）
    grid.py              # TerrainGrid：加载、坐标变换、阻挡判定
    passable.py          # 最近可通行点搜索
    pathing.py           # A* 寻路与路径压缩
    utils.py             # 数值钳制工具

  rwpp/                # 真机环境（原 rwpp_env.py）
    config.py            # RwppGameConfig 与默认常量
    process.py           # 无头进程启动与帧预算
    transport.py         # 行协议读取
    decoding.py          # 桥接观测 → 状态、动作编码
    vision.py            # 观测级战争迷雾
    reward.py            # HP 潜势、血量比例、军队距离
    spaces.py            # Box / MultiDiscrete space
    env.py               # RwppDuelEnv：reset/step/终止结算

  strategy/            # 战略层（LLM + 规则降级）
  training/            # PPO、向量化环境、训练与评估入口

web/
  backend/app/         # FastAPI 后端（routers 按资源分组）
  frontend/js/         # 监控大屏（ES Modules）
scripts/rwpp_service/  # 宿主机真机任务托管服务
```

模块边界、拆分规则与验证方式见 [目录结构与模块边界](./docs/目录结构与模块边界.md)。

## 文档

完整技术方案见：

- [铁锈战争强化学习系统架构](./铁锈战争强化学习系统架构.md)
- [训练、推理与演示架构](./训练推理与演示架构.md)
- [平台搭建方案](./docs/平台搭建方案.md)
- [文档索引](./docs/README.md)

配套规格与检查表：

- [环境规格说明](./docs/环境规格说明.md)
- [确定性与测试方案](./docs/确定性与测试方案.md)
- [安全与发布检查表](./docs/安全与发布检查表.md)
- [性能与容量基线](./docs/性能与容量基线.md)
- [协议 v1 说明](./docs/协议v1说明.md)
- [RWPP 兼容性验证](./docs/RWPP兼容性验证.md)
- [铁锈战争安装与获取](./docs/游戏安装与获取.md)
- [经典强化学习案例与 RTS 启示](./docs/经典强化学习案例与RTS启示.md)
- [RWPP 角色与适配决策](./docs/RWPP角色与适配决策.md)

## 获取游戏

PC 版通过 Steam 官方渠道获取：

- [Steam 商店页面](https://store.steampowered.com/app/647960/Rusted_Warfare/)
- [安装说明](./docs/游戏安装与获取.md)

本仓库不包含游戏本体、`assets/` 或第三方安装包。

## 展示页

- [在线展示页](https://zp-home.github.io/ironzero/)
- [IronZero Lab 静态展示页](./site/index.html)
- 页面源码与本地预览说明见 [site/README.md](./site/README.md)

本地预览：

```powershell
python -m http.server 8000 --directory site
```

## 训练与推理平台（实机）

一条命令启动平台（宿主机另需跑真机服务）：

```powershell
python scripts\rwpp_real_service.py --port 8300 --max-parallel 4   # 宿主机：真实 1.15 运行时 + IronZero 扩展
docker compose -f web/docker-compose.yml up -d --build
start http://127.0.0.1:8000
```

平台只保留两个主页面，都面向**真实《铁锈战争》1.15 无头对局**，且都带监控大屏（KPI + 曲线）：

- **实机推理**：选地图（71 张）+ 策略（scripted）+ 时长 + **并发实例 1..4** → 宿主机同时拉起 N 个独立真实对局，帧实时回传；大屏显示并发任务卡、实时帧率、双方血量/存活单位曲线，大沙盘按真实地图渲染、自动跟随、可录制 `.webm`
- **实机训练**：选地图/局数/每局 ticks + **并发线程 1..4** → 并行跑多个真实游戏进程逐局采集，大屏显示进度 KPI、每局耗时柱状图、累计帧/tick 曲线 + 产物回放（播放跳回实机推理）
- **真机回放**：每局落盘 `ironzero.replay/1`，回放库标注「真机」，点击即可按真实 Lake 地图播放

> 并发说明：每个实例是一个独立无头游戏进程（独立 bridge 端口 5130+），游戏模拟本身单线程；并发提升的是批量采集吞吐，单局时长不变。实测 2 路并发推理 / 训练 2 局 `concurrency=2` 均无冲突（`python scripts/rwpp_concurrency_check.py` 可复验）。

![实机推理](docs/assets/platform_real_infer.png)

![实机训练](docs/assets/platform_real_train.png)

实机地图渲染需要先把本机 1.15 客户端资源复制到本地缓存（`.game_cache/`，已 gitignore，不提交不分发）：

```powershell
.\web\scripts\prepare_assets.ps1 -GameDir "E:\path\to\Rusted Warfare1.15PC"
```

Dockerfile / compose / API / 目录说明见 [web/README.md](./web/README.md)。

## 真机 PPO 训练（策略在环）

策略网络直接在真实 1.15 无头对局里决策（观测走 bridge `obs` 流，动作注入原生 `CommandController`），默认场景是 `Artillery` 挑战图：引擎预置 **22 我方单位 vs 25 AI 单位**，开局即真实交战。

```powershell
$env:PYTHONPATH = "src"
python -m ironzero.training.train_rwpp --updates 6 --envs 2 --episode-ticks 900 --device auto
# 产物：runs/rwpp_ppo/checkpoint.pt、live.jsonl（可直接接大屏）、rwpp_summary.json
```

- 每实例 = 一个独立无头游戏进程（端口 5140+），`--envs` 控制并行实例数；实例跨 update 复用，只有单局结束才重启 JVM
- 观测 364 维、动作 5 头 `[kind5, group5, x8, y8, target9]`、策略网络 168k 参数；奖励 = HP 潜势差 + 交战接近 + 与战略目标接近 + 击杀/损失/无效动作/空闲分项 + 终局 ±1（每项都进 `reward_breakdown` 与 PPO live 事件）
- 加速：`--frame-ms` **只控制每帧墙钟预算**（16=真实时间，2≈快，1≈更快），不改变仿真时长；每局仿真时长 = `--episode-ticks`（60 tick ≈ 1 秒，默认 3600 ≈ 60 秒）。引擎判出胜负会提前结束该局
- 胜负上报：桥接每帧轮询原生结算（`TriggerEngine` 的 mainBuilings 目标 + `Team.defeated`），终局 `obs` 带 `result`/`terminated`，`done.result` 为 `victory`/`defeat`/`timeout`；训练评估、推理回放和平台界面都显示真实胜负，不再把失败/胜利记成超时
- 观测迷雾：默认 `--vision-radius 260`（策略只看我方 260px 内敌人，胜负仍按全量状态）
- 细节与场景选型见 `docs/平台搭建方案.md` 5.7 节，平台侧在「实机训练」页切到「PPO 真机训练」即可看训练大屏：

![PPO 真机训练大屏](docs/assets/platform_real_ppo.png)

## 模型推理与 LLM + RL 混合（骨架）

- **模型推理**：`python scripts/play_rwpp_policy.py --checkpoint runs/rwpp_ppo/checkpoint.pt --ticks 3600` 让 PPO 策略在真机里打一局并生成回放（`meta.result` 为真实胜负）；平台「实机推理」页策略下拉选「PPO 模型」即可
- **LLM 战略层**：`--use-llm` 开启后每 3 秒调用一次 OpenAI 兼容 LLM（环境变量 `IRONZERO_LLM_BASE_URL / IRONZERO_LLM_API_KEY / IRONZERO_LLM_MODEL`），LLM 只输出 `StrategicDirective`（intent / attack_region / rally_point），**不控制单位级动作**；无 key 或调用失败自动退回规则策略，关闭后 RL 独立运行
- 动作空间：`move_group / attack_move / attack_target / stop / hold` 五类；单位组头真实生效（4 组 + 广播），对齐设计文档 4.2

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

从固定的 RWPP/RWML/game-lib 版本自动构建兼容 runtime：

```powershell
.\scripts\build_rwpp_compat.ps1 `
  -GameDir "D:\path\to\Rusted Warfare" `
  -BuildRoot "$env:LOCALAPPDATA\IronZero" `
  -ProxyUrl "http://127.0.0.1:7897"
```

脚本会完成源码下载、兼容补丁、RWML Desktop 构建、RWPP Server 构建、assets 组装和 180 tick 冒烟测试。

执行 RWPP 无头冒烟测试：

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir "D:\rwpp-runtime" `
  -JavaHome "D:\jdks\jdk-25" `
  -Ticks 180
```

协议模型见 `src/ironzero/protocol.py`，机器可读 schema 见 `schemas/ironzero/1/messages.schema.json`。

离线参考环境与训练（不需要 JDK 25 / RWPP 运行时）：

```powershell
$env:PYTHONPATH = "src"

# 从本机游戏资源导入数值清单（已提交 configs/game_assets.json，可重复校验）
python tools/import_game_assets.py --game-dir "E:\path\to\Rusted Warfare1.15PC"

# 训练 micro-duel 策略（16 个并行环境，CPU 约 4 分钟）
python -m ironzero.training.train --updates 200 --envs 16 --rollout-steps 64 --out runs/micro_duel

# 评估 checkpoint，同时跑随机合法动作基线
python -m ironzero.training.evaluate_cli runs/micro_duel/checkpoint.pt --episodes 48

# 以 NDJSON/TCP 启动 mock 游戏侧，验证 Bridge 协议链路
python -m ironzero.mock_server --port 5123
```

设计说明、里程碑与环境准备清单见 [平台搭建方案](./docs/平台搭建方案.md)。

文档覆盖：

- 游戏接入路线对比
- Java Agent 与 RWPP Headless 架构
- 状态空间、动作空间和奖励函数设计
- 桥接协议与游戏线程安全
- PPO、多环境与评估方案
- 推理部署与 LLM + RL 混合架构
- 分阶段实施计划和风险控制

## 计划路线

- [x] 只读探针：读取 tick、单位、资源、建筑和迷雾（真机 bridge 已接）
- [x] 动作桥：移动、攻击、停止/守区 + 单位组（建造/生产待接）
- [x] Micro 场景：固定地图、小规模单位对战（离线 8v8 + 真机 22v25 均已训练/评估）
- [ ] 经济场景：采矿、建造、生产和资源点
- [x] 完整 1v1 部分：多地图 + 战争迷雾（仅观测级半径裁剪）+ 内置 AI 基线（单一难度；自博弈待做）
- [ ] 自博弈与模型晋级
- [x] LLM + RL 混合指挥（战略层骨架 + 规则降级，PPO 战术层已打通）

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
