# IronZero 平台（训练 / 推理 / 监控大屏）

> 作者：周培智
> 状态：可运行（Docker + 本机开发两种方式）
> 依赖：Docker Desktop（推荐）或本机 Python 3.11+ / Torch

一个面向《铁锈战争》强化学习项目的本地平台，用来**启动训练、实时看指标、跑推理、看 agent 对打、录制回放**。

```text
┌──────────────────────────── 浏览器 ────────────────────────────┐
│ 训练监控（曲线/KPI/日志）  │  对局沙盘（实时帧）  │  回放库      │
└──────────────────────────────┬─────────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼─────────────────────────────────┐
│ FastAPI 平台后端  web/backend/app.py                           │
│  /api/runs      训练任务（子进程 + events.jsonl + 日志）        │
│  /api/infer     推理任务（逐帧推流 frames.jsonl）               │
│  /api/replays   回放库（JSON 帧序列）                           │
│  /api/checkpoints 模型列表                                     │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│ ironzero 训练 / 推理库（src/ironzero）                          │
│  PPO + 动作掩码 + 确定性微操环境（真实 1.15 坦克数值校准）       │
└────────────────────────────────────────────────────────────────┘
```

## 1. 启动（Docker，推荐）

```powershell
# 首次构建（需要网络；通过 host.docker.internal 走本机代理）
docker compose -f web/docker-compose.yml up -d --build

# 打开大屏
start http://127.0.0.1:8000
```

- 宿主机 `runs/` 挂载进容器 `/app/runs`，模型、日志、回放都持久化在宿主机。
- 宿主机 `web/frontend/` 以只读方式挂载进容器，改前端刷新浏览器即可，不必重建镜像（后端改动需重建）。
- 停止：`docker compose -f web/docker-compose.yml down`
- 查看日志：`docker logs -f ironzero-platform`
- 只重建镜像：`docker compose -f web/docker-compose.yml up -d --build`

如果构建时拉依赖失败，确认 Docker Desktop 能访问代理；也可以在 `web/docker-compose.yml` 里修改 `HTTPS_PROXY` 构建参数。

## 2. 启动（本机开发，不装 Docker）

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 8000 --reload
```

前端是纯静态文件（`web/frontend`），由同一个服务在 `/` 与 `/static/*` 提供，改完刷新即可。
静态预览（不启动后端时）：`python -m http.server 8011 --directory web/frontend`。

## 3. 页面功能

### 训练监控

- 表单启动 PPO：`updates / envs / rollout_steps / opponent / seed / eval_every / eval_episodes / lr / entropy`
- KPI：最新 update、训练窗口胜率、平均回报、entropy、策略/价值损失
- 曲线：评估胜率、平均回报（含随机基线参考线）
- 日志：训练子进程 stdout 尾巴，失败原因直接可见
- 停止：终止训练子进程（已产出的 checkpoint 保留）

训练任务目录（宿主机 `runs/platform/train-*`）：

```text
meta.json       任务元数据与状态
events.jsonl    逐 update / eval / done 事件（前端增量拉取）
stdout.log      训练日志
metrics.json    训练结束后的完整指标
checkpoint.pt   模型权重
```

### 对局沙盘（人工观测 agent 对打）

- 选择 checkpoint（列表来自 `runs/**/checkpoint.pt`）、局数、对手、随机种子、动作模式（贪婪/采样/随机）
- 后端逐帧推流，前端 canvas 实时渲染：坦克、血条、编号、指令箭头、shot 弹道、kill 爆炸
- 支持播放 / 暂停 / 0.5–4 倍速 / 进度条 / 跳到最新
- 「录制视频」：`canvas.captureStream(30)` + `MediaRecorder`，浏览器直接下载 `.webm`
- 每局结束自动写入回放库

### 回放库

- 列出所有录制回放（对手、结果、回报、帧数、时间）
- 点击载入沙盘重放；支持删除
- 回放是纯 JSON（`ironzero.replay/1`），可以直接用别的工具消费

## 4. API 速查

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET/POST | `/api/runs` | 列表 / 启动训练 |
| GET | `/api/runs/{id}` | 状态、最新指标、日志尾巴 |
| GET | `/api/runs/{id}/events?after=N` | 增量训练事件 |
| POST | `/api/runs/{id}/stop` | 停止训练 |
| GET | `/api/checkpoints` | 可用模型 |
| POST | `/api/infer` | 启动推理（逐帧推流） |
| GET | `/api/infer/{id}` | 推理状态 |
| GET | `/api/infer/{id}/frames?after=N` | 增量帧 |
| GET | `/api/replays` | 回放列表 |
| GET/DELETE | `/api/replays/{id}` | 读取 / 删除回放 |

## 5. 目录

```text
web/
  backend/app.py        FastAPI 服务（任务、事件、回放）
  frontend/             纯静态大屏（index.html / app.js / style.css）
  Dockerfile            平台镜像（python:3.12-slim + CPU torch）
  docker-compose.yml    一键启动
  scripts/              开发/运维小脚本
src/ironzero/           训练、环境、回放、协议（被平台调用）
runs/                   产物（已 gitignore）
```

## 6. 界面

![训练监控](../docs/assets/platform_training.png)

![对局沙盘](../docs/assets/platform_sandbox.png)

![回放库](../docs/assets/platform_replays.png)

## 7. 边界

- 平台当前跑的是**离线参考环境**（按真实 1.15 坦克数值校准的 8v8 微操），不是游戏本体；真机 RWPP 接入后只需把后端里的训练/推理命令换成 TCP 适配器版本。
- 平台默认只在本机/内网使用，不要暴露到公网。
