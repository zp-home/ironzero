# `scripts/rwpp_service/` — 宿主机真机服务（模块边界说明）

> 本包由 `scripts/rwpp_real_service.py`（1014 行）拆分而来，**只搬结构、不改行为**。
> 入口仍是脚本 `scripts/rwpp_real_service.py`，它现在只是一个薄 shim。

## 1. 为什么宿主机要单独起一个服务

平台后端跑在 Docker 容器里，容器内没有 Java，所以真实的 Rusted Warfare 1.15
运行时留在 Windows 宿主机上，通过这个小的 HTTP 服务对外提供能力：

```powershell
python scripts\rwpp_real_service.py --port 8300 --max-parallel 4
```

* `--max-parallel N` 决定同时能跑几局真机对局：服务会先把端口 `5130 .. 5130+N-1`
  放进端口池 `PORT_SLOTS`，每个并发实例独占一个 bridge 端口。
* `--host` 默认 `0.0.0.0`（与拆分前一致；如需收敛到回环请显式传 `--host 127.0.0.1`）。

## 2. 模块划分与依赖方向

```
server.py       main(argv)：argparse → 填端口池 → ThreadingHTTPServer((host, port), Handler)
  └─ handler.py Handler(BaseHTTPRequestHandler)：路由分派 + JSON/CORS 响应
       ├─ launch.py  StartMixin：把 HTTP payload 变成 job（/infer /train /ppo 的准入校验）
       │    ├─ ppo.py      run_ppo / _read_live_events / resolve_preset / resolve_init_checkpoint
       │    ├─ runners.py  run_infer / run_train / _run_recorder / _run_player
       │    └─ paths.py    REPO_ROOT / RUNS / _update / _relative / _append_replay
       └─ proc.py     stop_job / stop_all_jobs
            └─ state.py   全部进程内可变全局态（唯一所有者）
```

依赖是单向的：`state` ← `paths` ← `proc` ← `runners`/`ppo` ← `launch` ← `handler` ← `server`。
没有循环导入，也没有模块在导入期启动线程或进程。

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `state.py` | 42 | `JOBS` / `JOBS_LOCK` / `PORT_SLOTS` / `MAX_PARALLEL` / `ACTIVE_PROCESSES` / `STOPPED_JOBS` / `PPO_PROCESSES` / `PPO_STOP` / `TICK_OVERHEAD_MS` / `EPISODE_TIMEOUT_MARGIN_S` |
| `paths.py` | 36 | 仓库根与脚本路径常量、任务记录读写小工具 |
| `proc.py` | 94 | 受管子进程执行（登记 → communicate → 清理）、单任务/全量停止、单局超时预算 |
| `runners.py` | 283 | `/infer`（单局）与 `/train`（多局并行采集）的执行体 |
| `ppo.py` | 294 | `/ppo` 训练执行体、live 事件读取、arena/预热权重预设收敛 |
| `launch.py` | 259 | `StartMixin`：`_start` / `_start_ppo` / `_ppo_running` |
| `handler.py` | 170 | `Handler`：`do_GET` / `do_POST` / `do_OPTIONS` / `_send` |
| `server.py` | 35 | `main(argv)` 与 `ThreadingHTTPServer` 装配 |

## 3. 路由表

所有前缀、状态码、响应字段与拆分前**逐字节一致**（前端按路径硬编码调用）。

| 路径 | 方法 | 成功响应 | 说明 |
| --- | --- | --- | --- |
| `/health` | GET | `{ok, source, jobs, running, ppo_running, max_parallel, free_slots}` | 健康检查 + 端口池余量 |
| `/jobs` | GET | `{"jobs": [...]}` | 最近 50 条任务（running 的 `elapsed` 实时刷新） |
| `/jobs/{id}` | GET | 任务记录本身 | 单任务详情；不存在 → 404 `{"error": "job not found"}` |
| `/ppo/{id}/events` | GET | `{next, events, status, job}` | 按 `?after=<seq>` 增量返回 live 事件 |
| `/infer` | POST | `{job_id, status, kind}` | 单局实机推理（`policy=scripted\|none\|model\|ppo`） |
| `/record` | POST | 同上 | `/infer` 的别名，走同一分支 |
| `/train` | POST | `{job_id, status, kind}` | 多局并行采集（`episodes` 1..50、`concurrency` 1..max_parallel） |
| `/ppo` | POST | `{job_id, status, kind}` | 实机 PPO 训练 |
| `/jobs/{id}/stop` | POST | `{ok, job_id, stopped}` | 停止单任务 |
| `/jobs/stop_all` | POST | `{ok, stopped_count}` | 停止全部 running |
| `/jobs/stop-all` | POST | 同上 | 别名 |
| `/stop_all` | POST | 同上 | 别名 |

未知路径 → 404 `{"error": "not found"}`；非法 JSON → 400 `{"error": "invalid json"}`。
所有响应都带 `Access-Control-Allow-Origin: *`（容器内后端跨主机调用）。

## 4. 任务生命周期

### 4.1 创建（`launch.py`）

1. 解析 payload 并校验（`policy` / `ticks` 120..216000 / `concurrency` / `episodes` /
   PPO 的 `frame_ms` `vision_radius` `updates` `envs` `rollout_steps` `episode_ticks` `device`）。
   校验顺序与文案与拆分前完全一致。
2. **互斥检查**：已有 running 的 PPO 任务 → 409；启动 PPO 时若还有 running 的实机任务 → 409；
   启动实机任务时若端口池为空 → 409。
3. 在 `JOBS[job_id]` 写入初始记录（`status="running"`、`started_ts`、`episodes_metrics` 等）。
4. 起一个 daemon 线程跑 `run_infer` / `run_train` / `run_ppo`，**立即**回 `job_id`。

### 4.2 执行（`runners.py` / `ppo.py` / `proc.py`）

* 实机任务每局从 `PORT_SLOTS` 取一个端口 → `_run_managed_proc` 启动
  `record_rwpp_replay.py`（scripted/none）或 `play_rwpp_policy.py`（model/ppo checkpoint），
  子进程登记进 `ACTIVE_PROCESSES[job_id]`；结束后端口归还 `PORT_SLOTS`。
* `run_train` 用 `min(concurrency, episodes, MAX_PARALLEL)` 个 worker 线程共享任务队列与端口池，
  每次完成一局就累加 `episodes_done` / `episodes_metrics` / `frames` / `ticks_total`。
* `run_ppo` 拉起 `python -m ironzero.training.train_rwpp`，stdout 写 `runs/platform/rwpp-ppo-{id}/train.log`；
  每 2 s 读一次 `live.jsonl`，把 `updates_done` / `latest_update` / `evaluations` / `elapsed`
  写回 job —— 这就是 `/ppo/{id}/events` 的数据来源。

### 4.3 终态

| 终态 | 触发 |
| --- | --- |
| `done` | 拿到 replay（实机）或 trainer 退出码 0（PPO） |
| `failed` | 异常 / 无 replay / trainer 非 0 退出（带 `error`） |
| `stopped` | `stop_job` / `stop_all_jobs` 被调用 |

停止路径 `proc.stop_job()` 会持 `JOBS_LOCK` 把 job 置 `stopped`、把 job_id 加进
`STOPPED_JOBS`，然后 kill `ACTIVE_PROCESSES[job_id]` 与 `PPO_PROCESSES[job_id]`
（PPO 同时把 job_id 加进 `PPO_STOP`，让 `run_ppo` 把终态记成 `stopped` 而非 `failed`）。

## 5. 全局态约定（重要）

所有可变全局状态只在 `state.py` 里定义，其它模块 `from .state import JOBS, JOBS_LOCK, ...`：

* `JOBS` / `ACTIVE_PROCESSES` / `STOPPED_JOBS` / `PPO_PROCESSES` / `PPO_STOP` / `PORT_SLOTS`
  是**容器对象**，import 的是引用，**只能原地增删，不得重新赋值**。
* `MAX_PARALLEL` 是 `int`，会被 `main()` 改写。读取方一律
  `from . import state` 再取 `state.MAX_PARALLEL`，否则 import 那一刻的旧值会被绑死。

## 6. 路径语义（`REPO_ROOT` 的坑）

拆分前 `scripts/rwpp_real_service.py` 用 `Path(__file__).resolve().parents[1]` 拿仓库根。
现在同样的代码在 `scripts/rwpp_service/paths.py` 里，多了一层目录，因此是 **`parents[2]`**：

```
parents[0] = scripts/rwpp_service
parents[1] = scripts
parents[2] = <仓库根>
```

验证方式：

```powershell
python -X utf8 -c "import sys; sys.path.insert(0,'scripts'); from rwpp_service.paths import REPO_ROOT,RUNS,RECORDER,PLAYER; print(REPO_ROOT); print(RUNS); print(RECORDER.exists(), PLAYER.exists())"
# E:\qwq\强化学习笔记\铁锈战争
# E:\qwq\强化学习笔记\铁锈战争\runs\platform
# True True
```

## 7. 自测清单

```powershell
# 入口 shim 可导入
python -X utf8 -c "import sys; sys.path.insert(0,'scripts'); import rwpp_real_service; print(rwpp_real_service.main)"

# 冒烟起服务（另开一个终端请求）
python scripts\rwpp_real_service.py --port 8311 --max-parallel 2
# GET http://127.0.0.1:8311/health → {"ok": true, "jobs": 0, "running": 0, "max_parallel": 2, "free_slots": 2, ...}
# GET http://127.0.0.1:8311/jobs   → {"jobs": []}

# 全量回归
python -m pytest -q
```

> Windows 控制台把中文显示成乱码属于**显示问题**，文件本身是 UTF-8（无 BOM）、LF 换行，
> 用带 `-X utf8` 的解释器或 read 工具查看。
