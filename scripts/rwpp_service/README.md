# `scripts/rwpp_service/` — 宿主机真机服务（模块边界说明）

> 本包由 `scripts/rwpp_real_service.py`（1014 行）拆分而来，**只搬结构、不改行为**。
> 入口仍是脚本 `scripts/rwpp_real_service.py`，它现在只是一个薄 shim。

## 1. 为什么宿主机要单独起一个服务

平台后端跑在 Docker 容器里，容器内没有 Java，所以真实的 Rusted Warfare 1.15
运行时留在 Windows 宿主机上，通过这个小的 HTTP 服务对外提供能力：

```powershell
python scripts\rwpp_real_service.py --port 8300 --max-parallel 10
```

* `--max-parallel N` 决定同时能跑几局真机对局：服务会先把端口 `5130 .. 5130+N-1`
  放进端口池 `PORT_SLOTS`，每个并发实例独占一个 bridge 端口。
  默认值取自 `state.MAX_PARALLEL`（当前 **10**），不是写死的字面量。
* `--host` 默认 `0.0.0.0`（与拆分前一致；如需收敛到回环请显式传 `--host 127.0.0.1`）。

### 1.1 改完源码必须重启服务（最容易踩的坑）

本服务的准入校验、并发上限、命令行拼装全部在**进程启动时**载入内存：`main()`
把 `--max-parallel` 写进 `state.MAX_PARALLEL`、`launch.py` 的 `_PPO_CHECKS` 也是
模块级常量。所以**改完 `.py` 不重启，跑着的旧进程仍按旧规则判定**，前端会收到
一个和当前源码完全对不上的 400，例如：

```
POST /api/rwpp/ppo -> HTTP 400 {"detail":"host rwpp service error: envs must be 1..8"}
```

源码里早就是 `1..16` 了，但 8300 上那个进程是改动之前起的。判断方法：

```powershell
# 上限是否已生效（对比 state.MAX_PARALLEL 与源码）
curl.exe -s http://127.0.0.1:8300/health      # 看 max_parallel / free_slots
# 是谁占着 8300、它的命令行是什么
Get-NetTCPConnection -State Listen -LocalPort 8300 |
  ForEach-Object { Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" } |
  Select-Object ProcessId, CreationDate, CommandLine
```

`CreationDate` 早于源码 `LastWriteTime` 就说明必须重启。重启步骤：

```powershell
# 1) 停旧进程（连同它的父 shell，否则父进程可能重新拉起）
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*rwpp_real_service*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
# 2) 确认端口释放
(Get-NetTCPConnection -State Listen -LocalPort 8300 -ErrorAction SilentlyContinue).Count  # 期望 0
# 3) 用默认（10 路）重新起，日志落到 runs/_logs/
$env:PYTHONPATH="src"; $env:IRONZERO_RWPP_RUNTIME="E:\IronZeroBuild\rwpp-runtime-1.15"
Start-Process python -ArgumentList "scripts\rwpp_real_service.py","--port","8300" `
  -WorkingDirectory $PWD -WindowStyle Hidden `
  -RedirectStandardOutput "runs\_logs\rwpp_service.stdout.log" `
  -RedirectStandardError  "runs\_logs\rwpp_service.stderr.log"
```

重启会丢掉服务进程内的 `JOBS` 表（前端"恢复运行中任务"会查不到已完成的历史），
所以**别在训练跑到一半时重启**——先停任务再重启。

### 1.2 两套 runtime 与 bridge jar 必须同步

仓库里同时存在多个 RWPP 运行时，`assets`/`lib`/`bin` 内容一致，**只有
`extensions\ironzero-bridge.jar` 会不同步**，而桥接的参数解析全在这个 jar 里：

| 路径 | 角色 |
| --- | --- |
| `E:\IronZeroBuild\rwpp-runtime-1.15` | `RwppGameConfig.DEFAULT_RUNTIME`，训练/评估默认读它；也是 `build_rwpp_extension.ps1` 的默认安装目标 |
| `C:\IronZero\rwpp-runtime-1.15` | 历史启动方式用 `IRONZERO_RWPP_RUNTIME` 指向它 |
| `.runtime\bridge-dist\ironzero-bridge.jar` | 构建暂存产物，**不是权威源** |

症状：Python 侧已经按新参数拼命令行，旧 jar 却不认识该参数，于是"代码明明改了但
真机行为没变"。**唯一权威源是 `build_rwpp_extension.ps1` 重新编译出来的那份**——
`.runtime\bridge-dist` 只是上一次运行的暂存，可能比 runtime 里的还旧，**不要
直接从它拷去覆盖**，否则等于把桥接回滚。

正确顺序（构建脚本本身会把新 jar 装进 `-RuntimeDir`）：

```powershell
# 1) 从当前 Java 源码重新构建；这会写 .runtime\bridge-dist 并装入 -RuntimeDir
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_rwpp_extension.ps1 `
  -RuntimeDir "E:\IronZeroBuild\rwpp-runtime-1.15" -JavaHome "E:\jdk25copy"
# 2) 再把刚构建的那份分发到其余在用的 runtime
Copy-Item "E:\IronZeroBuild\rwpp-runtime-1.15\extensions\ironzero-bridge.jar" `
          "C:\IronZero\rwpp-runtime-1.15\extensions\" -Force
# 3) 校验：三处哈希必须完全一致
Get-FileHash .runtime\bridge-dist\ironzero-bridge.jar,
  E:\IronZeroBuild\rwpp-runtime-1.15\extensions\ironzero-bridge.jar,
  C:\IronZero\rwpp-runtime-1.15\extensions\ironzero-bridge.jar | Select-Object Hash, Path
```

快速确认 jar 里是否含某个新参数（不用解压工具，也不依赖 PowerShell 版本）：

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead("E:\IronZeroBuild\rwpp-runtime-1.15\extensions\ironzero-bridge.jar")
$e = $zip.GetEntry("ironzero/IronZeroBridge.class")
$ms = New-Object System.IO.MemoryStream; $e.Open().CopyTo($ms); $zip.Dispose()
$bytes = $ms.GetBuffer()
$blob = -join (0..($ms.Length-1) | ForEach-Object { $b=$bytes[$_]; if ($b -ge 32 -and $b -lt 127) { [char]$b } else { ' ' } })
[regex]::Matches($blob, '--[a-z][a-z-]+') | ForEach-Object Value | Sort-Object -Unique
# 期望含 --physics-ms（新增）、--bridge-port/--frame-ms/--ticks/--map 等
```

`tests/test_concurrency_and_pacing.py::test_deployed_bridge_jar_knows_the_flags_python_sends`
会在本机存在 runtime 时自动比对"`process.py` 能拼出的每个 `--flag`"是否都在 jar 里，
所以忘了同步会在 `pytest` 阶段直接红掉（无 runtime 的 CI 上自动 skip）。

### 1.3 重启不丢训练：任务落盘 + 自动接管（`recover.py`）

`run_ppo` 的轮询线程活在**服务进程里**，而 trainer 是独立子进程（它自己再拉一批
JVM）。所以服务一重启，内存里的 `JOBS` 就空了，但机器上其实还在训练：Web 大屏
"什么都看不到"，用户既不能观察也不能停止，只能杀掉重跑。**这曾经真实发生过**——
重启一次服务，正在跑的第 280/500 轮训练从平台界面上消失了。

现在由 `recover.py` 兜住：

| 机制 | 说明 |
| --- | --- |
| 任务落盘 | `paths._update` 每次变更都把快照写到 `runs/_rwpp_service/jobs/<id>.json`（`os.replace` 原子写；写失败被吞掉，绝不影响在跑的任务） |
| 恢复顺序 | `server.main()` 在 `listen()` **之前**调 `recover_jobs()`，所以第一次 `/jobs` 轮询就能看到接管的任务 |
| 恢复校验 | 落盘记录若声称 `running`，必须按产物重新判定，否则重启会把几天前就跑完的任务复活成"正在跑"（见下表） |
| 无记录兜底 | 本功能上线前启动的任务没有落盘记录，于是扫描 `runs/platform/rwpp-ppo-*`：有 `live.jsonl`、没有 `rwpp_summary.json`、且 `live.jsonl` 仍在更新 → 接管 |
| pid 归属 | 用 `psutil` 按 trainer 命令行里的 `--out <dir>` 精确匹配 pid；没有 psutil 时降级为只读监控（能看到进度，停止需手工杀） |
| 场景字段 | 地图名只存在于 trainer 的 argv（启动摘要里没有 `map`），所以 `--map`/`--map-name`/`--frame-ms`/`--vision-radius`/`--units-per-team` 从命令行取 |
| 停止 | 接管任务没有本地 `Popen` 句柄，`stop_job` 改为按 pid 用 `taskkill /T /F` 整棵杀（Windows 零依赖） |

恢复后的终态判定**只看证据，不猜**：

| 证据 | 判定 |
| --- | --- |
| `rwpp_summary.json` 出现 | `done`（并读出 summary） |
| `train.log` 出现 `Traceback` | `failed`（错误取最后一行） |
| pid 已消失且无 summary | `failed` |
| `live.jsonl` 超过 `STALE_SECONDS`(300s) 没更新且无 summary | `stale` |

`stale` 是刻意**不报 failed**：训练可能只是卡在一个很慢的 update 上，诚实地说"没动静了"
比谎报失败更有用。把 `status=stale` 当作"需要人看一眼"即可。

不想接管（例如要干净地换一份配置）时加 `--no-recover`：

```powershell
python scripts\rwpp_real_service.py --port 8300 --no-recover
```

回归用例：`tests/test_service_recovery.py`（落盘/恢复、`running` 重新校验、只接管新鲜的
`live.jsonl`、命令行场景字段、不重复接管）。`psutil` 是可选依赖，缺失时只影响"能否按
pid 停止"，不影响进度可见性。

## 2. 模块划分与依赖方向

```
server.py       main(argv)：argparse → 填端口池 → recover_jobs() → ThreadingHTTPServer((host, port), Handler)
  └─ handler.py Handler(BaseHTTPRequestHandler)：路由分派 + JSON/CORS 响应
       ├─ launch.py  StartMixin：把 HTTP payload 变成 job（/infer /train /ppo 的准入校验）
       │    ├─ ppo.py      run_ppo / _read_live_events / resolve_preset / resolve_init_checkpoint
       │    ├─ runners.py  run_infer / run_train / _run_recorder / _run_player
       │    └─ paths.py    REPO_ROOT / RUNS / JOB_STATE_DIR / _update / _persist_job / _relative
       ├─ proc.py     stop_job / stop_all_jobs（含接管任务的 pid 树停止）
       │    └─ recover.py  load_persisted_jobs / recover_jobs / kill_pid_tree
       └─ state.py   全部进程内可变全局态（唯一所有者）
```

依赖是单向的：`state` ← `paths` ← `recover` ← `proc` ← `runners`/`ppo` ← `launch` ←
`handler` ← `server`。没有循环导入，也没有模块在导入期启动线程或进程
（`recover_jobs()` 只在 `main()` 里被显式调用）。

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `state.py` | 53 | `JOBS` / `JOBS_LOCK` / `PORT_SLOTS` / `MAX_PARALLEL` / `ACTIVE_PROCESSES` / `STOPPED_JOBS` / `PPO_PROCESSES` / `PPO_STOP` / `TICK_OVERHEAD_MS` / `EPISODE_TIMEOUT_MARGIN_S` |
| `paths.py` | 80 | 仓库根与脚本路径常量、任务记录读写与**落盘**（`JOB_STATE_DIR` / `_persist_job`） |
| `recover.py` | 489 | 重启后恢复/接管在跑的任务、pid 归属、`taskkill` 停止、产物级终态判定 |
| `proc.py` | 120 | 受管子进程执行（登记 → communicate → 清理）、单任务/全量停止、单局超时预算 |
| `runners.py` | 323 | `/infer`（单局）与 `/train`（多局并行采集）的执行体 |
| `ppo.py` | 330 | `/ppo` 训练执行体、live 事件读取、arena/预热权重预设收敛 |
| `launch.py` | 281 | `StartMixin`：`_start` / `_start_ppo` / `_ppo_running` |
| `handler.py` | 183 | `Handler`：`do_GET` / `do_POST` / `do_OPTIONS` / `_send` |
| `server.py` | 60 | `main(argv)`（含 `--no-recover`）与 `ThreadingHTTPServer` 装配 |

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
