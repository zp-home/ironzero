# web/backend/app/ — 平台后端（FastAPI）

IronZero 训练/推理平台的后端服务。原单文件 `web/backend/app.py`（763 行 / 拆分时已增至
891 行）按职责拆成本包，**行为零变化**：路由路径、HTTP 方法、查询参数名/默认值/约束、
响应字段名与拆分前逐字节一致（OpenAPI 全量 diff 为空）。

入口仍是 `web.backend.app:app`，`web/Dockerfile` 的 `CMD` 与 `web/scripts/dev.ps1`
无需改动。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 只做转发：`from .server import app, create_app`，`__all__ = ["app", "create_app"]` |
| `server.py` | `create_app()`：`FastAPI(title="IronZero Platform", version=APP_VERSION)`、CORS 中间件、`include_router`、`/static` 挂载；模块级 `app = create_app()` 供 uvicorn 使用 |
| `paths.py` | `REPO_ROOT` / `RUNS_ROOT` / `FRONTEND_DIR` / `APP_VERSION` / `BEIJING_TZ`、`_resolve_checkpoint_roots()`、`_resolve_inside()`、`RWPP_SERVICE` |
| `models.py` | 4 个 pydantic 请求体：`TrainRequest` / `InferRequest` / `RwppRecordRequest` / `RwppPpoRequest`（字段默认值与约束即前端契约） |
| `jobs.py` | 任务目录与生命周期：`_now`/`_new_job_dir`/`_write_meta`/`_read_meta`/`_read_jsonl`/`_tail`/`_python`/`_job_env`/`_spawn`/`_job_status`/`_latest_from_events`/`_list_train_runs`/`_find_checkpoints`，以及进程注册表 `_jobs` + `_jobs_lock` |
| `routers/` | 按资源分组的 HTTP 路由，一个文件一组，见下表 |

`paths.py` 的根目录用 `Path(__file__).resolve().parents[3]` 推导（原 `app.py` 是
`parents[2]`）：包内文件比原模块多一层，`REPO_ROOT` 仍指向仓库根。`web/backend/maps.py`
未改动，路由通过 `from ... import maps as gamemaps` 复用同一模块对象。

## 路由分组

| 模块 | 路由 |
| --- | --- |
| `routers/system.py` | `GET /api/health`、`GET /`（dashboard）、`GET /favicon.ico` |
| `routers/runs.py` | `GET /api/runs`、`POST /api/runs`、`GET /api/runs/{run_id}`、`GET /api/runs/{run_id}/events?after=`、`POST /api/runs/{run_id}/stop` |
| `routers/infer.py` | `POST /api/infer`、`GET /api/infer/{infer_id}`、`GET /api/infer/{infer_id}/frames?after=` |
| `routers/replays.py` | `GET /api/replays?limit=&offset=`、`GET /api/replays/{replay_id:path}`、`DELETE /api/replays/{replay_id:path}` |
| `routers/checkpoints.py` | `GET /api/checkpoints` |
| `routers/maps.py` | `GET /api/maps?force=`、`GET /api/maps/{map_id}`、`GET /api/assets/{asset_path:path}` |
| `routers/rwpp.py` | `/api/rwpp/{status,record,infer,train,jobs}`、`/api/rwpp/infer/{job_id}/frames?after=`、`/api/rwpp/ppo`、`/api/rwpp/ppo/{job_id}/events?after=`、`/api/rwpp/{ppo,jobs}/{job_id}/stop`、`/api/rwpp/stop_all` |

注册顺序在 `server.py` 中固定为：system → runs → infer → replays → checkpoints →
maps → rwpp → `/static` 挂载 → **favicon**。`/favicon.ico` 由 `system.favicon_router`
在挂载之后注册，只是为了保持拆分前的路由表顺序（路径互不重叠，不影响匹配）。改动
提权顺序请同步核对 `app.routes`。

`routers/rwpp.py` 内的 `_rwpp_call()` 是把请求转发给宿主机实机服务
（`IRONZERO_RWPP_SERVICE`，默认 `http://127.0.0.1:8300`）的唯一出口；上游的
400/404/409/429 原样透传，其余错误码映射为 502，连不上映射为 503。

## job 生命周期

一个 job = `RUNS_ROOT` 下的一个目录（默认 `runs/platform/<kind>-<YYYYmmdd-HHMMSS>-<6hex>/`）。

1. **创建** — `_new_job_dir(kind)` 生成 id 与目录，`kind` 取 `train` 或 `infer`。
2. **启动** — 路由拼好子进程命令后调用 `_spawn()`：先写 `meta.json`
   （`status="running"` + `config` + `command`），再 `Popen` 起 worker，stdout/stderr
   重定向到 `stdout.log`，并把 `{process, dir, meta, log}` 注册进全局 `_jobs`。
3. **运行** — `_spawn` 起一个 daemon 线程 `watch-<job_id>` 阻塞在 `process.wait()` 上。
   期间产物由 worker 自己写：训练任务是 `events.jsonl`（`kind=update/eval/done`），
   推理任务是 `frames.jsonl`（`type=frame/episode_end/done`）。
4. **结束** — watcher 拿到退出码后关闭日志句柄，把 `status` 改成 `succeeded`/`failed`，
   补 `finished_at` 与 `exit_code`，重写 `meta.json`。
5. **查询** — `_job_status()` 优先看内存里的 `_jobs`（进程还活着 ⇒ `running`），
   否则回落到 `meta.json` 的 `status`。所以**服务重启后历史 job 仍可查询**，只是不再
   出现在 `_jobs` 里。
6. **停止** — `POST /api/runs/{run_id}/stop` 只对当前进程内的 job 生效（不在 `_jobs`
   里直接 404）：`terminate()` → 等 8 秒 → 仍然活着就 `kill()`，然后把 meta 改成
   `stopped`。

约束：`_jobs` 的读写一律持 `_jobs_lock`；`meta.json` 用 `ensure_ascii=False, indent=2`
写 UTF-8，时间戳统一走 `_now()`（东八区 `BEIJING_TZ`）。

## 契约提醒（前端硬编码，勿改）

- `/api/maps` 的 `force: bool = False`（普通查询参数，非 `Query(...)`）；
- `/api/infer/{id}/frames`、`/api/runs/{id}/events`、`/api/rwpp/infer/{id}/frames`、
  `/api/rwpp/ppo/{id}/events` 的 `after: int = Query(default=0, ge=0)`；
- `/api/replays` 的 `limit: int = Query(default=0, ge=0, le=1000)`、
  `offset: int = Query(default=0, ge=0)`；
- 响应字段名（如 `next` / `frames` / `status` / `frames_total` / `latest_frame` /
  `eval_count` / `final_eval` / `log_tail`）逐字保留；
- `app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), check_dir=False), name="static")`
  与 CORS（`allow_origins=["*"]` 等）原样保留。

## 自测

```powershell
# 路由表（应与拆分前完全一致，35 条）
python -X utf8 -c "from web.backend.app import app; import json; rs=[(r.path, sorted(getattr(r,'methods',[]) or [])) for r in app.routes]; print(len(rs)); open('.tmp_routes_after.json','w',encoding='utf-8').write(json.dumps(sorted(rs),ensure_ascii=False,indent=1))"

# 冒烟
python -m uvicorn web.backend.app:app --port 8123   # GET /api/health -> 200，GET / -> index.html

# 全量回归
python -m pytest -q
```
