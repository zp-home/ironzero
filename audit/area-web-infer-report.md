# Web / 推理链路 审计报告（area-web-infer）

审计人：audit-web-infer　｜　审计时间：2026-11-11　｜　基线：工作区未提交改动（`web/frontend/app.js` +148、`web/frontend/index.html` +37、`web/backend/app.py` +25）
审计范围：`web/backend/{app.py,maps.py}`、`web/frontend/{app.js,index.html,style.css}`、`scripts/` 下 7 个推理脚本、`site/`、`src/ironzero/strategy/*`、`tests/{test_strategy,test_rwpp_scripts}.py`
方法：全量 `grep` 交叉验证 + AST 级调用图（Python 脚本计算可达性）+ 实际 `pytest` 复跑。所有断言均带 `文件:行号`。

---

## 1. 一句话结论

**前端 7957 行里有约 6% 是彻底死函数、另有整整 900 行"旧版离屏监控大屏"因 DOM 被删而全程静默 no-op；后端 `CHECKPOINT_ROOTS = [Path("/app/runs")]` 把推理页的模型下拉框在非 Docker 环境下彻底打死；`llm_strategy.py` 只接到训练链路、推理链路 0 接入，`use_llm` 开关在 UI 上"传得动、看不见"，整条战略层在推理侧是装饰品。**

最严重 3 条（详见 §5/§6/§7）：
1. `web/frontend/app.js:33` `apiBase()` 被硬编码成 `http://127.0.0.1:8000`，同时把原来"同源部署 / `?api=` 覆盖"的适配层逻辑删掉 —— Docker、反代、局域网访问全部 CORS 失败。
2. `web/backend/app.py:42` + `:680-699` 把 checkpoint 根目录写死成 `Path("/app/runs")`，Windows 开发机（`/app/runs` 不存在）上 `/api/checkpoints` 恒返回 `[]`，模型推理按钮全部空转；且 `except:` 裸吞异常。
3. `src/ironzero/observations.py:313-317` 用 `# ... (keep existing group logic)` 把"无 group_id 时开放全部组"的掩码逻辑删成"掩码到 group 0"，真机只能指挥 1/4 部队 —— 这是 11 个 pytest 失败中的 1 个，且是**功能级回归**，不是测试过时。

---

## 2. 前后端 API 路由双向对照表（完整）

后端共 30 条业务路由（`web/backend/app.py`，`@app.get/@app.post/@app.delete` 全量枚举，并用 `import web.backend.app; app.routes` 实际加载核对一致），前端 `Api` 门面共 28 个方法（`web/frontend/app.js:84-113`）。

### 2.1 前端 → 后端

| 前端调用（app.js 行） | 方法 + URL | 后端是否存在 |
|---|---|---|
| `Api.health` L85 → 调用点 L982 | GET `/api/health` | ✅ app.py:362 |
| `Api.runs` L86 → 调用点 L1432 | GET `/api/runs` | ✅ app.py:384 |
| `Api.run` L87 → L1458, L1495 | GET `/api/runs/{id}` | ✅ app.py:429 |
| `Api.startRun` L88 → L1557 | POST `/api/runs` | ✅ app.py:389 |
| `Api.stopRun` L89 → L1520 | POST `/api/runs/{id}/stop` | ✅ app.py:457 |
| `Api.runEvents` L90 → L1465, L1501 | GET `/api/runs/{id}/events?after=N` | ✅ app.py:448 |
| `Api.checkpoints` L91 → L2007, L7601 | GET `/api/checkpoints` | ✅ app.py:680（**实现已坏，见 §6-1**） |
| `Api.startInfer` L92 → L2627 | POST `/api/infer` | ✅ app.py:483 |
| `Api.infer` L93 → L2720 | GET `/api/infer/{id}` | ✅ app.py:572 |
| `Api.inferFrames` L94 → L2734 | GET `/api/infer/{id}/frames?after=N` | ✅ app.py:593 |
| `Api.replays` L95 → L2174 | GET `/api/replays` | ✅ app.py:644 |
| `Api.replay` L96 → L2403 | GET `/api/replays/{id}` | ✅ app.py:660 |
| `Api.deleteReplay` L97 → L2490 | DELETE `/api/replays/{id}` | ✅ app.py:668 |
| `Api.maps` L98 → L3061 | GET `/api/maps` | ✅ app.py:708 |
| `Api.mapDetail` L99 → L3596 | GET `/api/maps/{map_id}` | ✅ app.py:728 |
| `Api.assetUrl` L100 → L3044, L3301 | GET `/api/assets/{path}` | ✅ app.py:737 |
| `Api.rwppStatus` L101 → L2241 | GET `/api/rwpp/status` | ✅ app.py:776 |
| `Api.rwppRecord` L102 → L2296 | POST `/api/rwpp/record` | ✅ app.py:782 |
| `Api.rwppJob` L103 → L2350,5621,5639,6460,6593 | GET `/api/rwpp/jobs/{id}` | ✅ app.py:833 |
| `Api.rwppJobs` L104 → L6214, L7424 | GET `/api/rwpp/jobs` | ✅ app.py:824 |
| `Api.rwppInfer` L105 → L6073 | POST `/api/rwpp/infer` | ✅ app.py:787 |
| `Api.rwppInferFrames` L106 → L5600, L5698 | GET `/api/rwpp/infer/{id}/frames?after=N` | ✅ app.py:805 |
| `Api.rwppTrain` L107 → L6399 | POST `/api/rwpp/train` | ✅ app.py:800 |
| `Api.rwppPpo` L108 → L7363 | POST `/api/rwpp/ppo` | ✅ app.py:838 |
| `Api.rwppPpoEvents` L109 → L7222 | GET `/api/rwpp/ppo/{id}/events?after=N` | ✅ app.py:845 |
| `Api.rwppPpoStop` L110 → L7411 | POST `/api/rwpp/ppo/{id}/stop` | ✅ app.py:850 |
| `Api.rwppJobStop` L111 → L6434 | POST `/api/rwpp/jobs/{id}/stop` | ✅ app.py:855 |
| `Api.rwppStopAll` L112 | POST `/api/rwpp/stop_all` | ✅ app.py:860 —— **前端从未调用（0 次）** |

**结论：前端无 404（无"调用了后端没有的路由"）。** 唯一的不匹配是反向的：`rwppStopAll` 定义了但零调用。

### 2.2 后端 → 前端（哪些接口 UI 没有入口）

| 后端路由 | 前端是否调用 | 判定 |
|---|---|---|
| GET `/` (app.py:373) | 否 | 浏览器直连，正常 |
| GET `/favicon.ico` (app.py:869) | 否 | 页面无 `<link rel="icon">`（index.html 检索 0 命中），且 `web/frontend/favicon.svg` **文件不存在** → 永远 404 |
| POST `/api/runs` (app.py:389) | 仅死代码 | ⚠️ 唯一调用点 `app.js:1557` 在 `submitStartRun` 内，而 `submitStartRun` 只被 `bindEvents:4931` 的 `if (els.startForm)` 守卫包裹，`start-form` 元素不存在 → **接口活着，UI 已死** |
| POST `/api/runs/{id}/stop` (app.py:457) | 仅死代码 | ⚠️ 同上，`stopCurrentRun` 仅由不存在的 `els.stopBtn` 触发（`app.js:4932`） |
| GET `/api/runs/{id}/events` (app.py:448) | 仅死代码 | ⚠️ 仅 `selectRun`/`refreshCurrentRun` 调用，而 `selectRun` 只由 `renderRunList` 产出的事件触发，`run-list` 容器不存在 → 列表永不渲染，无法选中 |
| POST `/api/infer` (app.py:483) | 仅死代码 | ⚠️ `submitInfer` L2627 由 `els.inferForm` 守卫（L4933），`infer-form` 不存在 |
| GET `/api/infer/{id}` (app.py:572) | 仅死代码 | ⚠️ `pollInfer` L2720，只由 `startInferPolling`（L2659，在死函数 `submitInfer` 内）启动 |
| GET `/api/infer/{id}/frames` (app.py:593) | 仅死代码 | ⚠️ `pollInfer` L2734，同上 |
| POST `/api/rwpp/stop_all` (app.py:860) | 否 | **完全无入口**：无"全部停止"按钮 |
| POST `/api/rwpp/record` (app.py:782) | 仅 `policy='scripted'` | ⚠️ `app.js:2296` 硬编码 `policy: 'scripted'`，后端 `RwppRecordRequest.policy` 支持 `scripted/none/model/ppo`（app.py:91）四选一，三个值永远走不到 |
| POST `/api/rwpp/infer` (app.py:787) | 是 | 见 §7-2：`ticks/policy/frame_ms` 与后端 `RwppRecordRequest` 字段名对不上 |

> 说明：上表"仅死代码"= 后端实现完整、前端函数也写了，但触发它们的 DOM 元素在 `index.html` 中被删除，因此运行时 100% 不可达。这正是「死代码」与「孤儿接口」的叠加态，修复时应二选一：要么补回 UI，要么删后端。

---

## 3. 前端悬空 DOM 引用 / 孤儿控件清单

验证方法：脚本提取 `index.html` 全部 `id="..."`（175 个），与 `cacheEls()` 的 id 数组（`app.js:394-439`，213 个）、`getElementById('...')`（仅 1 处，`app.js:4863` 的 `fe-train-hint`）、`querySelector('#...')`（0 处）做集合差。

### 3.1 悬空引用（JS 引用了，HTML 里根本没有 → `els.xxx` 恒为 `null`）：**47 个**

| 分类 | id（均 `app.js:394-439`） | 影响 |
|---|---|---|
| 旧「训练监控」整页控件（22 个） | `runs-count`, `run-list`, `start-form`, `start-btn`, `kpi-update`, `kpi-update-foot`, `kpi-win`, `kpi-win-foot`, `kpi-return`, `kpi-return-foot`, `kpi-entropy`, `kpi-entropy-foot`, `legend-win`, `chart-win`, `tip-win`, `legend-return`, `chart-return`, `tip-return`, `run-status`, `stop-btn`, `meta-run`, `meta-kind` | 整块功能静默失效 |
| 旧「训练监控」元信息/日志（6 个） | `meta-created`, `meta-config`, `meta-eval`, `log-count`, `log-panel` | 日志面板永远空白 |
| 旧「离线参考引擎」推理表单（10 个） | `infer-form`, `ckpt-select`, `infer-sim`, `episodes-input`, `infer-opponent`, `infer-seed`, `infer-mode`, `infer-btn`, `infer-hint` | 旧的 `/api/infer` 链路无入口 |
| 旧沙盘计分/事件流（7 个） | `score-own`, `score-enemy`, `hp-own`, `hp-own-val`, `hp-enemy`, `hp-enemy-val`, `sb-ckpt`, `sb-opponent`, `sb-episode`, `sb-frames`, `event-feed` | HUD 与事件流不渲染 |

> 逐条证据示例：`app.js:1379` `const box = els.runList;` → `runList` 对应 `run-list`，HTML 中无此 id；`app.js:1888` `if (!els.chartWin || !els.chartReturn) return;` → 二者均不存在，`drawCharts()` **永远第一行就 return**；`app.js:1971` `const select = els.ckptSelect;` → `ckpt-select` 不存在，`renderCheckpoints()` 永远空转。

### 3.2 孤儿控件（HTML 有 id，JS 从不引用）：**8 个**

| id | index.html 行 | 判定 |
|---|---|---|
| `fe-frame-ms` | 276 | **既是孤儿又是 bug**：这是本次未提交改动新增的「推理速度」下拉框（diff `index.html:+271..+281`），但 `cacheEls()` 未收录 → `els.feFrameMs` **永远 undefined**。而 `app.js:6059` 写的是 `els.feFrameMs ? ... : 2`，三元恒走 2 → **UI 上选 16 / 8 / 4 / 1 全部无效，实际永远 2ms**。这是"控件看得见、点了没用"的典型半成品。 |
| `fe-kpis` | 56 | 容器，仅 CSS 用 |
| `fe-train-kpis` | 324 | 同上 |
| `fe-train-ppo-kpis` | 464 | 同上 |
| `legend-fps` | 91 | 图例文本从不更新（写死"最近 60s"） |
| `legend-train-duration` | 359 | 同上 |
| `panel-infer` | 54 | 由 `switchTab` 用 `.tab-panel` class + `panel.id === target` 处理（`app.js:4837-4838`），非悬空，正常 |
| `panel-train` | 314 | 同上，正常 |

### 3.3 `els` 缓存了但从不读（5 个）

`app.js:410,437,451` 等处缓存了 `rwpp-status-ppo` / `rwpp-status-ppo-text` / `fe-ppo-reward-note` / `replay-content` / `replay-pagination`，全文检索 `els.xxx` 零命中。其中 `rwpp-status-ppo` 因为 `renderRwppStatus()`（`app.js:2206-2212`）改用 `document.querySelectorAll('.rwpp-status')` 批量刷，属**新旧两套写法并存**的残留。

### 3.4 其它 DOM 层问题

| 位置 | 问题 |
|---|---|
| `app.js:5200` | `document.querySelector('.chart-stack')` —— `chart-stack` 在 HTML 中 **0 命中**，ResizeObserver 分支永不执行（死代码）。 |
| `app.js:5154-5164` | `chartPairs` 依赖 `els.chartWin/tipWin/chartReturn/tipReturn`，4 个全 null → `handleChartHover` 全函数死（见 §5）。 |
| `app.js:4513` 与 `app.js:4621`、`app.js:4517` 与 `app.js:4625` | `event.type === 'shot'` / `'kill'` 判断在不同函数中重复抄写（`drawFrameEvents` 与 `spawnFrameEffects`）。 |

---

## 4. 功能完成度表格

| # | 功能 | 前端 | 后端 | 完成度 | 证据 |
|---|---|---|---|---|---|
| 1 | 实机推理（并发 N 路） | ✅ 完整（任务卡/KPI/图表/沙盘） | ✅ `app.py:787` + `rwpp_real_service` | **可用** | `feStartInferInner` app.js:5993-6105 |
| 2 | 实机训练（数据采集） | ✅ 完整 | ✅ `app.py:800` | **可用** | `feStartTrain` app.js:6362 |
| 3 | PPO 真机训练 | ✅ 完整（含奖励分解、战略面板） | ✅ `app.py:838` | **可用（除 LLM 层）** | `feStartPpo` app.js:7282 |
| 4 | 回放库 | ✅ 完整（分页/删除/折叠） | ✅ `app.py:644,660,668` | **可用** | `loadReplays` app.js:2167 |
| 5 | 真机录制按钮 | ⚠️ 只能 scripted | ⚠️ 后端4模式只用1 | **半成品** | app.js:2296 硬编码 |
| 6 | **推理速度选择（frame_ms）** | ❌ 控件无效 | ✅ 支持 0..16 | **未完成** | `fe-frame-ms` 未进 `cacheEls` → app.js:6059 恒 2 |
| 7 | **模型推理（PPO checkpoint）** | ✅ UI 完整 | ❌ `checkpoints` 目录写死 `/app/runs` | **非 Docker 环境不可用** | app.py:42,685 |
| 8 | **LLM 战略层（推理侧）** | ⚠️ 只有训练页有开关 | ❌ 推理链路 0 接入 | **未完成（见 §7-1）** | 见下方详证 |
| 9 | 旧「训练监控」大屏（`/api/runs`） | ❌ DOM 全删，函数变 no-op | ✅ 5 条路由完整实现 | **前端半死不活** | §3.1 |
| 10 | 旧「离线参考引擎」推理（`/api/infer`） | ❌ DOM 全删 | ✅ 3 条路由 + `record.py` 完整 | **死功能** | §3.1 |
| 11 | 地图地形渲染 | ✅ 完整（tileset/迷雾/跟随/道具） | ✅ `maps.py` 全量 | **可用** | app.js:3234-3420 |
| 12 | 沙盘滚轮缩放 | ⚠️ 数学可疑 | — | **待确认** | app.js:3974-3976 的 `cam.scale / customScale` 在 `cam.scale` 已被改写后求逆，等于 `1/1`，见 §6-5 |
| 13 | 单帧"跳到最新"/倍速/进度条 | ✅ 可用 | — | 可用 | app.js:5085-5134 |
| 14 | 录制视频（浏览器 MediaRecorder） | ✅ 实现 | — | 可用（需 `canvas.captureStream`） | app.js:4720-4800 |
| 15 | 战略指令展示面板 | ✅ UI 完整 | ⚠️ 数据只在训练时产生 | **仅训练可见** | app.js:7864 |
| 16 | `/api/rwpp/stop_all` | ❌ 无按钮 | ✅ | **未完成** | app.js 0 调用 |
| 17 | favicon | ❌ 无 link、无文件 | ⚠️ 路由在 | **未完成** | app.py:869-874 / index.html 无 link |
| 18 | `site/` 静态站 | ✅ 完整 | — | **独立存活，未被废弃（见 §8）** | `.github/workflows/pages.yml:37` |

### 关键功能详证 #8：LLM 战略层只在训练侧接线

| 环节 | 是否接入 `strategy` | 证据 |
|---|---|---|
| 训练 `train_rwpp.py` | ✅ 接入 | `train_rwpp.py:161` `strategy=_build_strategy(args.use_llm)`；`_build_strategy` 在 `:268-281` |
| 训练 → 环境 | ✅ 透传 | `rwpp_ppo.py:34,49` / `vec_rwpp.py:37,57` → `rwpp_env.py:498,506` |
| 环境内使用 | ⚠️ **只用于 reward shaping** | `rwpp_env.py:685-693` 只把 `_strategy_potential()` 差分成 `strategy` 奖励分量；`directive`（意图/攻击区/撤退阈值）**从未下发给动作**（`decode_action` L320-362 完全无 `directive` 入参） |
| 推理 `play_rwpp_policy.py` | ❌ 0 接入 | 全文 grep `strategy/use_llm/directive` **0 命中**；`RwppDuelEnv(...)` 在 L182 构造时未传 `strategy=` |
| 推理 `eval_rwpp_policy.py` | ❌ 0 接入 | grep 0 命中 |
| 推理 `record_rwpp_replay.py` | ❌ 0 接入 | grep 0 命中 |
| `probe_ironzero25.py` | ❌ 0 接入 | 它自己的 `--strategy` 是另一套 `nearest/lowhp/enemy_centroid` 硬编码启发式（L102-120），与 `StrategyDirective` 无关，**同名不同物** |
| `MacroController` | ❌ 平台 0 接入 | 仅 `training/{vec,evaluate}_macro_micro.py` 与 `test_macro_micro_pipeline.py` 引用；`web/backend`、`rwpp_real_service.py` grep `macro` 0 命中 |

**结论**：`use_llm` 开关从 `index.html:693` → `app.js:7301` → `app.js:7377` → `app.js:842`（`RwppPpoRequest.use_llm`）→ `rwpp_real_service.py:778,935` → `train_rwpp.py:161` 全链路**是通的**，但它只影响训练期的 reward shaping 与日志（`ppo.py:361-363` 发出 `strategy` 事件），**推理侧完全没有战略层**。审计线索"可能只在训练接了、推理没接"——**成立**。

---

## 5. 死代码清单

调用图方法：剥离注释后按大括号匹配切出 253 个函数体，做可达性传播；根集合 = 顶层语句引用 + `bindEvents()` 中**未被 `if (els.nullXxx)` 守卫的**绑定（即把 `submitStartRun`/`stopCurrentRun`/`submitInfer`/`handleChartHover` 视为不可达种子，因为它们的守卫元素为 null）。

### 5.1 完全不可达函数：18 个 / 476 行（占 7997 行的 6.0%）

| 行范围 | 行数 | 函数 | 死因 |
|---|---|---|---|
| L328-337 | 10 | `normalizeResult` | 定义了从未调用（`normalizeReplay` 已覆盖） |
| L1429-1445 | 17 | `refreshRuns` | 只被死函数 `submitStartRun` 调用 |
| L1491-1509 | 19 | `refreshCurrentRun` | 只被死函数 `stopCurrentRun` 调用 |
| L1511-1529 | 19 | `stopCurrentRun` | 绑定到不存在的 `#stop-btn`（app.js:4932） |
| L1531-1570 | 40 | `submitStartRun` | 绑定到不存在的 `#start-form`（app.js:4931） |
| L1897-1909 | 13 | `nearestPoint` | 只被 `handleChartHover` 调用 |
| L1911-1964 | 54 | `handleChartHover` | `bindEvents:5155-5161` 的 4 个元素全 null，注册不上 |
| L2001-2013 | 13 | `loadCheckpoints` | 无人调用（被 `loadFeCheckpoints` L7585 取代） |
| L2502-2533 | 32 | `applyInferStart` | 旧 `/api/infer` 链路残留 |
| L2535-2580 | 46 | `applyInferBatch` | 同上 |
| L2582-2592 | 11 | `pushFrameEvents` | 同上 |
| L2594-2669 | 76 | `submitInfer` | 绑定到不存在的 `#infer-form`（app.js:4933） |
| L2671-2698 | 28 | `startDemoInference` | 只被 `submitInfer` 调用 |
| L2700-2704 | 5 | `startInferPolling` | 只被 `submitInfer` 调用 |
| L2706-2711 | 6 | `stopInferPolling` | 只被 `pollInfer` 族调用 |
| L2713-2758 | 46 | `pollInfer` | 只由 `startInferPolling` 启动 |
| L2760-2788 | 29 | `handleInferFinished` | 只被 `pollInfer` 调用 |
| L3190-3201 | 12 | `mapViewRegion` | 定义了从未调用（`sandbox.cameraRegion` 在 L3204 被非此途径填充） |

> 另：`API_DEFAULT_BASE`（L14）、`API_URL_OVERRIDE`（L15-21）两个常量 + 头部注释 L10-11 承诺的 `?api=` / `window.IRONZERO_API_BASE` 能力，随 `apiBase()` 被改写成常量（L32-34）后**全部失效** —— 且这正是本次未提交 diff 的改动点（`git diff` 显示删掉了 12 行适配逻辑）。

### 5.2 更严重的一层：「活着但静默 no-op」的旧监控大屏

`app.js:1037-1964`（「训练监控」+「图表 canvas」两节，约 **928 行**）里的 31 个函数在调用图上**可达**（`boot()` → `enterDemo()`/`exitDemo()` → `renderRunList()` 等），但每一个都在第一行对 null 元素 early-return：

| 函数 | 首行守卫 | 结果 |
|---|---|---|
| `resetRunData` L1080-1084 | `els.logPanel` / `updateKpis()` / `updateRunHeader()` / `drawCharts()` | 全部空转 |
| `updateRunHeader` L1314 | `if (!els.metaRun) return;` | 空转 |
| `updateKpis` L1332 | `if (!els.kpiUpdate) return;` | 空转 |
| `renderRunList` L1379 | `const box = els.runList;` → L1380 `if (!box) return;` | 空转 |
| `appendLog` L1109 | `const panel = els.logPanel; if (!panel) return;` | 空转（`state.logs` 仍增长，纯内存垃圾） |
| `drawCharts` L1888 | `if (!els.chartWin \|\| !els.chartReturn) return;` | 空转 |
| `renderEventFeed` L2956 | `const box = els.eventFeed;` → `if (!box) return;` | 空转 |

调用链实证：`enterDemo()` L1009-1015 → `renderRunList()` → return；`boot()` L7967-7974 反复调用 `feRender*` 才是真正生效的路径。**这 928 行应在重构中整体删除（连同 `state.runs/runDetail/series/latest/logs` 等 20+ 个只为它服务的 state 字段）。**

### 5.3 CSS 死代码

`style.css` 共 1846 行，**28 个顶层规则块 / 205 行（11.1%）**没有任何 HTML class 或 JS `className`/`classList` 使用：
`.col`(L283)、`.run-list`(L464)、`.kpi-grid`(L551)、`.chart-stack`(L607)、`.chart-wrap`(L617,623)、`.legend`(L631,641)、`.chart-tip`(L651)、`.meta-list`(L671,682)、`.log-panel`(L696)、`.sandbox-grid`(L718)、`.sandbox-side`(L958)、`.scoreboard`(L969)、`.hp-row`(L987)、`.hp-track`(L998)、`.event-feed`(L1012-1026)、`.infer-grid`(L1245)、`.infer-main`(L1253,1261,1266,1275)、`.infer-side`(L1281)、`.fe-train-grid`(L1297)、`.fe-train-main`(L1305,1364,1370)。
外加 `log-empty/win/loss/pos/neg/ok/error/warn/info/unknown/done/kill/shot/terrain/vs/draw` 等 20 个状态类无产物（其中 `win/pos/neg` 由 JS 拼接产生，属正常，已排除）。

---

## 6. 垃圾代码清单

### 6.1 【P0】`apiBase()` 硬编码，适配层被删

```js
// web/frontend/app.js:32-34
function apiBase() {
  return 'http://127.0.0.1:8000';
}
```
- 文件头 L7-11 的注释仍在承诺「默认使用相对路径 `/api/...`，同源部署（FastAPI :8000 / Docker）无需修改代码。调试时可用 `?api=...` 或 `window.IRONZERO_API_BASE`」——**注释与实现直接矛盾**。
- Docker 部署（`web/docker-compose.yml` 映射 8000:8000）时浏览器访问 `http://127.0.0.1:8000` 恰好还能通，但一旦换成 `host.docker.internal`、局域网 IP、反代域名，全部跨域失败；`web/README.md` 里"平台默认只在本机/内网使用"的定位也被写死成 localhost。
- `API_DEFAULT_BASE`(L14) / `API_URL_OVERRIDE`(L15-21) 成为纯垃圾变量。

### 6.2 【P0】`/api/checkpoints` 被改写成不可移植的坏实现

```python
# web/backend/app.py:680-699
@app.get("/api/checkpoints")
def list_checkpoints() -> list[dict[str, Any]]:
    import datetime, os          # ← 模块顶部已 import，重复
    found = []
    # 使用 OS 原生遍历，不依赖 Pathlib 的复杂解析，确保 100% 成功
    for root, dirs, files in os.walk("/app/runs"):
        for file in files:
            if file.startswith("checkpoint") and file.endswith(".pt"):
                path = os.path.join(root, file)
                try:
                    ...
                except:            # ← 裸 except，连 KeyboardInterrupt 都吞
                    continue
    return found
```
问题三重：
1. **路径写死 `/app/runs`**（同样问题见 L42 `CHECKPOINT_ROOTS = [Path("/app/runs")]`）。实测本机 `runs/` 下有 **158 个 `checkpoint*.pt`**，而 `/app/runs` 在 Windows 上不存在 → 接口恒返回 `[]` → 推理页"PPO 模型"下拉永远是"（暂无检测到 Checkpoint）"。
2. 本次未提交 diff **删除了** 原本正确的 `IRONZERO_CHECKPOINT_ROOTS` 环境变量支持（diff: `-CHECKPOINT_ROOTS = [Path(item).resolve() for item in os.environ.get("IRONZERO_CHECKPOINT_ROOTS", ...)...]`），而 `docker-compose.yml:22` 仍在设置 `IRONZERO_CHECKPOINT_ROOTS: /app/runs` —— **配置项成了摆设**。
3. 与 `_find_checkpoints()`（L291-356）功能完全重复：后者仍在文件里，且**没有任何调用点**（grep 确认只有定义行）→ 一个是死函数，一个是坏实现，同一个需求两份代码。
4. 返回字段与 `_find_checkpoints` 不一致（缺 `map` / `update` / `eval`），导致前端 `sortCheckpoints`（L7521-7544）的 `win_rate` / `update` 两种排序**永远退化成按时间排**，`feCheckpointDisplayName`（L7546-7583）的 "胜率 x%" 分支永不显示。这是"排序下拉框看着能用其实无用"的根因。
5. 附带断链：`docker-compose.yml:21` 的 `IRONZERO_GAME_ASSETS` 在 `maps.py:36` **有读**，`IRONZERO_REPO_ROOT`/`IRONZERO_RUNS` 在 app.py 有读，唯独 `IRONZERO_CHECKPOINT_ROOTS` 变成死配置。

### 6.3 【P1】`observations.py` 用自然语言注释冒充代码

```python
# src/ironzero/observations.py:313-317
group_mask = np.zeros(cfg.groups + 1, dtype=bool)
# ... (keep existing group logic)
for unit in own:
    group_mask[int(unit.get("group_id") or 0) % cfg.groups] = True
group_mask[-1] = True
```
```python
# src/ironzero/observations.py:322
# ... rest of the function remains similar
```
这是把"我没写完"直接写进代码。后果：
- 原本 L313-321（diff 中的 `-` 行）有 `group_ids_known` 分支：真机 bridge 不吐 `group_id` 时 `group_mask[:cfg.groups] = True`（全组可指挥）。被删后，真机状态 `group_id` 全缺省 → `int(None or 0) % 4 == 0` → **只有 group 0 可指挥，25 辆坦克里只有 group 0 那批能收到命令**。对应测试 `tests/test_rwpp_25v25.py:174 test_masks_without_group_id_expose_all_groups` **实际失败**（已复跑确认）。
- 同一函数 L304 新增 `funds = ...`、L306 `np.ones(7)`、L308-311 的经济掩码，与 `micro_config.py:103` 的 `action_nvec kind=7` 配套 —— 但这套 7 维 kind 只改了一半：`rwpp_env.py:333` 仍有 `if vector.shape[0] != 5: raise ValueError("action must have 5 heads")`。**7 维动作空间与 5 头校验并存，自相矛盾。** 这正是 11 个 pytest 失败（含 `test_env.py:40` 期望 `(5,)` 实际 `(7,)`、`policy.py:73` 的 `masked_fill` 维度冲突）的共同根因。

### 6.4 【P1】巨型函数（单函数超过 100 行，共 10 个）

| 函数 | 行范围 | 行数 |
|---|---|---|
| `drawSandbox` | L3907-4250 | **344** |
| `bindEvents` | L4926-5208 | **283** |
| `drawLineChart` | L1605-1827 | **223** |
| `drawSandboxSpriteUnit` | L3703-3901 | **199** |
| `buildDemoBattle` | L714-862 | **149** |
| `feDrawPpoCharts` | L6942-7079 | **138** |
| `computeFollowRegion` | L3410-3542 | **133** |
| `renderReplays` | L2033-2165 | **133** |
| `feStartPpo` | L7282-7403 | **122** |
| `feStartInferInner` | L5993-6105 | **113** |

`bindEvents` 283 行内混了 40+ 个 `if (els.xxx)` 绑定，其中至少 4 个守卫元素不存在（§3.1/§5.1），是本文件"堆积式迭代"的最直接证据。

### 6.5 【P2】新增滚轮缩放的数学是错的

```js
// web/frontend/app.js:3936
const customScale = sandbox.customScale || 1.0;
// web/frontend/app.js:3974-3976
cam.scale *= customScale;
cam.ox -= (cam.region.x * (cam.scale - (cam.scale / customScale)));
cam.oy -= (cam.region.y * (cam.scale - (cam.scale / customScale)));
```
L3975 中 `cam.scale` 已在 L3974 被乘过，于是 `cam.scale / customScale` 恰好还原旧值，`cam.scale - cam.scale/customScale` = 新增增量 —— 这个表达式本身没错，但 **`cam.ox` 在 L3211 已按 `region.x * scale` 计算过居中偏移**，这里再减一次属于二次补偿，缩放中心会漂移；且 `drawSandboxSpriteUnit`（L4197）与 `mapCamera` 的 `pad` 都未同步该 scale。**标注「待确认」**：需在浏览器中实测才能判定是"中心漂移"还是"仅视觉跳动"。确定的问题是该改动未同步到 `handleSandboxClick` 的坐标反投影（L4903 用 `view.ox + unit.x * view.scale`，而 `view` 由 L3995 填充 —— 这条链是自洽的，但 `customScale` 一旦 >1，点击命中半径 `bestDistance=22` 是屏幕像素，缩放后需要随之缩放，未处理）。

### 6.6 【P2】其它垃圾

| 位置 | 问题 |
|---|---|
| `app.js:3843-3846` | 新增的空 `else if` 分支，体内只有两行注释「处理其他非建筑、非坦克单位的渲染（可选）」「目前保持原样或让它 fall through」——**空语句块**，纯噪音。 |
| `app.js:3821` | `unit.type === 'tank'` —— **`unit` 对象上没有 `type` 字段**（`normalizeUnit` L270-285 只产 `id/team/group/x/y/hp/maxHp/order/target`；全库 grep `.type =` 无任何单位赋值）。后果：所有非建筑单位（含 tank/builder/其他）都进不了贴图分支，**永远落到 L3849 的 `!rendered` 矢量保底绘制**。这是本次未提交 diff 引入的**贴图回退回归**：`git diff` 显示原本是 `} else if (!isBuilding && spriteCache.tank) {`，被加了 `&& unit.type === 'tank'`。 |
| `app.js:3711` | `unit.unit_type` 同理：`normalizeUnit` 不保留该字段，而 `scripts/record_rwpp_replay.py:94` / `play_rwpp_policy.py:86` / Java `IronZeroBridge.java:304` **都在产出 `unit_type`**。整条"按实体类型差异化渲染"（L3713-3718 的 6 个 is* 判定）在归一化阶段就被砍掉，只剩 `max_hp` 兜底猜测。 |
| `app.js:2231` | `} catch (_) {}` —— 全文件唯一真正的空 catch，且吞的是 `checkHealth()`，会把"健康检查抛异常"静默当成"重连失败"。 |
| `app.js:5147-5152` | `els.logPanel` 为 null，整段 scroll 监听永不注册。 |
| `app.js:5616` / `5710` 等 | 大量 `// xxx is best effort` / `// fall back to cached artifacts` 的 catch 注释（共 ~14 处），属"为了过 lint 而空吞"，可接受但应集中到一个 `swallow()` helper。 |
| `app.js:6059` | `els.feFrameMs` 未定义（见 §3.2），三元恒真/恒假 —— 典型的"加参数忘改另一处"。 |
| `web/backend/app.py:682` | 函数内重复 `import datetime, os`，顶部 L17/L19 已导入。 |
| `web/backend/app.py:697` | `except:` 裸捕获。 |
| `web/backend/maps.py:24` | `WATER_TILESET_HINTS = ("water",)` 定义后**从未使用**（grep 仅命中定义行），实际用的是 L25 `WATER_NAMES`。 |
| `web/backend/maps.py:135-139` | `_read_map_header()` 定义了**从未调用**（grep 仅定义行）。 |
| `web/backend/maps.py:396-418` | `unit_sprites()` 内用 8 个 `if (ASSETS_ROOT / x).is_file()` 的 if-else 三元硬编码贴图路径，同时 L415-418 又"保留原有 tanks/*.png 映射"，两套逻辑并存。 |
| `scripts/record_rwpp_replay.py:34-36` | `RUNTIME = Path(r"E:\IronZeroBuild\rwpp-runtime-1.15")`、`JAVA_HOME = Path(r"E:\jdk25copy")` 硬编码本机路径（对比 `eval_rwpp_policy.py:58-59` 已改成 `argparse` 默认值 + `DEFAULT_*` 常量）。 |
| `scripts/probe_ironzero25.py:30` | `DEFAULT_JAVA_HOME = Path(os.environ.get("IRONZERO_JAVA_HOME", r"E:\jdk25copy"))` —— 默认值硬编码某台机器。 |
| `scripts/rwpp_concurrency_check.py:1` | 文件首行自述「Ad-hoc concurrency validation ... （temporary）」—— 临时脚本进仓库。 |
| `web/frontend/index.html:8,244,246,247,254,255,621,630` 等 | 大量内联 `style="..."`（至少 12 处），与 1846 行 `style.css` 的体系冲突。 |
| `web/frontend/style.css` | 13 处 `!important`。 |

---

## 7. 半成品清单

### 7.1 【P0】`use_llm`：开关传得动、效果看不见

- `index.html:692-695` 有「启用 LLM 战略层」勾选框 → `app.js:7301` 读取 → `app.js:7377` `use_llm: useLlm` 发往后端 → `app.py:121` `RwppPpoRequest.use_llm` → `rwpp_real_service.py:778,935` → `train_rwpp.py:161`。**链路完整。**
- 但效果只到 `rwpp_env.py:688-692` 的 `strategy` **reward 分量** + `info["strategy"]` 透传 + `ppo.py:361-363` 的 `strategy` 事件。
- `StrategicDirective` 的 `intent / attack_region / rally_point / retreat_hp_threshold / priority_target`（`directive.py:78-85`）**从不参与动作选择**：`decode_action`（`rwpp_env.py:320-362`）签名里根本没有 directive。
- 推理侧（`play_rwpp_policy.py` / `eval_rwpp_policy.py` / `record_rwpp_replay.py`）grep `strategy|use_llm|directive` = **0 命中**，`RwppDuelEnv(...)` 构造（`play_rwpp_policy.py:182`）不传 `strategy=`。
- UI 后果：`feRenderPpoStrategy()`（`app.js:7864-7925`）在推理页/回放页**完全不出现**（只挂在训练页 `index.html:569-575`），故用户勾了 LLM、训练日志里也只有一行 `using LLMStrategy (llm=True)`。
- 另：`LLMStrategy.propose` 的节流（`llm_strategy.py:310-315`）在 `_refresh_strategy()`（`rwpp_env.py:849-858`）每步都调用一次的前提下，被节流 5s 缓存的 directive 会**反复参与 potential 差分**，`strategy_component` 在缓存窗口内恒为 0 —— 语义上等价于"strategic shaping 每 5 秒才动一次"，与"每 tick 观察"的直觉不符。标注**待确认**（需确认 `strategy_coef` 调度意图）。

### 7.2 【P1】推理页 `policy` / `ticks` / `frame_ms` 字段名与后端对不齐

前端（`app.js:6060-6068`）发出：
```js
{ ticks, policy: 'model'|'scripted', frame_ms, map_name, map_path, map_width, map_height, checkpoint? }
```
后端 `RwppRecordRequest`（`app.py:89-102`）字段名一致，转发给 `rwpp_real_service` 的 `/infer`（`app.py:787-797`）。但：
- `app.py:790-791` 在 `/infer` 里 `payload.pop("episodes")` / `pop("concurrency")`，**`/train`（app.py:800`）却原样转发** —— 同一个 pydantic 模型被两个语义不同的端点评分复用，靠 pop 打补丁。
- `/api/rwpp/record` 的 `policy` 支持 `scripted|none|model|ppo`（app.py:91），而 `rwpp_real_service._start()`（L949-960）只读 `policy`/`checkpoint`，`play_rwpp_policy.py` **根本没有 `--policy` 参数**（`build_parser` L111-129 无此选项）→ 前端传 `policy='model'` 时，服务端怎么处理 **待确认**（需读 `rwpp_real_service.py:230-290` 的 `_run_player` 参数拼装；本节已确认 `play_rwpp_policy` 侧无对应 CLI）。
- `frame_ms` 前端恒发 2（§3.2 的 bug），后端默认也是 2（app.py:95）→ 用户改不动。

### 7.3 【P1】`RwppPpoRequest` 有两个字段后端完全不落地

- `app.py:122-124` 定义了 `units_per_team` / `init_checkpoint` / `init_mode`，`app.py:842` 原样 `request.model_dump()` 转发。
- `rwpp_real_service.py:825-841` 确实读了 `init_checkpoint` / `init_mode`（有 warmstart 回退逻辑），**但 `init_mode` 从未写回 job 记录**（L899 的 `"use_llm": use_llm` 附近无 `init_mode`），前端 `feRestorePpoJob` 也就无法恢复该 UI 状态（`app.js:7462` 只恢复 `use_llm`）。属**只传了一半**。

### 7.4 【P1】`/api/rwpp/stop_all` 后端有，UI 无入口

`app.py:860-862` 完整实现，`web/README.md` 未文档化，前端 `Api.rwppStopAll`（`app.js:112`）**0 调用**。并发 4 路推理卡死时用户只能逐个点停止（`feStopPpo` / `feStopTrain` 各自独立），**没有"全部停止"** 的逃生口。

### 7.5 【P2】favicon 三处断链

- `index.html` 无 `<link rel="icon">`（grep 0 命中）
- `web/frontend/favicon.svg` **文件不存在**（目录只有 3 个文件）
- `app.py:869-874` 路由存在但 `icon.is_file()` 为假 → 恒 404
- 反倒 `site/favicon.svg`（323 B）是有的 → 静态站比主产品更完整

### 7.6 【P2】`RwppRecordRequest` 的 `map_width/map_height` 默认值互相矛盾

- `app.py:98-99` `2600.0 / 2600.0`（旧 Lake 图尺寸）
- `app.py:115-116` `RwppPpoRequest` `2700.0 / 2500.0`（IronZero25 尺寸）
- `scripts/play_rwpp_policy.py:118-119` 又默认 `2700.0 / 2500.0`
- `scripts/record_rwpp_replay.py:55-56` 默认 `2600.0 / 2600.0`
- `play_rwpp_policy.py:270-271` 写 replay meta 时用的是 `micro.map_width/height` 而非 `args.map_width/height` —— **前端沙盘按 meta 渲染，但后端按 args 起游戏，两个尺寸可能对不上**，属**待确认**（需实跑一局验证）。前端 `app.js:6066-6067` 用 `mapItem.width_px` 传，链路能自洽，但默认值不一致说明拷贝粘贴失控。

### 7.7 【P2】`sandboxStrategyLabel()` 分支几乎不可达

`app.js:7771-7780`：`sandbox.policy === 'model'` 只有在 `app.js:5678`（`feApplyWatchTask`）赋值；而 `submitInfer`（另一条赋值路径）已死。`sandbox.checkpoint === 'rwpp-live'` 只在 L5677 赋值。因此 HUD 上的「模型策略 · xxx」标签只在"从任务恢复观看"这一条路径下出现，正常推理启动（`feStartInferInner`）走 L6048 分支但**只设 `simMode`，不设 `policy`** → 标签恒为'脚本策略 · '或空。**功能上"策略名显示"是残的。**

### 7.8 【P2】`site/` 判定：**未废弃，独立存活**（架构冗余判断）

结论：`site/` **不是**被废弃的上一代产物，它是**面向 GitHub Pages 的项目主页**，与 `web/frontend/`（运行时监控大屏）**职责不重叠**：

| 证据 | 内容 |
|---|---|
| `.github/workflows/pages.yml:8` | 触发路径 `- "site/**"` |
| `.github/workflows/pages.yml:37` | `path: site` → 部署站点根 |
| `site/README.md:6-12` | 明确写明 `python -m http.server 8000 --directory site` 的预览方式 |
| `README.md:76-82` | 根 README 把 `site/index.html` 列为「IronZero Lab 静态展示页」并给预览命令 |
| `LICENSING.md:19-20` | 把 `site/` 与 `site/README.md` 分别列入 Apache-2.0 / CC-BY-4.0 清单 |
| `THIRD_PARTY_NOTICES.md:64` | 单独声明 `site/index.html` 引用 Steam CDN 图片的版权归属 |
| `命名方案.md:38,166,168,169,200` | 反复引用 `site/index.html` 作为品牌落地文件 |

代码层面也**不重复**：
- `site/main.js` 全 44 行，只做 nav 折叠 + 年份 + hover 视差（`site/main.js:12-43`），**零网络请求**（无 `fetch`）。
- 与 `web/frontend` 的 CSS class 交集仅 8 个（`brand/brand-mark/brand-sub/compact/done/enemy/topbar/unknown`），均为通用命名，非复用。
- `site/index.html` 21668 B / 231 行，是营销落地页（Mission/Training Loop/Architecture/Roadmap/Open Source 五节）。

**真正的架构冗余不在 `site/` vs `web/frontend/`，而在 `web/frontend` 内部**：新旧两代大屏（`/api/runs` 旧监控 vs `/api/rwpp/*` 实机新大屏）在同一文件里并存，旧的 DOM 已删、JS 未删（§5.2）。**修复建议：保留 `site/`，清理 `web/frontend` 内的旧代代码。**

---

## 8. 修复优先级建议

### P0（阻断，必须先修；3 条）

1. **恢复 API 基址适配层** —— `web/frontend/app.js:32-34` 回滚为原 `apiBase()`（用 `window.IRONZERO_API_BASE` → `API_URL_OVERRIDE` → 同源判断 → `API_DEFAULT_BASE`）。这一条不修，任何非 localhost 部署都是白屏 + 满屏 CORS 报错。
2. **修 `/api/checkpoints`** —— `web/backend/app.py:42` 与 `:685` 的 `/app/runs` 恢复为 `CHECKPOINT_ROOTS`/`os.walk` 遍历多个 root（`IRONZERO_CHECKPOINT_ROOTS` + `REPO_ROOT/"runs"`），删掉裸 `except:`，删掉重复 import；同时删掉已死的 `_find_checkpoints()`（L291-356）或反之 —— **二选一，不要两份**。不修则 PPO 模型推理在本机 100% 不可用。
3. **修 `observations.py` 的 group_mask 与 action_nvec 矛盾** —— `src/ironzero/observations.py:313-317` 恢复 `group_ids_known` 分支（真机无 `group_id` → 全组开放）；`rwpp_env.py:333` 的 `!= 5` 校验与 `policy.py:38` 的 `nvec=(5,4,8,8,9)` 默认值需与 `micro_config.py:103` 的 `kind=7` 统一。这一条修完，11 个 pytest 失败中至少 8 个会消失。

### P1（功能残缺，用户可感知；5 条）

4. **把 `fe-frame-ms` 接进 `cacheEls()`** —— `app.js:394-439` 补 `'fe-frame-ms'`（对应 `els.feFrameMs`），否则新增的「推理速度」下拉框永远无效；同时把 `app.js:6059` 的 `parseInt` 补 `clampInt(...,0,16,2)` 防空值。
5. **修 `unit.type` 贴图回归** —— `app.js:3821` 改回 `!isBuilding && spriteCache.tank`（去掉 `&& unit.type === 'tank'`），或在 `normalizeUnit`（L270-285）里补 `unit_type`/`is_building`/`is_builder`/`is_factory` 四个字段透传。当前所有单位退化到矢量绘制。
6. **删掉旧代监控大屏** —— `app.js:1037-1964`（约 928 行）+ 对应 `cacheEls` 的 22+ 个悬空 id + `state.runs/runDetail/series/latest/logs/eventCursor/...` + `style.css` 的 205 行孤儿样式 + 后端 `/api/runs` 5 条路由（若确定不再维护）。若暂时想保留后端，至少给前端这 928 行加一个 `if (!document.getElementById('run-list')) return;` 的总开关，避免每次 `enterDemo()` 白跑一遍。
7. **LLM 战略层：要么补全、要么显式降级** —— 若短期不做推理侧战略层：(a) 把 `index.html:692-696` 的开关文案改成「仅训练期 reward shaping 生效，不影响推理」；(b) 在 `RwppDuelEnv.__init__` 给 `strategy` 加 `assert`/警告，防止后续误以为已接通。若要做：`decode_action` 需接 `directive`（`attack_region` → `attack_move` 坐标，`retreat_hp_threshold` → 强制 `hold`），`play_rwpp_policy.py:182` 需透传 `strategy=`。
8. **清理 `/api/rwpp` 的 pop 补丁** —— `app.py:787-797` 里 `payload.pop("episodes"/"concurrency")` 的做法应改成独立 pydantic 模型（`RwppInferRequest` / `RwppTrainRequest`），否则后续加字段必踩坑；同时给 `/api/rwpp/stop_all` 补一个 UI 按钮（并发场景的逃生口）。

### P2（整洁度，可排期；5 条）

9. **拆巨型函数** —— 优先 `bindEvents`(283 行) → 按 Tab 拆成 `bindInferEvents/bindTrainEvents/bindPpoEvents/bindSandboxEvents`；其次 `drawSandbox`(344 行)。
10. **统一 `map_width/height` 默认值** —— `app.py:98,115` / `record_rwpp_replay.py:55` / `play_rwpp_policy.py:118` 四处不一致，建议抽到 `maps.py` 的 `DEFAULT_ARENA_W/H` 常量。
11. **`site/` 保留不动**（§7.8），但可在 `README.md:76` 补一句"`site/` 是 GitHub Pages 展示页，与 `web/frontend/` 运行时大屏无关"，避免下一轮审计再误判。
12. **清理临时脚本/硬编码路径** —— `scripts/rwpp_concurrency_check.py`（自述 temporary）、`record_rwpp_replay.py:34-36` / `probe_ironzero25.py:30` 的 `E:\` 路径 → 统一走 `argparse` 默认 + 环境变量（照抄 `eval_rwpp_policy.py:58-59` 的写法）。
13. **补 favicon** —— 把 `site/favicon.svg` 复制到 `web/frontend/favicon.svg` 并在 `index.html` 加 `<link rel="icon">`。

### 附：本次审计的交叉验证结果

| 已知事实 | 实测 | 结论 |
|---|---|---|
| 工作区有未提交改动 `app.js +148, index.html +37, app.py +25` | `git diff --stat` 完全一致（9 文件，194+/70-） | ✅ 一致 |
| `pytest` = 11 failed, 190 passed | 复跑 `python -m pytest -q`：**11 failed, 190 passed in 19.51s** | ✅ 完全一致 |
| 11 个失败的根因 | 全部指向 `micro_config.py:77 enable_economy=False→True` 与 `:103 kind 5→7` 这一处未完成的改动 | 见 §6.3 |
| 本次审计范围内测试 | `pytest tests/test_strategy.py tests/test_rwpp_scripts.py` → **42 passed** | ✅ 策略层与推理脚本的自测是健康的，问题都在"接线"和"前端" |

> 审计过程中**未修改任何源码**。仅在 `audit/` 下新增本报告；期间误生成的一个临时文件 `audit/dead.json` 已删除。
