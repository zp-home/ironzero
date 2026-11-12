# `web/frontend/js/` — 前端 ES Modules

`web/frontend/app.js`（单文件 8082 行）拆分而来的 ES Module 包。

**拆分原则：只搬结构、不改行为（behavior-preserving refactor）。** 所有函数体逐字剪切粘贴，
唯一改动是行首增加 `export`、文件头增加 `import`。没有修改任何数值、字符串文案、正则、逻辑分支。

> `web/frontend/app.js` **保留为回滚备份**，`index.html` 已不再引用它。
> 接线为 `<script type="module" src="/static/js/main.js">`（含 `onerror` 兜底到 `./js/main.js`）。

---

## 1. 入口与加载链路

```
index.html
  └── <script type="module" src="/static/js/main.js">
        └── js/main.js   →  boot()  +  DOMContentLoaded 调度
```

`main.js` 是唯一入口，只含 `boot()` 与末尾的调度块：

```js
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { boot(); });
} else {
  boot();
}
```

`boot()` 开头先跑三行 `state.fe` 初始化（见 §4），随后与原 `boot()` 完全一致：
`cacheEls() → bindEvents() → startClock() → switchTab(tabFromHash()) → …`

部署路径：FastAPI 用 `app.mount("/static", StaticFiles(directory=FRONTEND_DIR))`，
所以线上 URL 是 `/static/js/main.js`；`js/` 目录必须位于 `web/frontend/` 下，两者才能同时定位到文件。

---

## 2. 模块清单

| 模块 | 行数 | 职责 |
| --- | ---: | --- |
| `main.js` | 77 | 启动入口 + `DOMContentLoaded` 调度 + `state.fe` 分段初始化编排 |
| `api.js` | 111 | API 适配层：`API_DEFAULT_BASE`/`API_URL_OVERRIDE`/`ApiError`/`apiBase`/`api()`/`encodePathId`/`Api` 门面 |
| `util.js` | 125 | 纯工具：`pickVal`/`num`/`numOrNull`/`clamp`/`clampInt`/`fmtNum`/`fmtPct`/`pad2`/`clockString`/`fmtDateTime`/`parseBeijingDate`/`fmtShortTime`/`basename`/`safeStringify`/`mulberry32`/`BASE_FPS` |
| `normalize.js` | 156 | 后端字段归一化：`normalizeMap/Team/Unit/Event/Frame/Result/Replay`、`STATUS_META`/`statusMeta`/`resultMeta` |
| `state.js` | 164 | **共享可变状态**：`els`、`state`、`cacheEls()`、`createSandboxState()` |
| `demo.js` | 363 | 演示数据：`DEMO`、`buildDemoEvents`/`demoDetailFromEvents`/`buildDemoLogTail`/`buildDemoBattle`/`initDemoData`/`demoReplayPayload` |
| `connection.js` | 126 | 连接状态与降级：`clockTimer`/`startClock`/`renderConn`/`showDemoChips`/`checkHealth`/`retryConnection`/`enterDemo`/`exitDemo` |
| `train.js` | 346 | 训练监控 · 事件流：指标点、日志面板、`applyEvent`、KPI、详情 |
| `trainRuns.js` | 207 | 训练监控 · run 列表：`renderRunList`/`refreshRuns`/`selectRun`/`start`/`stop` |
| `charts.js` | 396 | 图表（canvas 自绘）：`niceTicks`/`drawLineChart`/`hexToRgba`/`chartSeriesConfigs`/`renderLegend`/`drawCharts`/`nearestPoint`/`handleChartHover` |
| `replays.js` | 209 | Checkpoint 列表 / 回放库：分页、折叠、`renderReplays`/`loadReplays` |
| `rwpp.js` | 329 | 真机 RWPP 录制：状态轮询、录制任务、`openReplay`/`removeReplay` |
| `sandbox.js` | 303 | 对局沙盘 · 推理任务：`submitInfer`、帧流分批接收、`pollInfer` |
| `playback.js` | 339 | 对局沙盘 · 播放：`resetSandbox`、帧步进/倍速、`interpolateUnits`、事件特效、`updatePlayerUi` |
| `recorder.js` | 114 | 对局沙盘 · 录制：`captureStream` + `MediaRecorder`，`start/stop/cancel/finalizeRecording` |
| `terrain.js` | 329 | 实机地图 · 目录与地形预渲染：`loadMaps`/`renderTerrain`/`buildTerrainEntry`/`mapCrop`/`mapCamera` |
| `view.js` | 330 | 实机地图 · 相机与视图：`computeFollowRegion`/`selectMap`/`applyViewMode`/`syncMapSelect`/`syncMapToggles` |
| `sprites.js` | 382 | 单位贴图：加载/缓存/帧动画/朝向角/`drawSandboxSpriteUnit` |
| `render.js` | 450 | 沙盘渲染主流程：`drawSandbox`、`fogLayer`、`drawFogLayer`、边框、空态 |
| `hud.js` | 244 | 沙盘渲染叠加层：`drawSandboxHud`/`drawSandboxUnit`/`drawFrameEvents`/`drawShotLine`/`drawKillBurst` |
| `dashboard.js` | 151 | Tab 切换 / 提示条 / 动画循环 / 主轮询：`switchTab`/`tabFromHash`/`toast`/`handleSandboxClick`/`animationLoop`/`tickMain` |
| `bindings.js` | 309 | DOM 事件绑定：`bindEvents()`（原 283 行单函数） |
| `control.js` | 247 | 指挥中心公共件：`FE_*`/`DASH_COLORS`、`dashSetup`/`dashGrid`/`drawDashLine`/`drawDashBars`、`feSetText`/`feSetStatusPill` |
| `feInfer.js` | 269 | 实机推理任务对象：任务创建、帧摄入、标记完成/失败、观看同步 |
| `feInferUi.js` | 266 | 实机推理展示层：FPS、图表、任务卡、KPI、状态灯 |
| `feInferRun.js` | 281 | 实机推理调度：启停、轮询、收尾、任务恢复 |
| `feTrain.js` | 377 | 实机训练（多线程数据采集）：启停、轮询、KPI、日志、产物 |
| `feMaps.js` | 112 | 地图下拉控制：`feMapList`/`feInferMapList`/`feDefaultMap`/`renderFeMapControls` |
| `ppo.js` | 368 | PPO 常量与展示层：`PPO_MAP`、事件解析、日志/摘要/KPI、`feApplyTrainMode` |
| `ppoCharts.js` | 147 | PPO 六联图：`feDrawPpoCharts` |
| `ppoRun.js` | 286 | PPO 调度：启停、轮询、任务恢复 |
| `checkpoint.js` | 224 | 模型推理 · checkpoint 选择：加载、排序、显示名、下拉渲染 |
| `model.js` | 246 | 策略 UI 同步 / 奖励分解 / 战略指挥面板 |

合计 **33 个文件 / 8383 行**。

### 与规范 3.5 节模块名的对应

`docs/目录结构与模块边界.md` 3.5 节按 `app.js` **7497 行**估算模块大小。实际文件为 **8082 行**
（多出 585 行），若严格照搬该节模块划分，`train.js`/`sandbox.js`/`terrain.js`/`render.js`/`control.js`/`ppo.js`
会显著超过 400 行验收线。因此按**同一内聚原则追加细分**，均为纯剪切粘贴、不跨节搬迁：

| 规范模块 | 实际拆分 |
| --- | --- |
| `train.js`（训练监控整段） | → `train.js` + `trainRuns.js` |
| `sandbox.js`（沙盘推理/帧流/播放/录制） | → `sandbox.js` + `playback.js` + `recorder.js` |
| `terrain.js`（目录/地形/相机/贴图） | → `terrain.js` + `view.js` + `sprites.js` |
| `render.js`（沙盘渲染） | → `render.js` + `hud.js` |
| `dashboard.js`（Tab/事件绑定/轮询/启动） | → `dashboard.js` + `bindings.js` |
| `control.js`（指挥中心） | → `control.js` + `feInfer.js` + `feInferUi.js` + `feInferRun.js` + `feTrain.js` + `feMaps.js` |
| `ppo.js`（实机 PPO 训练） | → `ppo.js` + `ppoCharts.js` + `ppoRun.js` |
| `model.js`（checkpoint/评估/LLM/奖励/战略） | → `model.js` + `checkpoint.js` |

命名与职责边界未变，只是把一个「段」拆成若干个 400 行以内的文件。

### 超过 400 行的文件

只有 `render.js`（**450 行**）。原因：`drawSandbox()` 单个函数就有 **340 行**，
是不可切分的整体（一个 canvas 绘制主循环，内部共享大量局部变量）。强行拆会改变函数体，
违反「不改写函数体」铁律。**未超过 600 行上限。**

`charts.js`（396）与 `control.js`（247）均在 400 行以内，无需例外说明。

---

## 3. 模块依赖图（谁 import 谁）

```
                        ┌──────────────────────────────┐
   leaf（无依赖）        │  api.js   util.js   state.js │
                        └──────────────┬───────────────┘
                                       │ state/util 被几乎所有人 import
  normalize.js ──► util
  demo.js      ──► normalize, util
  control.js   ──► state
  charts.js    ──► state, util
  connection.js──► api, dashboard, demo, playback, replays, state, terrain, train, trainRuns, util
  train.js     ──► charts, normalize, state, util
  trainRuns.js ──► api, charts, dashboard, demo, normalize, state, train, util
  recorder.js  ──► dashboard, state
  replays.js   ──► api, demo, normalize, rwpp, state, util
  rwpp.js      ──► api, connection, dashboard, demo, feInferUi, normalize, playback, replays,
                   state, terrain, util, view
  sandbox.js   ──► api, dashboard, demo, normalize, playback, replays, rwpp, state, terrain, util, view
  playback.js  ──► demo, normalize, recorder, render, rwpp, state, terrain, util, view
  terrain.js   ──► api, feMaps, sprites, state, util, view
  view.js      ──► api, dashboard, render, sprites, state, terrain, util
  sprites.js   ──► api, state, terrain, util
  render.js    ──► hud, playback, sprites, state, terrain, util, view
  hud.js       ──► model, playback, terrain, util
  dashboard.js ──► checkpoint, connection, feInferRun, feInferUi, feTrain, playback, ppo,
                   ppoRun, recorder, render, replays, rwpp, state, terrain
  feInfer.js   ──► api, dashboard, feInferUi, normalize, playback, render, state, util
  feInferUi.js ──► control, dashboard, feInfer, normalize, state, util
  feInferRun.js──► api, dashboard, feInfer, feInferUi, feMaps, normalize, playback, replays,
                   rwpp, state, util, view
  feTrain.js   ──► api, control, dashboard, feInferUi, normalize, replays, rwpp, state, util
  feMaps.js    ──► control, feInferUi, state, view
  ppo.js       ──► control, feInferUi, feTrain, model, ppoCharts, state, util
  ppoCharts.js ──► control, ppo, state, util
  ppoRun.js    ──► api, dashboard, feInferUi, feMaps, ppo, rwpp, state, util
  checkpoint.js──► api, demo, model, state, util
  model.js     ──► checkpoint, control, ppo, state, util
  bindings.js  ──► charts, checkpoint, connection, dashboard, feInferRun, feInferUi, feTrain,
                   model, playback, ppo, ppoRun, recorder, render, replays, rwpp, sandbox,
                   state, terrain, trainRuns, util, view
  main.js      ──► bindings, checkpoint, connection, control, dashboard, feInferRun, feInferUi,
                   feTrain, model, playback, ppo, ppoCharts, ppoRun, replays, rwpp, state, terrain
```

### 扇入最高的模块（改这些要格外小心）

| 模块 | 被 import 次数 |
| --- | ---: |
| `state.js` | 27 |
| `util.js` | 25 |
| `api.js` | 13 |
| `dashboard.js` | 13 |
| `normalize.js` | 11 |
| `terrain.js` | 11 |

`state.js` / `util.js` / `api.js` 是叶子模块（无出边），不存在循环风险，是安全的地基。

---

## 4. 共享可变状态

`state.js` 是全站单例：

```js
export const els = {};      // DOM 缓存表，cacheEls() 填充
export const state = { … }; // 全局可变状态
export function createSandboxState() { … }
export function cacheEls() { … }
```

其它模块一律：

```js
import { els, state } from './state.js';
```

**铁律：绝不重新赋值 `state` / `els` 本身，只能改属性。** 因为 ESM 的 import 绑定是只读活绑定，
重新赋值 `state`（`state = {...}`）会直接抛 `TypeError: Assignment to constant variable`，
且其它模块拿到的会是旧引用。原代码本来就只改属性，拆分后语义不变。

---

## 5. 循环依赖如何处理

### 5.1 现状

依赖图存在**一个大的强连通分量（SCC）**，包含 22 个模块：

```
{checkpoint, connection, dashboard, feInfer, feInferRun, feInferUi, feMaps, feTrain,
 hud, model, playback, ppo, ppoCharts, ppoRun, recorder, render, replays, rwpp,
 sprites, terrain, trainRuns, view}
```

最典型的小回路是 `render → hud → model → ppo → ppoCharts → ppo → …` 与
`terrain ⇄ view`、`model ⇄ checkpoint`、`ppo ⇄ model`。

**这不是拆分引入的缺陷，而是原单文件代码的真实耦合结构**——在一个文件里这些互相调用
天然"无环"（都只是同一作用域内的函数）。拆开后函数调用关系被投影到模块图上，回路就显形了。
按照规范 3.5 节，**ES Module 循环依赖是允许的：函数声明会提升，运行时调用没问题。**

### 5.2 关键约束（这是真正会炸的地方）

> **顶层立即执行的代码，不能依赖循环依赖中尚未初始化的 `const`/`let`/`class`。**

原因：ESM 求值时，import 声明被提升 → 依赖模块先求值 → 本模块再求值。
若 A 的顶层代码在求值期读取 `B.SOME_CONST`，而 B 又（经由回路）依赖 A，
则 B 可能尚未执行到那条 `const` 声明，此时该绑定处于 **TDZ**，访问会抛
`ReferenceError: Cannot access 'SOME_CONST' before initialization`。

`function` 声明不受影响：它们在模块求值**之前**就已完成实例化（hoisting），
所以顶层代码调用循环依赖回路里的函数是安全的。

### 5.3 本项目的处理

原 `app.js` 有 **4 条顶层 `Object.assign(state.fe, …)` 语句**（分别位于控制中心、
PPO、模型三段），它们是**真正会在模块求值期执行的代码**，且读写 `state.fe`。
`state.fe.ppo` 由 `control` 段先建立、后被 `ppo` 段扩展——如果原样留在模块顶层，
22 节点回路的求值顺序不保证与原文件一致，就会出现「`ppo` 段先跑、`state.fe.ppo` 还不存在」。

按规范 3.5 节「**如果遇到，把初始化逻辑挪进 `boot()` 或函数里**」，处理如下：

| 原顶层语句 | 现位置 | 调用点 |
| --- | --- | --- |
| 第 1 段 `Object.assign(state.fe, { concurrency, inferTasks, … })` | `control.js` → `initFeControlState()` | `boot()` |
| 第 2 段 `Object.assign(state.fe, { trainMode, ppo: { … } })` | `ppo.js` → `initFePpoState()` | `boot()` |
| 第 3 段 `Object.assign(state.fe, { checkpointList, … })` | `model.js` → `initFeModelState()` | `boot()` |
| 第 4 段 `Object.assign(state.fe.ppo, { evalEvery, … })` | `model.js` → `initFeModelState()` | `boot()` |

`main.js` 的 `boot()` 开头**按原文件顺序**调用：

```js
export async function boot() {
  initFeControlState();   // 原顺序 1
  initFePpoState();       // 原顺序 2  ← 建立 state.fe.ppo
  initFeModelState();     // 原顺序 3、4 ← 扩展 state.fe 与 state.fe.ppo
  cacheEls();
  …
}
```

`boot()` 由 `DOMContentLoaded` 触发，此时**全部 33 个模块都已求值完毕**，
不存在任何 TDZ 风险；且三行调用保持了原文件中的先后次序（第 3、4 段合并进同一个
`initFeModelState()`，内部顺序不变），`state.fe.ppo` 在被扩展前必定已由 `initFePpoState()` 建好。
原语句内容**逐字保留**，只是外面包了一层函数。

### 5.4 已验证

对全部 33 个模块做了顶层语句的静态审计（提取每个模块「求值期」会真正执行的语句，
解析其中的标识符到「来自哪个模块、是 function 还是 const」），结论：

- 33 个模块里，**没有任何一处**模块求值期的跨模块引用指向 `const`/`let`/`class`；
- 所有跨模块引用都指向 `function` 声明（已提升，安全）；
- 模块顶层唯一会"执行"的表达式都是各自文件内的**字面量/自引用**初始化
  （`const DASH_COLORS = {…}`、`const Api = {…}`、`new Map()` 等），不跨模块取值。

再加上 Node DOM stub 的完整 `boot()` 冒烟通过，可确认**未引入顶层执行顺序变化**。

### 5.5 如果以后要消除回路

回路本身不影响正确性，只是让依赖图难读。若要治理（非本次范围），推荐方向：
把 `state.js` 里 `state.fe` 的形状定义抽成 `feStateSchema.js` 供各方只读引用，
或把跨段共用的纯数据（如 `PPO_MAP`、`DASH_COLORS`）下沉到独立常量模块。

---

## 6. 验证方式

```powershell
cd E:\qwq\强化学习笔记\铁锈战争

# 1) 语法（逐文件，ESM 语义）
Get-ChildItem web\frontend\js\*.js | ForEach-Object {
  node --input-type=module --check < $_.FullName
}

# 2) 行数（应全部 ≤400，仅 render.js 450）
Get-ChildItem web\frontend\js\*.js | ForEach-Object {
  "{0,-14} {1}" -f $_.Name, (Get-Content $_.FullName | Measure-Object -Line).Lines
}

# 3) 本地起静态服务看页面
python -m http.server 8000 --directory web/frontend
#   浏览器开 http://127.0.0.1:8000/  （http.server 无 /static mount，
#   此时应走 onerror 兜底加载 ./js/main.js）
```

## 7. 与其他子树的边界

- **不改** `web/backend/`（含 `app/` 包与 `unit_sprites.py`）。
- **不改** `tests/`。
- `/api/maps` 响应里的 `unitSprites` 键由后端「沙盘贴图契约」工作流提供；
  前端 `state.maps.unitSprites`（`state.js`）本来就在消费它，本次拆分**未改动**该逻辑。
