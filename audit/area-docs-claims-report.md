# IronZero 文档承诺 ↔ 代码现实 对账报告

> 审计员：audit-docs-claims
> 审计时间：2026-11-11（Asia/Shanghai）
> 审计范围：根目录文档（`README.md`、`训练推理与演示架构.md`、`铁锈战争强化学习系统架构.md`、`命名方案.md`、`LICENSING.md`、`SECURITY.md`、`THIRD_PARTY_NOTICES.md`）、`docs/` 全部、`configs/`、`schemas/`、`tools/`、`scripts/*.ps1` 与 `scripts/*.py`、`.github/`、`web/Dockerfile`+`web/docker-compose.yml`、`vendor/rwpp/pinned-versions.json`、`runs/` 产物
> 审计方式：只读。文档声称逐条 grep / 读码 / 读产物核对；未启动游戏、未跑真机训练（避免占用 5140–5200 端口）
> 代码基线（重要）：
> - **HEAD = `3ecd629`**（`git log`），在导出副本上跑 `python -m unittest discover -s tests` → **201 tests OK**（复现命令见 §5）
> - **工作树 = HEAD + 9 个已修改文件 + 3 个未跟踪文件**，同一命令 → **201 tests，10 FAIL + 1 ERROR**；且 `load_policy` 无法加载仓库里任何既有 checkpoint（详见 §2.1 R-00）
> - 本报告所有"已实现"判定以 **HEAD** 为准，凡标注「工作树回归」处指未提交改动导致的破坏

---

## 1. 一句话结论

**文档里"打法层"（协议、掩码、PPO、真机桥、回放、平台、25v25 评估）确实有代码且大部分能用；但"战略层"（课程学习、自博弈/模型晋级/Elo/ModelPool、lan_eval、realtime 推理、Policy Service、MatchSpec、Episode 状态机、迷雾审计、CI 门禁）几乎全部是"文档架构图 + 一个空壳文件"，其中 `curriculum.py` 是纯占位死代码；另有 3 处可执行的事实性虚标（102 项测试、73 张地图、经济/建造"已实现"），以及 1 处新引入的未提交回归把现有 checkpoint 全部打成不可加载。**

最严重的 3 条虚标（详见 §2）：
1. **R-01 课程学习**：`README.md:24` 与 `铁锈战争强化学习系统架构.md:659-668` 声称支持 7 阶段课程；全仓唯一"实现"是未跟踪的 `src/ironzero/training/curriculum.py`（17 行，两个分支返回同一个 config，从未被 import）。**不存在**。
2. **R-02 自博弈 / 模型晋级 / Elo / ModelPool**：`README.md:229` 标为待办但 `训练推理与演示架构.md:564-575`、`命名方案.md:91` 把"自博弈血统/晋级门槛自动判定"写成系统能力；`grep -i 'self_play|ModelPool|elo|league|promote'` 在 `src/ tools/ scripts/ web/ tests/ configs/` **零命中**。**不存在**。
3. **R-03 经济与建造系统**：`docs/经济与建筑系统方案.md` 全篇 + README 路线把"采矿/工厂/生产队列/二本升级"当闭环能力，实际只在 **mock 引擎**里成立；真机 Java `applyStep`（`IronZeroBridge.java:619-642`）**只认 5 类动作**，`build_extractor/build_structure/produce_unit/upgrade_building` 发出后被记为 rejected。真机 `runs/econ_*/` 的 `reward_components` 全为 `{}`，证明经济奖励从未在真机生效。

---

## 2. 文档声称 ↔ 代码现实 主表格

> 现实列：**已实现** = 有可运行代码且被主链路调用/有测试；**部分** = 有代码但只覆盖 mock / 只在某一条路径 / 只落 schema；**不存在** = 无实现或仅文档；**待确认** = 需真机/外部运行时才能判定
> 严重度：阻断 / 高 / 中 / 低

### 2.1 训练与算法（最高风险区）

| # | 声称的功能 | 文档出处(file:line) | 验证方式（grep 到哪些文件） | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-00 | （非文档声称，但影响全部"已验证"结论）工作树可加载既有 checkpoint、201 测试全绿 | `README.md:14`（"通过 102 项测试"）、`docs/闭环差距分析.md:368`（要求单测全绿） | 工作树 vs HEAD 各跑一次 `unittest discover -s tests`；`load_policy('runs/ironzero25_ft1/checkpoint.pt')` | **回归**：HEAD 201 OK；工作树 10 FAIL+1 ERROR（`test_env`/`test_rwpp_25v25`/`test_training_pipeline`/`test_mock_engine`），`load_policy` 报 `size mismatch heads.0.weight [5,256] vs [7,256]`。根因：未提交改动把 `micro_config.action_nvec` 的 kind 头 5→7（`src/ironzero/micro_config.py:103`），而 `observations.py` 的 `kind_mask`、`policy.ACTIVE_HEADS`、全部既有 checkpoint 仍是 5 | 阻断 |
| R-01 | 课程学习（7 阶段：8v8 → 攻移/撤退/集火 → 建造者/工厂 → 资源点/经济 → 迷雾/多地图 → 内置 AI → 自博弈） | `README.md:24`、`铁锈战争强化学习系统架构.md:95,659-668`、`README.md:218`（"文档覆盖：PPO、多环境、课程学习和评估方案"）、`docs/平台搭建方案.md:467`（"课程推进"作为第 6 步） | `grep -rn 'curriculum|Curriculum' src tools scripts web tests configs` → 只命中 `src/ironzero/training/curriculum.py:5`；该文件**未被任何模块 import**（`training/__init__.py` 无导出），且 `get_config()` 两个分支都 `return self.base_config`，`self.stage = 3` 是硬编码注释 | **不存在**（纯占位死代码，且文件未跟踪 = 连提交都没有） | 高 |
| R-02 | 自博弈 / 模型晋级 / Elo / ModelPool / 对手池 / 联盟训练 | `README.md:229`（标待办）、`铁锈战争强化学习系统架构.md:95,668,901`、`训练推理与演示架构.md:564-575,1269`（晋级门槛"自动判定"）、`命名方案.md:88-91,124`（"Zero = 自博弈血统"） | `grep -rni 'self_play|selfplay|self-play|ModelPool|model_pool|opponent_pool|league|promote|promotion' src tools scripts web tests configs` → **0 命中**；`grep -i 'elo'` 只命中无关词（`below`/`Envelope`/`velocity`）。对手只有 `opponent: scripted|idle|random`（`micro_config.py:98`），真机恒为 `scripted` | **不存在** | 高 |
| R-03 | 多进程并行训练（8–32 实例，每实例独立端口/日志/episode） | `训练推理与演示架构.md:397-427`、`铁锈战争强化学习系统架构.md:645-657`、`README.md:24` | `src/ironzero/training/vec_rwpp.py:22-64`（N 个 `RwppDuelEnv`，`ThreadPoolExecutor` 管 socket）；`rwpp_env.py` 用 `subprocess.Popen` 拉起独立 rwpp-server；`scripts/rwpp_real_service.py:41,1062`（`MAX_PARALLEL=4`，端口池 5130..5133） | **部分**：确实"每实例一个游戏进程 + 独立端口"，但 Python 侧训练仍是单进程线程池；实测/脚本上限 4（服务强制 `concurrency must be 1..4`，`rwpp_real_service.py:965`），文档承诺的 8/16/32 从未出现 | 中 |
| R-04 | PPO + GAE + 动作掩码（层级采样，未用头不进对数概率） | `铁锈战争强化学习系统架构.md:620-643`、`docs/平台搭建方案.md:133-140` | `src/ironzero/training/ppo.py`（GAE/clip/异步 eval/best 保存）、`policy.py:108-137`（`_active_matrix` 按 kind 屏蔽无用头）、`tests/test_training_pipeline.py` | **已实现**（唯一完全达标的核心项） | 无 |
| R-05 | 多环境 wrapper 套件：`ActionMaskWrapper / FrameStackWrapper / RewardNormalizationWrapper / EpisodeStatsWrapper / ScenarioCurriculumWrapper` | `铁锈战争强化学习系统架构.md:450-460` | `grep -rn 'FrameStackWrapper|ActionMaskWrapper|RewardNormalizationWrapper|EpisodeStatsWrapper|ScenarioCurriculumWrapper'` → **0 命中**；`src/ironzero` 无 `wrappers/` 目录 | **不存在** | 中 |
| R-06 | 行为克隆（人类/原生 AI 数据 → BC 预训练 → PPO 微调） | `铁锈战争强化学习系统架构.md:698-709` | `scripts/bc_warmstart_rwpp.py` 存在；`train_rwpp.py:87` 有 `--init-checkpoint`；但 `bc_warmstart_rwpp.py:68` 的 `--episodes` **从未被使用**（`collect()` 只按 `--transitions` 截断，`L169`） | **部分**（BC 脚本半成品，`--episodes` 参数失效） | 中 |
| R-07 | LLM 负责战略、RL 负责战术（1–5 s 一次战略层；关闭 LLM 后 RL 独立运行） | `README.md:137,230`、`铁锈战争强化学习系统架构.md:40,753-767`、`训练推理与演示架构.md:22` | `src/ironzero/strategy/{directive,rule_strategy,llm_strategy}.py`；`rwpp_env.py:849-870`（`_refresh_strategy`/`_strategy_potential` 只做 reward shaping）；`train_rwpp.py:89,161`；`tests/test_strategy.py` 36 用例 | **部分（训练侧已实现，推理侧未接）**：`--use-llm` 只在 `train_rwpp.py` 与 `rwpp_real_service.py:522` 的 PPO 分支；`scripts/play_rwpp_policy.py:111-129` 的 argparse **没有 `--use-llm`**，Web 推理请求 `RwppRecordRequest`（`web/backend/app.py:89-103`）也没有该字段 → README 把它写在"模型推理与 LLM + RL 混合"标题下会让人以为推理也能开 | 中 |
| R-08 | 基线必须保留：随机合法动作 / 规则脚本 / 原生 AI easy·medium·hard / 训练前后 PPO | `铁锈战争强化学习系统架构.md:671-696`、`训练推理与演示架构.md:534-562`、`环境规格说明.md:719-736` | `evaluate.py`（`evaluate_random`+`scripted`）；`scripts/baseline_rwpp.py`（真机脚本基线）；`grep -rn 'difficulty|native_ai|aiDifficulty'` → **0 命中**；Java 侧 `AI` 只有 `--unlock-ai`（`aggressiveMode=true`，`IronZeroBridge.java:548`），无难度参数 | **部分**（easy/medium/hard 三档原生 AI 基线不存在） | 中 |
| R-09 | 固定种子真机确定性（`run_seed/episode_seed/map_seed/...` 派生并记录） | `docs/确定性与测试方案.md:34-77`、`docs/环境规格说明.md:511-543` | `rwpp_env.py` `reset(seed)` 只写 `self.seed`/`episode_id`；`IronZeroBridge.java` `grep -i seed` → **0 命中**；`docs/训练问题日志.md:34,60` 自认"仍未修" | **不存在**（引擎侧 seed 不可控，产物里也不记录） | 中 |

### 2.2 环境与接口契约

| # | 声称的功能 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-10 | Gymnasium 环境（`reset/step` 返回 5 元组，可注册） | `铁锈战争强化学习系统架构.md:402-412`、`README.md:14` | `src/ironzero/env.py:64,232-241`（镜像 API + 可选 `gymnasium.register("IronZeroMicroDuel-v0")`）；`rwpp_env.py:973-1000`（自带 `_BoxSpace/_MultiDiscreteSpace`）；`pyproject.toml` 依赖只有 `pydantic` | **部分**（"兼容"= 形似 + 可选注册；无 gymnasium 依赖，`RwppDuelEnv` 不继承 `gym.Env`，`info` 里也没有 spec 要求的 `reset_mode`/`schema_version`） | 低 |
| R-11 | `MatchSpec` 统一比赛契约（`match_id/mode/scenario/seed/opponent/max_ticks/decision_interval_ticks/action_repeat_ticks/fog_of_war/record_policy/versions`） | `训练推理与演示架构.md:192-227`、`L43`（"这些模式共用 MatchSpec"）、`L1185`（M0 产物） | `grep -rn 'MatchSpec|match_spec' src scripts web tests` → **0 命中** | **不存在** | 中 |
| R-12 | Episode 状态机 `CREATED → STARTING → WAITING_FIRST_OBS → RUNNING → TERMINATING → COMPLETED`，异常走 `FAILED → CLEANUP` | `训练推理与演示架构.md:229-245`、`环境规格说明.md:228-271` | `grep -rn 'EpisodeStatus|CREATED|WAITING_FIRST_OBS' src` → **0 命中**；`protocol.py:65` 只有 `HeartbeatStatus`；`RwppDuelEnv` 只有布尔 `self.done` | **不存在** | 中 |
| R-13 | `lockstep` 与 `realtime` 两种调度（训练/评估用 lockstep，推理/演示/lan_eval 用 realtime，超时执行安全默认动作） | `训练推理与演示架构.md:261-281`、`环境规格规格说明.md:183-224` | 全仓无 `realtime` 调度代码；`RwppDuelEnv.reset` 每次重启 JVM（`rwpp_env.py` 注释自承"真机不可原地 reset"）；无 `deadline_tick`/`POLICY_TIMEOUT` 执行路径（`POLICY_TIMEOUT` 仅在 `protocol.py:51` 定义，无引用的抛出点） | **部分**（只有 lockstep 式阻塞 `step`；realtime 与安全兜底不存在） | 中 |
| R-14 | 独立 Policy Service（`reset/act/close/metadata` + JSON 请求/响应 + 超时不阻塞游戏线程） | `训练推理与演示架构.md:583-637,1275` | `grep -rn 'Policy Service|PolicyService' src scripts` → 无服务实现；实际由 `scripts/play_rwpp_policy.py` / `eval_rwpp_policy.py` 内联加载 `load_policy` | **不存在**（只有内联脚本） | 低 |
| R-15 | Safety Supervisor（schema 校验 → 归属 → 可见性 → 掩码复验 → 资源/冷却 → 限频 → 去重 → 回执） | `训练推理与演示架构.md:182,701-727` | 采样期掩码：`observations.build_masks`；执行期校验：只有 mock `mock_engine.py:310-500` 的 `ACTION_MASKED/STALE_TICK/...`；真机 Java `applyStep` 只回 `accepted/rejected + ACTION_MASKED`（`IronZeroBridge.java:650-656`）；`SequenceGuard`（`protocol.py:407-431`，实现 STALE_EPISODE/STALE_TICK/DUPLICATE_DECISION）**全仓无人调用**（仅 `__init__.py` 导出） | **部分**（无独立安全层；真机不校验可见性/资源/冷却/限频；去重守卫未接线） | 高 |
| R-16 | 战争迷雾（游戏级可见性：`VISIBLE / EXPLORED_NOT_VISIBLE / NEVER_SEEN / DEAD_LAST_SEEN` + `last_seen_tick` 记忆 + 迷雾审计 `visible_enemy_count / hidden_enemy_leak_count / fog_audit_result`） | `环境规格说明.md:681-713`、`训练推理与演示架构.md:715-727`、`README.md:129`、`docs/平台搭建方案.md:325` | **观测级**过滤：`rwpp_env.py:371-397 apply_vision`（只按 `vision_radius` 距离裁剪敌人列表）；`mock_engine.py:625-635 _visible_to`；**无** `remembered_enemies` 生成（`protocol.py:268` 默认空列表、`mock_engine.py:1036` 恒 `[]`）、**无** `last_seen_tick` 写入、**无** `fog_audit_*` 任何字段（`grep 'fog_audit|hidden_enemy_leak|visible_enemy_count'` → 0 命中） | **部分**（只有"看得见/看不见"的距离裁剪，没有记忆、没有审计、不区分 EXPLORED/NEVER_SEEN） | 中 |
| R-17 | 观测历史（`frame_stack` 或 `recurrent_state`，否则必须标 `memoryless`） | `环境规格说明.md:414-423` | `grep -rn 'frame_stack|GRU|LSTM' src` → **0 命中**；`policy.py` 是纯 MLP，`flat_obs_dim` 单帧 | **不存在**（且文档要求的 `memoryless` 标注也没有） | 低 |
| R-18 | 观测/动作 schema 版本校验（不兼容 checkpoint 拒绝加载） | `环境规格说明.md:763-788`、`训练推理与演示架构.md:1094-1099`、`docs/安全与发布检查表.md`（模型 manifest） | `load_policy`（`ppo.py:590-620`）只从 payload 读 `micro_config`/`ppo_config` 重建网络；checkpoint payload 键只有 `model/micro_config/ppo_config/update/metrics`（实测 `torch.load` 于 `runs/micro_duel`、`runs/ironzero25_ft1`），**无** `schema_version/optimizer/RNG/git_commit` | **不存在**（§2.1 R-00 就是它的直接后果：形状不匹配只会抛出 RuntimeError，不是可读的版本拒绝） | 中 |
| R-19 | 版本与兼容性门禁（每次 run 记录 13 项版本；升级后重跑 8 项测试） | `训练推理与演示架构.md:1061-1092`、`环境规格说明.md:765-777` | `train_rwpp.py:170-190` 打印 `mode/device/gpu/envs/episode_ticks/units_per_team/obs_dim/action_heads/terrain`（无 `game_version/rwpp_commit/schema_version`）；`vendor/rwpp/pinned-versions.json` 只锁第三方 | **部分**（版本清单文件存在，run 产物不落版本） | 低 |
| R-20 | NDJSON 首版传输 + 消息大小上限 + 协议错误码全集被强制执行 | `docs/协议v1说明.md:11-30`、`训练推理与演示架构.md:342-368` | `bridge.py:29` `MAX_MESSAGE_BYTES = 1<<20` 且 `L79-94` 有实际裁剪；`protocol.py:40-54` 定义 13 个错误码；真机 Java 从不发 `error` 消息（只有 `receipt.rejected_reason="ACTION_MASKED"`） | **部分**（mock 侧完整；真机侧错误码几乎不落地） | 低 |

### 2.3 动作空间、经济与建造（高风险区）

| # | 声称的功能 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-21 | 第一版 5 类动作 `move_group/attack_move/attack_target/stop/hold` + 4 组 + 广播槽，真机生效 | `README.md:138`、`铁锈战争强化学习系统架构.md:270-278`、`docs/平台搭建方案.md:349-353` | Java `applyStep`：`IronZeroBridge.java:619-642`（5 类齐全，`setMoveWaypoint/setAttackMoveWaypoint/setAttackUnitWaypoint`，`holdGround` 切 `guardArea`）；组语义 `id % 4`（`L671,695`）；HEAD `observations.build_masks` 有 `group_ids_known` 分支（无 `group_id` 时 4 组全开） | **已实现**（HEAD）；⚠️ **工作树倒退**：`observations.py` 当前版本删掉了 `group_ids_known` 分支（`git diff` 可见），导致真机 group 掩码又退回"只剩 group0+广播"，并被 `tests/test_rwpp_25v25.py:178` 抓到 | 高（工作树回归） |
| R-22 | 经济与建造：`build_extractor(slot_id)`、`build_structure(type,x,y)`、`produce_unit(factory,type)`、`upgrade_building(id)`、`repair_target` + 矿井 T1/T2 + 工厂生产队列 + 资源点 | `docs/经济与建筑系统方案.md:55-89`（全篇）、`README.md:227`（路线"经济场景"）、commit `b6a931c` | 协议/schema：`protocol.py:118-146,166-176`、`schemas/ironzero/1/messages.schema.json`（完整）；**mock 引擎完整**：`mock_engine.py:340-447`（建/升/产/修）、`mock_engine.py:554-590`（income/生产队列）、`tests/test_economy.py` 7 用例；**真机 Java 只认 5 类动作**（`IronZeroBridge.java:619-642`，无 build/produce 分支） | **部分（仅 mock 闭环，真机不存在）**。真机发经济动作 → `rejected++`；`runs/econ_*/metrics.json` 的 `reward_components` 全为 `{}` 印证经济奖励从未进真机 | 高 |
| R-23 | 宏观-微操分层 RL（`MacroController` 规则运营 + Micro PPO 战术，`VecMacroMicroEnv` 端到端训练评估） | `docs/经济与建筑系统方案.md:11-51,82-89`、`README.md:230` 附近、commit `b6a931c` | `strategy/macro_controller.py`（建矿/建厂/排队/升 T2）、`training/vec_macro_micro.py:22-141`、`train_macro_micro.py`、`evaluate_macro_micro.py`、`tests/test_macro_micro_pipeline.py` 2 用例 | **部分（仅 mock）**：`VecMacroMicroEnv` 硬绑定 `MicroDuelEnv`（`vec_macro_micro.py:16,48-56`），未接 `RwppDuelEnv`；`runs/macro_micro_ppo/metrics.json` 只有 4 updates，`final_eval` = 0 胜 4 负（`loss_rate=1.0`），不构成可用结果；真机 Java 不支持这些动作 | 高 |
| R-24 | 防御炮塔（Turret：固定阵地、独立警戒与索敌攻击） | `docs/经济与建筑系统方案.md:64` | `mock_engine.py:51,375,386`（可建、有 450 HP）；但 `_select_target`（`mock_engine.py:609-611`）对建筑一律 `return None`，`_move` 对建筑 `return`（`L634-635`）→ 炮塔**不会开火** | **部分（可建造、不可战斗）** | 中 |
| R-25 | 分层动作空间"动作类型头：移动/攻击/**建造/生产**/撤退/停止" + `count` 头 + `count_mask` | `铁锈战争强化学习系统架构.md:290-312`、`环境规格说明.md:458-478` | `micro_config.py:100-110` mock 侧 kind 只有 7 档（`build_mine/build_factory` 注释与 `observations.py:309-311` 的索引 5/6 不一致）；无 `count` 头、无 `count_mask`；HEAD 实际 `action_nvec=(5,5,8,8,9)` | **部分** | 低 |
| R-26 | `Position` 掩码反映可通行性（"目标位置不可通行 → 拒绝"） | `铁锈战争强化学习系统架构.md:311,317`、`环境规格说明.md:487-497` | `observations.py:319-320` `x_mask/y_mask = np.ones(...)` 恒 True；`POSITION_BLOCKED` 只在 mock 引擎定义，真机不校验（`applyStep` 直接下发 waypoint）；地形只在**观测**里加第 4 通道 | **部分（掩码不反映地形；真机不做可通行校验）** | 中 |

### 2.4 评估、验收与"证据"

| # | 声称的功能 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-27 | N≥20 确定性评估 + Wilson 95% CI + 逐局明细落盘 | `docs/闭环差距分析.md:266-340`（S2-04 修复要求）、`docs/训练问题日志.md:57` | `evaluate.py:109-168`（`wilson_interval`，`summarize` 输出 `*_ci95`）；`scripts/eval_rwpp_policy.py:64`（`--episodes` 默认 20）；产物 `runs/final_eval_ft/eval_report.json`（N=40、`win_rate_ci95=[0.5981,0.8581]`、逐局 `episode_results` 40 条） | **已实现**（这是最扎实的一项；`runs/final_eval_ft` = 30胜/10负/0超时，N=40） | 无 |
| R-28 | `checkpoint_best.pt`（按 `(win_rate, mean_return)` 选 best）+ `load_policy` 从权重恢复 hidden | `docs/闭环差距分析.md:170`、`docs/训练问题日志.md:58,68` | `ppo.py:189-214 maybe_save_best`、`L547,561` 输出 `best_checkpoint`；`ppo.py:590-620` 从 `trunk.0.weight` 推断 hidden；`runs/*/checkpoint_best.pt` 实测存在（`micro_probe`、`ironzero25_ft1`、`macro_micro_ppo`、`econ_*`） | **已实现** | 无 |
| R-29 | `timeout` 有终局惩罚 / 评估 horizon 固定 ≥3600 tick | `docs/闭环差距分析.md:223`（S3-02）、`docs/训练问题日志.md:66` | `rwpp_env.py:710` `"timeout": 0.0` 恒 0（HEAD 与工作树一致） | **不存在**（仍恒 0；文档自认"仍未修"） | 低 |
| R-30 | 观测槽位稳定 + `entity_id` 特征（Markov 性） | `docs/训练问题日志.md:69`、`环境规格说明.md:382-389`（"entity_id 必须稳定、排序必须稳定"） | `observations.py:_entity_features` 10 维 = `[occupancy, x, y, hp, group, order, relx, rely, ndx, dist]`，**无 entity_id/team**；槽位按列表顺序填充，死亡左移 | **不存在**（文档自认遗留） | 中 |
| R-31 | 评估集与训练集分离（不同种子/地图/出生点/对手） | `训练推理与演示架构.md:511-521`、`docs/确定性与测试方案.md:420-426` | `final_eval_ft`（seed 61000+）vs `final_eval_ft0`（0 起点）vs `eval_ft20_holdout`——三组目录名显示做过 holdout；`eval_report.json` 只记录 `seed/map/map_path`，**无** train/holdout 标记字段 | **部分**（人工做过 holdout 目录，机制未固化） | 低 |
| R-32 | 模型晋级门槛自动判定（6 条 + 写进报告） | `训练推理与演示架构.md:564-575,1269` | `grep -rn 'promotion|晋级|threshold' src scripts web` → 无判定器；`eval_report.json` 无 `gate/pass` 字段 | **不存在** | 中 |
| R-33 | `README.md:14` "**已实现并通过 102 项测试**" | `README.md:14` | `grep -c '^\s*def test_' tests/*.py` = **201**；HEAD 全绿 201 | **数字过期**（102 → 201）；`docs/平台搭建方案.md:364` 的"148 个测试"同样过期 | 低（事实性虚标） |
| R-34 | `README.md:14` 200 轮 PPO 评估胜率 0% → 100%、随机基线 0%、平均回报 −1.33 → +1.21 | `README.md:14`、`docs/平台搭建方案.md:164-178` | `runs/micro_duel/summary.json`：`final_eval` 16 局 `win_rate=1.0`、`mean_return=1.2107`；`random_baseline` `win_rate=0.0`、`mean_return=-1.3268`；`total_seconds=246.6` | **已实现且可复现**（仅离线参考环境） | 无 |
| R-35 | `README.md:97`/`web/README.md:62`/`平台搭建方案.md:215` "**73 张**地图可选" | `README.md:97`、`web/README.md:62`、`docs/平台搭建方案.md:215` | 脚本遍历 `.game_cache/maps`：TMX 89 个 → 解析失败 0 / tileset 破损 11 → 可用 78；前端 `feInferMapList`（`web/frontend/app.js:6609`）只留 `skirmish/challenge/testing/ironzero` → **实际 69 张**；"73" 是"可用非 IronZero 图"（78-5），与界面不一致 | **数字不符**（69 ≠ 73），且界面上限受文件夹白名单限制 | 低 |

### 2.5 平台、回放与可视化

| # | 声称的功能 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-36 | 回放/沙盘：每局落盘 `ironzero.replay/1`，可重放/删除/跨工具复用 | `README.md:99,14`、`docs/平台搭建方案.md:219`、`训练推理与演示架构.md:873-907` | `src/ironzero/replay.py:29`（`REPLAY_SCHEMA="ironzero.replay/1"`，`build_meta/build_frame/write_replay/load_replay`）；`web/backend/app.py:644-668`（`GET/DELETE /api/replays`）；产物 `runs/recordings/**/replays/*.json`、`runs/platform/**/replays/` | **已实现** | 无 |
| R-37 | 沙盘真实地图地形模拟（A* 绕水、碰撞不穿墙）+ 迷雾 + 道具层 | `README.md:14,97`、`docs/平台搭建方案.md:215-217` | A*：`terrain.py:281-330`（4 邻域 A*、节点上限）；供 `MicroDuelAdapter(terrain=...)` 用（`mock_engine.py:112-135,468-491`、`training/record.py:89-145 --sim-mode terrain`）；前端迷雾：`app.js:3547,4281,4436`（视觉层）；道具层：`app.js:3312-3345` 只渲 `Items`/`Units` 两层瓦片 | **部分**（A* 只作用于**离线 mock 沙盘**，真机对局不走路网；前端迷雾是"按半径画暗"的视觉层，不是可见性数据源） | 中 |
| R-38 | 视频记录（`canvas.captureStream + MediaRecorder` → `.webm`） | `README.md:97`、`docs/平台搭建方案.md:218` | `web/frontend/app.js:4712-4815`（`startRecording/stopRecording/finalizeRecording`，mime 候选 `vp9/vp8/webm`，下载 `ironzero-sandbox-*.webm`） | **已实现**（浏览器端，非服务端） | 无 |
| R-39 | 平台"实机推理/实机训练"两页 + 并发 1..4 + PPO 真机训练大屏 | `README.md:95-101`、`docs/平台搭建方案.md:276-340` | `web/frontend/{index.html,app.js}`（`fe-ppo-*` 控件、`/api/rwpp/ppo`）；`web/backend/app.py:776-860`（`/api/rwpp/{status,record,infer,train,ppo,jobs,stop_all}`）；`scripts/rwpp_real_service.py`（端口池、409 互斥） | **已实现**（宿主服务必须另起，README L90 已写明） | 无 |
| R-40 | Docker 一键启动（compose 挂载 `runs/`、`.game_cache/`；健康检查） | `README.md:90-92`、`web/docker-compose.yml`、`web/Dockerfile` | `web/Dockerfile:31-45`（`HEALTHCHECK /api/health`、`EXPOSE 8000`）；`web/docker-compose.yml`（`../runs:/app/runs`、`../.game_cache:/app/game_cache:ro`、`IRONZERO_RWPP_SERVICE`） | **已实现**（未实跑镜像构建；`runs/_docker_build.log` 存在说明构建过） | 待确认 |
| R-41 | `runs/` 根 `_probe_*.py` 一类临时探测脚本是否被引用 | —（临时产物，非文档声称） | `grep -rn '_probe_cdp|_probe_click|_probe_panel|_test_appjs|_tests_out'`（全仓，排除 .git）→ **0 命中** | 无引用，纯残留（详见 §4） | 低 |

### 2.6 `lan_eval`（内网人机评估）

| # | 声称的功能 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-42 | 第四个运行面 `lan_eval`：隔离内网 RWPP Server + 真人客户端 + 审计日志 + 知情同意记录 + 结束后关监听 | `训练推理与演示架构.md:8,25,39,145-168,768-780,805-869,1286-1300`、`docs/确定性与测试方案.md:436-444`、`docs/安全与发布检查表.md:245` | `grep -rni '\blan\b|lan_eval|内网|人机' src scripts web tools tests` → 仅 `land`/`plan` 之类误命中；**无** `lan_eval` 入口、无房间启动/关闭脚本、无参与者确认记录、无审计日志格式。`docs/` 与根文档里共 23 处提到"内网人机" | **不存在（纯文档设计）** | 低（文档已把它写成"评估模式"而非已交付，但 `L39` 用表格与 train/evaluate/play 并列，容易被读成同级能力） |

### 2.7 许可证 / 安全边界声称（本审计范围内）

| # | 声称 | 文档出处(file:line) | 验证方式 | 现实 | 严重度 |
|---|---|---|---|---|---|
| R-43 | "当前仓库**不包含** RWPP 源代码或 RWPP 扩展代码" | `LICENSING.md:40` | `git ls-files src/ironzero_bridge` → `src/ironzero_bridge/java/ironzero/IronZeroBridge.java` **已被跟踪**（766 行）；其 `import` 含 `io.github.rwpp.server.ServerExtension`、`io.github.rwpp.api.game.impl.NativeMapLoader`、`io.github.rwml.api.ExtensionContext`（`IronZeroBridge.java:1-29`），并实现 `ServerExtension` → **这正是一个 RWPP 扩展** | **自相矛盾**：不是"复制上游源码"（这点 `THIRD_PARTY_NOTICES.md:33` 说"未复制代码"是准确的），但确实是"RWPP 扩展代码"，且按 AGPLv3 分发该扩展需单独标注许可证。`LICENSING.md:46-49` 建议的未来目录 `game-agent/rwpp-extension/` 与实际 `src/ironzero_bridge/java/` 不一致 | 中 |
| R-44 | "训练和默认推理**只绑定回环地址**、**不监听 0.0.0.0**" | `SECURITY.md:24`、`docs/安全与发布检查表.md:68-69,403` | `src/ironzero_bridge/java/ironzero/IronZeroBridge.java:201` ✅ `new ServerSocket(port, 1, InetAddress.getByName("127.0.0.1"))`；`src/ironzero/bridge.py:130,204` ✅ 默认 `host="127.0.0.1"`；**但** `scripts/rwpp_real_service.py:1061` `--host` **默认 `"0.0.0.0"`**（`L1067` `ThreadingHTTPServer((args.host, args.port), Handler)`），`web/Dockerfile:37` 也是 `--host 0.0.0.0` | **不成立**：宿主机真机服务（暴露 `/infer /train /ppo /stop_all`，可拉起游戏进程）默认对所有网卡监听；与 SECURITY.md 的边界声明冲突 | 中 |
| R-45 | 许可证范围表覆盖仓库全部内容 | `LICENSING.md:10-22` | 表中只列 `src/ tests/ tools/ scripts/ schemas/ vendor/ pyproject.toml site/ docs/ 根 md`；`grep` 检查目录名 → `web/`、`configs/`、`.github/`、`audit/` **未列入**。且 `LICENSING.md:15` 把 `scripts/` 说明为"PowerShell 脚本"，实际 20 个文件里 **14 个是 `.py`** | **部分（有 4 个目录无许可证归属）** | 低 |

---

## 3. 虚标 / 过时文档清单

### 3.1 需要改文案（写成了系统能力，实际只有 mock 或空壳）

| 文件:行 | 现有表述 | 问题 | 建议改法 |
|---|---|---|---|
| `README.md:24` | "支持 PPO、多环境并行、**课程学习**和后续自博弈" | 课程学习无实现；自博弈无实现 | 改成"规划：课程学习 / 自博弈（见 §路线，均未实现）" |
| `README.md:227` | `- [ ] 经济场景：采矿、建造、生产和资源点` | 下面 `[x] 完整 1v1`、`[x] LLM+RL` 都勾了，唯独经济未勾，但 `docs/经济与建筑系统方案.md` 与 commit `b6a931c` 又像"已交付" → **两份文档互相矛盾** | 勾选状态改为"🟡 仅离线 mock 闭环，真机桥不支持" |
| `README.md:228` | `- [x] 完整 1v1 部分：多地图 + 战争迷雾 + 内置 AI 基线（自博弈待做）` | 迷雾只是观测半径裁剪；"内置 AI 基线"无 easy/medium/hard 分档 | 改为"多地图 ✅ / 迷雾＝观测级半径裁剪 / 内置 AI 仅默认难度" |
| `README.md:14` | "已实现并通过 **102 项测试**" | 实际 201 项 | 改为 201（或去掉数字，改为 CI 徽章） |
| `README.md:97`、`web/README.md:62`、`docs/平台搭建方案.md:215` | "**73 张**地图可选" | 界面实际 69 张（文件夹白名单过滤后） | 改为"69 张可选（skirmish/challenge/testing/ironzero）" |
| `docs/平台搭建方案.md:364` | "全仓 **148** 个测试通过" | 实际 201 | 同步 |
| `训练推理与演示架构.md:39,192-227,229-245,583-637,564-575,701-727` | `lan_eval`、`MatchSpec`、Episode 状态机、Policy Service、晋级门槛、Safety Supervisor 均以"系统组件/必须"的口吻列出，且 M0–M7 里标为产物 | 六项全部无代码；`M0` 的验收"所有模式能读取同一份配置"根本不成立 | 加"**§22 实现状态**：以上 M0/M6/M7 组件均未实现"，或把 §4/§5/§9 标为"目标设计（未实现）" |
| `铁锈战争强化学习系统架构.md:450-460` | 5 个 wrapper（含 `ScenarioCurriculumWrapper`） | 一个都没有 | 标"设计建议" |
| `docs/经济与建筑系统方案.md:72-78` | "协议设计（Protocol v2）" 列出 5 个动作 | 协议里这些动作**已存在**（v1 schema 就有），但**真机不执行**；文档未区分 mock/真机 | 增加"§6 实现边界：仅 mock 引擎执行；Java bridge 不支持" |
| `docs/环境规格说明.md:700-713` | 迷雾审计 6 个必填字段 | 无一实现 | 标"待实现" |
| `LICENSING.md:40` | "当前仓库不包含 RWPP 源代码或 RWPP 扩展代码" | `src/ironzero_bridge/java/ironzero/IronZeroBridge.java` 已跟踪且 `implements ServerExtension`（见 R-43） | 改为"包含一个自研 RWPP 扩展（`src/ironzero_bridge/`），不复制上游源码"并补 AGPL 边界说明 |
| `SECURITY.md:24`、`docs/安全与发布检查表.md:68-69` | "只绑回环 / 不监听 0.0.0.0" | `scripts/rwpp_real_service.py:1061` 与 `web/Dockerfile:37` 默认 `0.0.0.0`（见 R-44） | 二者取一：改默认值为 `127.0.0.1`，或在文档里明确"平台容器对外 `0.0.0.0`，宿主服务需手动 `--host 127.0.0.1`" |
| `LICENSING.md:15` | `scripts/` = "PowerShell 脚本" | 实际 6 个 `.ps1` + 14 个 `.py` | 改为 "PowerShell + Python 脚本" |
| `docs/确定性与测试方案.md:330-381` | CI 分层（每次提交跑单测/schema/契约） | `.github/workflows/` 只有 `pages.yml`，**没有任何测试流水线** | 要么加 workflow，要么把 §7 标为"目标" |
| `docs/训练问题日志.md:30,36` | P-S2-10 "HP 分母未修"、P-S3-01 "仍未修" | **已修**：`play_rwpp_policy.py:92,198-199,209-210,231-232,292` 已传 `own_max_ref/enemy_max_ref` | 状态列改"已修复待验证" |
| `docs/闭环差距分析.md:45,181,365` | "HP 分母仍未修" | 同上，已修 | 同上 |
| `docs/训练问题日志.md:62` | P-S2-09 "无 curriculum" | **仍然正确**（但 `training/curriculum.py` 这个未跟踪空壳应删除，否则下轮审计会误判为"已实现"） | 删除空壳或明确注释 `# TODO` |

### 3.2 文档自列待办 ↔ 代码现状 交叉结论（第 3 项任务）

`docs/闭环差距分析.md` §3（S1-01..S3-04，共 23 条）+ `docs/训练问题日志.md` §1（P-S1-01..P-S3-08，共 24 条）的交叉核对结果：

| 文档编号 | 文档状态 | 代码现状（本审计实测） | 结论 |
|---|---|---|---|
| S1-01 / P-S1-04 观测截断 8 实体 | 修复中 | **已做**：`--units-per-team/--max-entities/--target-slots`（`train_rwpp.py:66-78`）；`runs/ironzero25_ft1/checkpoint_best.pt` 实测 `units_per_team=25, max_entities=25, target_slots=26, obs_dim=768` | 文档过期（低估） |
| S1-02 采样结构 `episodes=0` | 待修 | **部分改善**：`runs/ironzero25_ft` = 120 updates / **116 episodes**（`rwpp_summary.json`），`runs/ironzero25_ft1` = 120/100；但 `train_rwpp.py:41` 默认仍 `rollout_steps=8` | 文档仍准确（默认值未改），但已有可用训练 |
| S1-03 / P-S1-01,02 无真实胜负/假胜 | 已修复待验证 | **已做**：Java `detectGameResult` + `done.result`（`IronZeroBridge.java:346-415`）；`runs/final_eval_ft` 逐局 `result=victory/defeat`、`bridge_result` 一致 | 已验证（N=40） |
| S1-04 无建筑早判负 | 修复中 | **已做**：`--win-mode annihilation` + `ANNIHILATION_GRACE_TICKS=30` + 每帧 `clearNativeDefeatFlags`（`IronZeroBridge.java:50-52,128-140,353-380`） | 文档过期 |
| S1-05 `aiLimitedUse` AI 不参战 | 修复中 | **已做**：`--unlock-ai`（`IronZeroBridge.java:193,524-560`） | 文档过期 |
| S1-06 / P-S2-10 推理与训练不一致 | 修复前确认 | **已做**：`play_rwpp_policy.py:178` 从 checkpoint 读 `vision_radius`；`eval_rwpp_policy.py:75` 默认从 checkpoint 读；HP 参照已传 | 文档过期（低估） |
| S2-01 / P-S2-02 group 掩码 | 修复中 | HEAD **已做**（`group_ids_known` 分支）；⚠️ **工作树被改回**，见 R-21 | HEAD 已修 / 工作树回归 |
| S2-02 / P-S2-03 雾下 target 索引 | 修复前确认 | **已做**：`rwpp_env.py:603 decode_action(action, self.obs_state, ...)` 且注释说明 | 文档过期 |
| S2-03 / P-S2-01 归一化越界 | 修复中 | **部分**：25v25 用 `units_per_team=25` 自然修正；`observations.py:52-53` 仍无 `clip`；legacy 22v25 仍 2.75 | 文档准确 |
| S2-04 / P-S2-04 eval 1 局 | 修复中 | **已做**：`eval_rwpp_policy.py --episodes` 默认 20 + Wilson + `episode_results`；`runs/final_eval_ft` N=40。**但平台默认仍是 1**（`web/backend/app.py:120 eval_episodes=1`） | 文档过期；平台默认值为遗留 |
| S2-05 / P-S2-05 final 非 best | 修复中 | **已做**：`maybe_save_best` + `checkpoint_best.pt` + `runs/*/checkpoint_best.pt` 实际存在 | 文档过期 |
| S2-06 无地形观测 | 修复中 | **已做**：第 4 通道 `observations.py:104-107` + `--terrain`；25v25 checkpoint `map_channels=4` | 文档过期 |
| S2-07 / P-S2-07 固定种子 | 遗留 | **仍未做**：Java 无 `--seed` | 文档准确 |
| S2-08 / P-S2-08 端口/进程 | 待加固 | **仍未做**：`_shutdown_game` 只 `wait(15s)+kill()` 单句柄，未杀进程树 | 文档准确 |
| S2-09 / P-S2-09 curriculum/minibatch | 待决策 | **仍未做**；`rollout_steps=8`、平台 20 | 文档准确 |
| S2-10 / P-S3-02 GUI 单局 eval | 修复前确认 | **仍未做**：`app.py:120 eval_episodes=1` | 文档准确 |
| S3-01 / P-S3-01 推理 HP 分母 | 修复前确认 | **已做**（见 S1-06 行） | 文档过期 |
| S3-02 timeout 惩罚 | 修复前确认 | **仍未做**：`rwpp_env.py:703 "timeout": 0.0` | 文档准确 |
| S3-04 / P-S3-05 `load_policy` hidden 硬编码 | 修复前确认 | **已做**：从 `trunk.0.weight` 读 hidden；⚠️ 但**无 schema 版本校验**，工作树已因形状不匹配炸掉（R-00） | 部分 |
| S3-05 / P-S3-06 观测槽位无 entity_id | 修复前确认 | **仍未做**（`_entity_features` 无 id） | 文档准确 |
| S3-07 `_verify_ppo3600` 无结果 | 修复前确认 | 目录仍只有 `live.jsonl` 1 条 start | 文档准确 |
| S3-08 旧场景 AI 增援 | 修复前确认 | `runs/econ_train_shaping/eval_report.json` 仍见 `enemy_alive=32/33`（>25） | 文档准确 |

**交叉结论（3 句话）**：
1. **两份文档的"差距"有 12 条已经被代码补掉了**（S1-01/03/04/05/06、S2-01/02/04/05/06、S3-01/05 的一部分），但状态列没更新，现在读起来比现实更悲观。**这是"文档过期"，不是"功能缺失"**。
2. **真正还没做的稳定地集中在 2 类**：① 训练配置级（`rollout_steps`、平台 `eval_episodes=1`、timeout 惩罚、curriculum）② 环境/进程级（引擎 seed、进程树清理、观测槽位 id）。
3. **文档没有记录、但现实已产生结果的一条**：25v25 主验收其实已经跑出来了——`runs/final_eval_ft/eval_report.json`（N=40、win 30、Wilson95 [59.8%, 85.8%]、0 timeout、map=IronZero25、`vision_radius=0`）与 `runs/eval_ft20_holdout`（N=20、90%）。`docs/训练问题日志.md` §2 的回填表**全是"待回填"**，仍写着"E4 N≥20 确定性评估 待回填"。这是**唯一一处"成果被文档漏记"**，比虚标更值得先修。

---

## 4. 仓库垃圾清理建议

> 前提：`runs/ artifacts/ .game_cache/ .runtime/ __pycache__/` **均已被 `.gitignore` 覆盖**（逐条 `git check-ignore -v` 验证，见 §4.5），因此这些目录的问题是"占磁盘 + 污染审计视野"，不是"会被提交"。

### 4.1 `runs/` 目录级分类（共 59 个目录 / 1998 文件 / 481 MB）

**A 类：真实实验记录，建议保留（并在文档里引用）**

| 目录 | 判定依据 |
|---|---|
| `runs/micro_duel/` | `README.md:14`、`docs/平台搭建方案.md:164-189` 引用的 200 轮基线，`summary.json` 完整 |
| `runs/ironzero25_round1/`, `runs/ironzero25_ft/`, `runs/ironzero25_ft1/` | 25v25 三轮训练，`metrics.json/cpkt_best/eval_report` 齐全（ft1 有 120 updates/100 episodes） |
| `runs/final_eval_ft/`, `runs/final_eval_ft0/`, `runs/eval_ft20/`, `runs/eval_ft20_holdout/`, `runs/eval_bc_ironzero25/` | N=40/40/20/20/16 的确定性评估报告（**主验收证据**，`docs/训练问题日志.md` §2 待回填的就是它们） |
| `runs/bc_ironzero5/10/15/20/25/` | 行为克隆各规模数据集/权重，`bc_report.json` 存在 |
| `runs/macro_micro_ppo/` | 分层 RL 唯一产物（4 updates） |
| `runs/econ_train_full/`, `econ_train_shaping/`, `econ_train_shaping_v2/`, `econ_test_3/` | 经济系统实验（结论见 R-22/R-23，是有价值的"负结果"证据） |
| `runs/recordings/` | 22 个 `real-rwpp-*` 真机回放（20.6 MB），平台回放库的实际数据 |
| `runs/platform/rwpp-ppo-ppo-07de6fc6/` 等有 `metrics.json` 的 PPO run | `docs/闭环差距分析.md:130` 引用的"真实 defeat 终局"证据 |

**B 类：一次性调试残留，建议删除（共约 40 个目录）**

| 分组 | 目录 | 理由 |
|---|---|---|
| 冒烟 | `_smoke/`, `_smoke_loop/`, `_smoke25/`, `_rwpp_ppo_smoke{,2,3,4}/`, `rwpp_ppo_smoke/` | 一次性"能跑即弃"，无信息量 |
| 基准 | `_bench/`, `_bench_ppo_2_1/`, `_bench_ppo_4_2/`, `rwpp_ppo_bench/`, `rwpp_ppo_bench2/` | `_bench` 只有 meta+replay；`bench`/`bench2` 只有 1 个 live.jsonl |
| 旧失败 bench | `rwpp_ppo_bench3/`, `rwpp_ppo_bench4/` | **假胜/漏判的坏数据**（`docs/闭环差距分析.md:122-128`），保留会让后来者误用 → 建议移到 `runs/_archive/` 并加 `INVALID.md`，而不是直接留原地 |
| 探测 | `micro_probe/`, `micro_probe2/`, `micro_probe3/`, `micro_probe4/`, `_policy_test/`, `_rec_test/`, `_speed0/`, `_terrain_test/`, `_baseline_env_smoke/`, `probe_ironzero25_baseline/` | 探针产物（`probe_report.json`/`frames.jsonl`） |
| 单局 | `_eval_bc_single0/`, `_eval_bc_single1/`, `_eval_ft_single/` | 每目录 1–2 个文件、N=1 的评估，无统计意义 |
| 空/坏 | `final_eval_ft_repeat/`（**0 文件**）、`_verify_ppo3600/`（只有 start 事件） | 空目录 / 不可解释产物 |
| 缓存 | `_rwpp_ppo/`（**915 个 `game-*.log`，8.7 MB**） | 真机 JVM 日志堆积；`docs/闭环差距分析.md` 已把结论抽走，日志本身可清 |
| 半成品 | `econ_test_1/`, `econ_test_2/`（各 1 个 live.jsonl）、`rwpp_ppo_llm_test/`, `rwpp_ppo_llm_mock/`, `rwpp_ppo_reward/` | 单次试跑 |

**C 类：`runs/` 根目录 59 个散落文件（1.8 MB）—— 全部建议删除**
包括 `rwpp_real_service.log`（**903 KB**）、`_qemu_zip.log`（485 KB）、`_rwpp_bridge_smoke.jsonl`（98 KB）、20 份 `_rwpp_record_2026*.log`、`_docker_build.log`、`_edge_dom.html`（0 字节）、`_launch5435/5436.err/out`、`_rwpp_build{,2}.log` 等。这些是上一轮调试的"重定向输出"，无一个被文档引用。

**D 类：`runs/platform/` 内部（340 个子目录 / 389 MB）**
- `real-rwpp-*` 193 个、`rwpp-live-*` 92 个、`rwpp-ppo-*` 41 个。建议：只保留每个 `rwpp-ppo-*` 中带 `metrics.json` 的 run + `recordings/`，其余 `rwpp-live-*`（前端实时流快照，无 replay 索引价值）整批删除。**建议给平台加保留策略**：`IRONZERO_RUNS_RETENTION=20` 之类，只在最近 N 个 run 上保留 `frames.jsonl`。

### 4.2 `runs/` 根目录的临时探测脚本（第 5 项任务）

| 文件 | 是否被 .gitignore 覆盖 | 是否被任何代码/文档引用 | 结论 |
|---|---|---|---|
| `runs/_probe_cdp.py`（2.5 KB，Chrome CDP + websocket 抓页面） | ✅ `.gitignore:25 runs/` | ❌ 全仓 grep **0 命中** | **应删除** |
| `runs/_probe_click.py` / `_probe_click2.py` / `_probe_click3.py` | ✅ 同上 | ❌ 0 命中 | **应删除** |
| `runs/_probe_panel.py` | ✅ 同上 | ❌ 0 命中 | **应删除** |
| `runs/_test_appjs.js`（Node 里 eval `app.js` 抓 boot 异常） | ✅ 同上 | ❌ 0 命中 | **应删除** |
| `runs/_tests_out.txt`（`python -m unittest` 输出重定向，含 201/102 时代的历史日志） | ✅ 同上 | ❌ 0 命中 | **应删除** |

它们**本身不会进 git**，但会：① 让"`runs/` 是产物目录"这一约定失效（有人在里面写源码）；② 让审计反复误判"这里有未被 gitignore 的代码"。删除即可，无需改 `.gitignore`。

### 4.3 仓库根目录 / src 的残留（**这些反而没被 ignore，才是真缺口**）

| 文件 | git 状态 | 问题 | 建议 |
|---|---|---|---|
| `_audit_u3.py`（1.8 KB） | **未跟踪、未被 ignore** | 上一轮审计的 AST 扫描脚本（`grep`/`ast` 统计未使用符号） | 删除（或移入 `audit/` 并加 `audit/_*.py` 规则） |
| `_audit_unused.py`（1.7 KB） | 同上 | 同上 | 同上 |
| `_audit_unused2.py`（3.0 KB） | 同上 | 同上 | 同上 |
| `audit/_tmp_refs.py`、`audit/_tmp_refs2.py` | 未跟踪 | 分析辅助脚本 | 删除 |
| `audit/dead.json` | 未跟踪 | 12 字节的中间文件（本轮审计期间已消失，属其他会话临时文件） | 删除 |
| `src/ironzero/test_tutorial.py`（0.9 KB） | **未跟踪、未被 ignore** | 不是 `tests/` 下的用例（`unittest discover -s tests` 不会发现），且调用已不存在的 API：`info.get('raw_state')`、`env.step(action)`、`kind=5`（HEAD 下是 `build_extractor` 但 mask 未变） | **删除**，如需保留改为 `tests/test_tutorial_flow.py` 并修正 API |
| `src/ironzero/training/curriculum.py`（0.6 KB） | **未跟踪、未被 ignore** | 纯占位死代码，`get_config()` 两分支返回同一 config，无 import | **删除**（这才是 R-01 的"假实现"来源） |
| `.pytest_cache/` | ✅ 已 ignore（`.gitignore:7`） | — | 可删 |

### 4.4 `.gitignore` 缺口（第 6 项任务）

**结论：题目点名的 5 项全部已覆盖，无缺口；真正缺口是"根目录 `_*.py` 和 `audit/` 临时文件"。**

| 目标 | 是否覆盖 | 证据 |
|---|---|---|
| `runs/` | ✅ | `git check-ignore -v runs/_probe_cdp.py` → `.gitignore:25:runs/`；`git ls-files runs` = **0** |
| `artifacts/` | ✅ | `.gitignore:44`；`git ls-files artifacts` = 0 |
| `.game_cache/` | ✅ | `.gitignore:65`；413 个文件（含 89 张 TMX、tilesets、units 贴图）全部未跟踪 |
| `.runtime/` | ✅ | `.gitignore:43`；`bridge-dist/ironzero-bridge.jar`、`bridge-classes/*.class` 均未跟踪 |
| `__pycache__/` | ✅ | `.gitignore:2`（+ `*.py[cod]`） |
| `runs/` 下 `*.log` | ✅ | 双重覆盖（`runs/` 与 `*.log:30`） |
| **根目录 `_*.py` / `_*.js` / `_*.txt`** | ❌ | `_audit_u3.py`、`_audit_unused.py`、`_audit_unused2.py` 均 `git check-ignore` 无输出 |
| **`audit/_*.py` 等分析临时件** | ❌ | `audit/_tmp_refs.py`、`audit/dead.json` 未忽略 |
| `web/frontend/_*.png` | ✅ | `.gitignore:61`（仅 PNG，`_*.js` 未覆盖） |

建议在 `.gitignore` 追加（**只加规则，不删文件**）：

```gitignore
# Root-level scratch scripts / captures from ad-hoc debugging sessions
/_*.py
/_*.js
/_*.txt
/_*.log
audit/_*.py
audit/dead.json
```

> 注意：`/runs/` 已覆盖所有 `runs/_*`，无需为 `runs/` 另加规则。

---

## 5. 修复优先级建议

### P0（阻断：先让"能跑的东西"重新能跑，再谈文档）

1. **回滚或补全工作树里 `kind` 头 5→7 的半成品改动**。现状：`micro_config.py:103` 已改 7，但 `observations.py` 的 `kind_mask`/`policy.ACTIVE_HEADS`/`decode_action` 分支/`schemas` 未同步，导致
   - 201 测试 → 10 FAIL + 1 ERROR（复现：`$env:PYTHONPATH='src'; python -m unittest discover -s tests`）
   - **仓库里所有既有 checkpoint 不可加载**（`load_policy('runs/ironzero25_ft1/checkpoint.pt')` → `size mismatch [5,256] vs [7,256]`）
   - 真机 group 掩码从"4 组全开"退回"只剩 group0+广播"（`observations.py` 删掉 `group_ids_known` 分支）
   处理建议：要么整条链（含 `schemas/ironzero/1/messages.schema.json`、`tests/`）一次改完并重训，要么 `git checkout -- src/ironzero/{micro_config,observations}.py` 回到 201 全绿的 HEAD。
2. **删除 `src/ironzero/training/curriculum.py` 与 `src/ironzero/test_tutorial.py`**（未跟踪、无引用、后者调用已不存在的 API）。不删就会持续制造"课程学习已实现"的假证据。
3. **给 `load_policy` 加 `schema_version` + `obs_dim` + `action_heads` 三件套校验**，形状不符时抛可读错误而非 `RuntimeError`。这是 `环境规格说明.md:763-788` 与 `docs/安全与发布检查表.md` 都要求过、且已经在 P0-1 里真实咬人的项。

### P1（把已经做出来的成果写进文档 / 把虚标改掉）

4. **回填 `docs/训练问题日志.md` §2**：现在有 `runs/final_eval_ft`（N=40、30 胜、Wilson95 [59.8%, 85.8%]、0 timeout）、`runs/eval_ft20_holdout`（N=20、90%）、`runs/ironzero25_ft`（120 updates / 116 episodes）。这是全项目唯一"验收证据齐全却仍标待回填"的地方，优先级高于改任何一句 README。
5. **同步两份差距文档的状态列**：S1-01/03/04/05/06、S2-01/02/04/05/06、S3-01/05 共 12 条改"已修复待验证"（依据见 §3.2）。同时把 `P-S3-01/P-S2-10` 的"仍未修"删掉。
6. **改 3 处数字 + 1 处勾选**：`README.md:14`（102→201）、`docs/平台搭建方案.md:364`（148→201）、`README.md:97`+`web/README.md:62`+`docs/平台搭建方案.md:215`（73→69）、`README.md:227`（经济场景勾选状态改 🟡）。
7. **给 6 个空壳能力加显式"未实现"标记**（`训练推理与演示架构.md` §2/§5/§8.4/§9.2/§10、`铁锈战争强化学习系统架构.md:450-460`、`docs/环境规格说明.md:700-713`、`docs/确定性与测试方案.md:330-381`）：`lan_eval`、`MatchSpec`、Episode 状态机、Policy Service、Safety Supervisor、5 个 wrapper、迷雾审计、CI 门禁。**建议统一加一节 `## 0. 实现状态`**，逐条标 ✅/🟡/❌，这比在正文里插"待实现"更不容易再腐烂。
8. **修 2 处合规/安全自相矛盾**（R-43、R-44）：
   - `LICENSING.md:40` 改口 + 在 `src/ironzero_bridge/` 放一份 `LICENSE-AGPL-NOTICE.md`（该目录的 Java 已跟踪、`implements ServerExtension`，属 RWPP 扩展）；
   - `scripts/rwpp_real_service.py:1061` 的 `--host` 默认从 `0.0.0.0` 改为 `127.0.0.1`（或在 README 里把"需手动 `--host 127.0.0.1`"写成必做步骤），否则 `SECURITY.md:24` 与 `docs/安全与发布检查表.md:68-69` 的边界声明是假的。**注意：`web/Dockerfile:37` 的 `0.0.0.0` 是容器内监听，配合 compose 的 `127.0.0.1:8000:8000` 映射本身是安全的**，只需在文档里说清楚区别。

### P2（清仓 + 防复发）

9. **按 §4.1 删 B 类 40 个目录 + `runs/` 根 59 个散文件**（可回收约 **100 MB**，其中 `_rwpp_ppo/` 915 个日志 8.7 MB、`runs/` 根 `rwpp_real_service.log` 903 KB）。`rwpp_ppo_bench3/4` 不要直接删——移到 `runs/_archive/` 并放 `INVALID.md` 说明"假胜/漏判，禁止引用"。
10. **删除 §4.2 的 5 个 `runs/_probe*/_test_appjs/_tests_out`**（已 ignore、无引用，结论：**应删除**）。
11. **按 §4.4 补 `.gitignore` 的 `/_*.py`、`/_*.js`、`/_*.txt`、`audit/_*.py` 规则**，并删掉根目录 3 个 `_audit_*.py`。
12. **给 `runs/platform` 加保留策略**（最近 N 个 run 保留 `frames.jsonl`，旧 run 只留 `meta.json`+`replays/`），否则 92 个 `rwpp-live-*` 会继续以每次推理 ~1 MB 的速度堆积。

### P3（把"虚标"变成"真实现"，按投入产出排序）

13. **经济/建造打通真机**（最高价值）：Java `applyStep` 加 `build_extractor/build_structure/produce_unit/upgrade_building` 分支 + obs 输出 `resource_slots`/`production_queue_length`/`funds`（`funds` 已有，`income` 恒 0 需修）→ 再让 `VecMacroMicroEnv` 支持 `RwppDuelEnv`。这是 `docs/经济与建筑系统方案.md` 全篇的前提，也是 `macro_micro_ppo`（当前 4 updates / 0 胜 4 负）唯一出路。
14. **课程学习真正实现**：先补一个最小的 `CurriculumScheduler`（按 `update` 切换 `units_per_team` 8→15→25 与 `episode_ticks`），并在 `train_rwpp.py` 里接线 + 落 `live.jsonl` 事件。不需要 7 阶段，2–3 阶段即可摘掉这个虚标。
15. **`realtime` + 安全兜底**（`POLICY_TIMEOUT` 的落地）：`RwppDuelEnv.step` 已有 `obs_timeout`，加"超时执行 `hold` 并记录 `policy_timeout_rate`"即可，成本低、直接支撑 `训练推理与演示架构.md` §9.3 的承诺。
16. **迷雾审计**：至少落 `visible_enemy_count` + `hidden_enemy_count` 两个字段进 `info`，就能让 `环境规格说明.md:700-713` 从"全虚"变成"部分实"。
17. **CI**：把已全绿的 201 个测试 + `tools/validate_schema.py` 挂上 GitHub Actions（`.github/workflows/tests.yml`），这是 `docs/确定性与测试方案.md` §7 唯一的硬缺口，也是防止 P0-1 那类回归的最好办法。
18. **自博弈/模型晋级/Elo**：最后做。当前连"固定对手难度"都没参数化（`grep difficulty` 0 命中），先补 AI 难度三档基线，再谈对手池。

---

## 6. 复现命令（只读，供 Lead 复核）

```powershell
# 1) HEAD 与新工作树各跑一次单测（本报告 R-00 的证据）
$env:PYTHONPATH = "src"; python -m unittest discover -s tests

# 2) 工作树 checkpoint 加载失败复现
$env:PYTHONPATH = "src"
python -c "from ironzero.training.ppo import load_policy; load_policy('runs/ironzero25_ft1/checkpoint.pt', device='cpu')"

# 3) gitignore 6 项核对
git check-ignore -v runs/_probe_cdp.py artifacts/x .game_cache/x .runtime/x src/ironzero/__pycache__/x.pyc
git ls-files runs | Measure-Object -Line          # 期望 0

# 4) 临时脚本无引用
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch '\\\.git\\' } |
  Select-String '_probe_cdp|_probe_click|_probe_panel|_test_appjs|_tests_out'   # 期望 0 命中

# 5) 主验收证据
Get-Content runs/final_eval_ft/eval_report.json -Raw | ConvertFrom-Json |
  Select-Object episodes, wins, losses, timeouts, win_rate, win_rate_ci95

# 6) 空壳能力零命中
Get-ChildItem src,tools,scripts,web,tests,configs -Recurse -File -Include *.py,*.java,*.js |
  Select-String 'self_play|ModelPool|lan_eval|MatchSpec|CurriculumScheduler|FrameStackWrapper'   # 仅 curriculum.py:5
```

---

## 7. 一句话总结

**不是"没写完"，是"写完了的和没写的混在同一批动词里"**——协议、掩码、PPO、真机桥、25v25 评估（N=40、Wilson CI 齐全）、回放、平台、经济/mock 分层都真的能跑；课程学习、自博弈、lan_eval、MatchSpec、Episode 状态机、Policy Service、迷雾审计、CI 一个都没有。**当务之急是：先修掉工作树里那个把 checkpoint 全打死的半成品改动，再把已经跑出来的 25v25 验收结果回填进文档**——改文档的收益比补功能更大，因为最贵的那部分功能其实已经做完了。
