# 训练模块审计报告（area-training）

> **审计范围**：`src/ironzero/training/`（全部 17 个文件）、`src/ironzero/micro_config.py`、`scripts/bc_warmstart_rwpp.py`、`scripts/baseline_rwpp.py`、`tests/test_training_pipeline.py`、`tests/test_macro_micro_pipeline.py`
> **审计基线**：工作区有未提交改动（`git status` 显示 `micro_config.py`、`observations.py`、`rwpp_env.py`、`IronZeroBridge.java` 等已修改，`training/curriculum.py` 未跟踪）
> **完整测试基线**：`python -m pytest` = **11 failed, 190 passed**（本报告已复现）
> **审计时间**：2026-11-11
> **声明**：本报告只读代码 + 运行验证，未修改任何源码。所有结论均附 `file:line`；无法运行时验证的标「待确认」。

---

## 0. 交叉验证方法（可复现）

| 验证手段 | 命令/方式 | 用途 |
| :--- | :--- | :--- |
| 全量测试 | `$env:PYTHONPATH="src"; python -m pytest -q` | 复现 11 failed / 190 passed，逐条定位行号 |
| 符号引用检索 | `git grep -n "<symbol>"`（自动排除 `__pycache__`） | 判定死代码是否真的无人引用 |
| 行为对照实验 | 直接实例化 `MicroDuelEnv` / `PPOConfig` / `VecMacroMicroEnv` 并 step | 区分「声明了」与「真的被计算/真的有效」 |
| 分支对照 | 同一 seed 下 `enable_economy=True/False` 跑同序列动作，逐元素比对 obs | 验证开关两条分支是否真产生不同行为 |
| 张量实际形状 | `torch` 打印 `state_dict` shape / `_active_matrix` 输出 | 验证 5 头 ↔ 7 类动作、旧 checkpoint 兼容性 |
| checkpoint 全量加载 | 遍历 `runs/**/checkpoint*.pt` 调 `load_policy` | 量化契约变更的破坏面 |

**关键环境事实（已实测）**：`DEFAULT_CONFIG.action_nvec = (7, 5, 8, 8, 9)`。注意 **动作向量长度仍是 5（= 头数）**，变化的是**第一个头（kind）的取值域从 5 变成 7**。凡是把「头数 5」和「kind 取值域 5」混为一谈的判断都是错的，本报告已逐条区分。

---

## 1. 一句话结论

**训练管线骨架（PPO/GAE/掩码/checkpoint/eval 钩子）是健康的，但"经济系统 5→7 动作头"这次改动是一次**改到一半、且前后端都没跟上的特性**：python 侧只改了 `micro_config.py` / `observations.py` / `rwpp_env.py` 三处，而 **mock 引擎的动作解码器根本不支持这两个新动作（100% 被 INVALID_MESSAGE 拒绝且无任何惩罚）**、**真机 Java Bridge 完全不支持建造/生产**、**策略网络对这两个动作头只激活 group 头（用到的 x/y/target 头不参与梯度）**、**158 个历史 checkpoint 中 143 个已无法加载**。同时经济奖励字段有一半只在 dataclass 里躺着，`curriculum.py` 是彻底死代码。整体健康度：**核心算法 70 分，经济/分层特性 20 分，处于"半成品上线且默认开启"的危险状态**。

---

## 2. 功能完成度表格

| 功能 | 声称状态 | 证据(file:line) | 实际状态 | 严重度 |
| :--- | :--- | :--- | :--- | :--- |
| PPO + GAE + 掩码 + minibatch 多轮 | 已实现 | `ppo.py:266-330`；实测 1 update 可跑通并写出 4 个文件 | **已闭环**（离线） | — |
| best checkpoint（win_rate 优先） | 已实现 | `ppo.py:189-215`；实测生成 `checkpoint_best.pt` | **已闭环** | — |
| 异步评估线程 | 已实现 | `ppo.py:456-501` | **已闭环** | — |
| Wilson 置信区间 / 逐局报告 | 已实现 | `evaluate.py:109-128`、`evaluate_rwpp.py:92-137` | **已闭环** | — |
| 真机并行训练 `VecRwppEnv` | 已实现 | `vec_rwpp.py:88-145`、`rwpp_ppo.py:41-58` | **已闭环（仅 5 类战术动作）** | 低 |
| 经济系统（采矿/建造/生产） | 提交信息 `3ecd629`「完善分层控制与经济系统奖励分解」；README 路线图仍标 `[ ]` | `micro_config.py:76-95`；`mock_engine.py:335-444`；`observations.py:306-311`；`micro_config.py:103` | **改到一半**：mask/头数已改 7，但**动作解码器无分支、真机 Bridge 无指令**，policy 采样到即被丢弃 | **高** |
| 宏微观分层训练 | 已实现（`vec_macro_micro.py` 注释"end-to-end training and inference in an RTS setting with economy"） | `vec_macro_micro.py:1-8, 84-132`、`train_macro_micro.py:14-53` | **仅 mock 闭环，且训练与评估契约不一致**（见 §6-7） | **高** |
| 课程学习 Curriculum | README:24/218 承诺「课程学习」 | `curriculum.py:1-17`（两分支返回同一对象） | **死代码，无人 import** | 中 |
| 行为克隆热启动 BC | 脚本注释称可产 checkpoint 供 `train_rwpp --init-checkpoint` 微调 | `bc_warmstart_rwpp.py:1-22`；`train_rwpp.py:284-319` | **半成品**：`--episodes` 解析后从不使用；kind 头初始化权重被静默丢弃 | **高** |
| 经济奖励分解（extractor/factory/production/building_loss/idle_funds） | 提交信息「经济系统奖励分解」 | `mock_engine.py:1097-1110`、`1121-1128` | **部分实现**：mock 有效；真机 `income` 恒 0；`idle_funds` 在默认参数下**不可达** | 中 |
| `enable_economy` 开关 | dataclass 布尔开关 | `micro_config.py:77`（工作区已改为 `True`）；`mock_engine.py:181,185,554,997,1097` | **只在 mock 生效**；真机路径完全不读该字段；两条分支的 **observation 逐元素相同** | 中 |
| 观测契约 obs_dim | 文档承诺 768（25v25） | `micro_config.py:110-114`；实测默认 364 | 维度公式一致；但**经济状态（funds/income/resource_slots）不进观测** | **高** |
| `evaluate_macro_micro` 独立评估 | 声称"Independent inference and evaluation runner" | `evaluate_macro_micro.py:1-2, 20-97` | **存在但未接入训练循环**：`PPOTrainer.evaluate` 走的是 `evaluate_policy`（无宏控） | **高** |

---

## 3. 死代码清单

### 3.1 `curriculum.py` —— 确认「已死代码」（题目指定核查项）

| 位置 | 为什么是死的 | 证据 |
| :--- | :--- | :--- |
| `training/curriculum.py:5-17` 整个 `CurriculumScheduler` 类 | **全仓库无任何 import/实例化**。`git grep -n "curriculum\|CurriculumScheduler"` 命中仅：本文件、两篇 md 文档、`audit/*.md` 旧报告。代码侧零引用 | `curriculum.py:5`；grep 结果无 `from .curriculum import` / `import curriculum` |
| `curriculum.py:12-17` `if update_count < 1000: return self.base_config / else: return self.base_config` | 两个分支**返回完全相同的对象**（同一个 `self.base_config` 引用）。函数无论输入什么都返回同一值，等于 `return self.base_config` | `curriculum.py:12-17` |
| `curriculum.py:8` `self.stage = 3` | 赋值后**从未被读取**（类内无其他方法，外部无引用） | `curriculum.py:8`；全文件 14 行无 `self.stage` 读取点 |
| `curriculum.py:3` `import random` | 未使用 | `curriculum.py:3`（文件内无 `random.` 调用） |
| `curriculum.py:2` 空行起头、无 SPDX/模块 docstring | 与同目录 17 个文件风格不一致，是另一次 AI 会话直接丢进来的草稿 | 对比 `ppo.py:1-2` |

**结论：`training/curriculum.py` 是已死代码，17 行文件里 0 行有效逻辑。** 它不是"未完成"，而是"占位文件冒充课程学习实现"——README:24 与 README:218 对"课程学习"的承诺在代码中仍然不存在。

### 3.2 `micro_config.py` 中声明但无使用点的字段（题目指定核查项）

逐字段 `git grep` 全仓库结果：

| 字段 | 声明位置 | 使用点 | 判定 |
| :--- | :--- | :--- | :--- |
| `resource_slots_count: int = 4` | `micro_config.py:87` | **无使用点**（`git grep resource_slots_count` 仅命中声明行）。mock 引擎把 4 个资源点**硬编码**成 4 个 `MockResourceSlot` 字面量 | **死配置** |
| `extractor_t2_reward` | `micro_config.py:91` | `mock_engine.py:1100`（`getattr` 兜底） | 有效（仅 mock） |
| `extractor_reward` | `micro_config.py:90` | `mock_engine.py:1099`；**`rwpp_env.py:702` 复用为 income 系数** | 有效但语义错误（见 §4-8） |
| `factory_reward` | `micro_config.py:92` | `mock_engine.py:1102` | 有效（仅 mock） |
| `production_reward` | `micro_config.py:93` | `mock_engine.py:1103` | 有效但**只有宏控路径才可能触发**（PPO 单独无法造工厂，见 §6-2） |
| `building_loss_penalty` | `micro_config.py:94` | `mock_engine.py:1104` | 有效（仅 mock） |
| `idle_funds_penalty_coef` | `micro_config.py:95` | `mock_engine.py:1110`，但**阈值 1500 是硬编码魔数**，默认参数下永远达不到 | 半死（见 §4-9） |
| `extractor_t2_income_per_second` | `micro_config.py:81` | `mock_engine.py:564, 1005`（真实参与资金结算） | **有效** |
| `extractor_income_per_second` / `base_income_per_second` / `starting_funds` / `extractor_cost` / `extractor_upgrade_cost` / `factory_cost` / `tank_cost` | `micro_config.py:78-86` | `mock_engine.py:181,347,349,375,407,433,556,999`；`observations.py:308,310` | **有效** |
| `turret_cost` | `micro_config.py:85` | `mock_engine.py:375` 的 `else` 分支——但全仓库**没有任何调用方产出 `structure_type != "land_factory"`**（`rwpp_env.py:349`、`macro_controller.py:88`、`tests/test_economy.py:117,176,202` 全是 `land_factory`） | **不可达分支** |
| `unit_price` | `micro_config.py:51` | 训练侧无使用点；仅 `assets.py:205` 写入 | 有效（资产侧） |
| `terrain_path` / `map_channels` / `vision_radius` / `potential_coef` / `win_reward` / `loss_reward` / `opponent` | `micro_config.py:55,61,69,72-74,98` | `observations.py:120-147`、`rwpp_env.py:145-157`、`mock_engine.py:1077-1118` | 有效 |

**「只在 dataclass 里声明」的经济字段**：`resource_slots_count`（完全无用）、`idle_funds_penalty_coef`（默认参数下不可达）。其余经济字段**真的在 mock 引擎里被计算**，但全部**不参与真机训练**（真机 `rwpp_env.py` 只用了 `extractor_reward` 一个，且用法错误）。

### 3.3 `policy.py`

| 位置 | 为什么是死的 | 证据 |
| :--- | :--- | :--- |
| `policy.py:23-29` `ACTIVE_HEADS` | 定义了但**从未被引用**（`git grep ACTIVE_HEADS` 只命中定义行）。实际生效的是 `_active_matrix()`（`policy.py:108-117`）内联的同一套规则 = **重复实现的死常量** | `policy.py:23-29` vs `108-117` |
| `policy.py:151-153` `greedy_action()` | 全仓库**无调用点**（`git grep greedy_action` 仅命中定义）。调用方都直接用 `act(..., deterministic=True)`（如 `evaluate.py:57`、`record.py:203`） | `policy.py:151` |
| `policy.py:38` `nvec: tuple[int, ...] = (5, 4, 8, 8, 9)` | 默认值仍是**旧契约 5**。任何调用方省略 `nvec` 就会静默建出 5 值 kind 头，与 7 值 mask 冲突 → 直接 RuntimeError（已实测） | `policy.py:38`；实测 `RuntimeError: The size of tensor a (5) must match the size of tensor b (7)`（11 failed 之一） |

### 3.4 `ppo.py`

| 位置 | 为什么是死的 | 证据 |
| :--- | :--- | :--- |
| `ppo.py:11` `from dataclasses import asdict, dataclass, field` 中的 `field` | 未使用 | `git grep "field" ppo.py` 仅命中该 import |
| `ppo.py:110` `self.history: list[dict[str, Any]] = []` | 只初始化，**从未 append/读取**。实测 `trainer.train()` 后 `trainer.history == []` | `ppo.py:110`；实测输出 `trainer.history = []` |
| `ppo.py:247-251` `last_masks = masks_to_tensors(...)` / `_ = last_masks` | 算了 mask 张量又立刻丢弃。`last_values = self.policy.forward(last_obs)` 也**没有传 mask**，说明这段是复制粘贴残留 | `ppo.py:247-251` |
| `ppo.py:523-541` `has_final` + 「final evaluation」整块 | 循环内 `update == cfg.total_updates` 时（`ppo.py:456`）**必然**已写入 `update==total_updates` 的评估记录，因此 `has_final` 恒为 True；该块只在 `total_updates==0`（空训练）时才可达 | `ppo.py:456`、`523-524`；实测 1-update 运行 `evaluations[0]["update"] == 1 == total_updates` |
| `ppo.py:580-589` `checkpoint_latest.pt` | 内容与 `checkpoint.pt` **同源同权重**（只少了 `metrics` 字段）。实测两者 `model` 逐元素相等；全仓库除测试外无读取方，`web/backend/app.py:297` 只是 glob 列文件名。属于"双份写盘"冗余 | `ppo.py:568-590`；实测 `same model weights? True` |
| `ppo.py:61` 注释 `actions: torch.Tensor  # (T, B, 5)` | 若指"头数"是对的，但紧邻的 kind 头已是 7 值，注释在 5/7 语境下极具误导性；同样的陈旧注释见 `train_rwpp.py:18` | `ppo.py:61`、`train_rwpp.py:17-18` |

### 3.5 其他

| 位置 | 为什么是死的 | 证据 |
| :--- | :--- | :--- |
| `evaluate_cli.py:29-30` `if not isinstance(micro, MicroDuelConfig): micro = MicroDuelConfig()` | **不可达分支**：`load_policy` 一定返回 `MicroDuelConfig` | `ppo.py:604` `micro = MicroDuelConfig(**payload["micro_config"])` 无条件构造；`evaluate_cli.py:28` 直接赋值 |
| `vec_rwpp.py:45-47, 73-76` `persistent=False` 分支 | 所有生产调用方都传 `persistent=True`（`rwpp_ppo.py:48`、`bc_warmstart_rwpp.py:160`），默认值也是 True；`persistent=False` 路径无任何测试覆盖 | `vec_rwpp.py:36`、`rwpp_ppo.py:48`、`bc_warmstart_rwpp.py:160` |
| `rwpp_ppo.py:56` `baseline_fn=None` | 传了一个"默认就是 None"的参数，随后又在 `rwpp_ppo.py:61-64` 覆写 `baseline()` 返回 `{}` → 该实参完全无作用 | `rwpp_ppo.py:56` vs `61-64`、`ppo.py:145-147` |
| `macro_controller.py:44-50` `reset_stats()` | 全仓库**无调用点**（`git grep reset_stats` 仅命中定义） | 见 §5-4 |
| `baseline_rwpp.py:148` `del obs` | 为了消除未使用变量而写的语法补丁；本身说明 `reset()` 的返回值在此路径被完全丢弃 | `baseline_rwpp.py:147-148` |

---

## 4. 垃圾代码清单

| # | 位置 | 问题 | 证据 |
| :--- | :--- | :--- | :--- |
| 4-1 | `micro_config.py:103` vs `protocol.py:33-34` | **命名与行为不符**：`action_nvec` 注释写 `build_mine, build_factory`，而协议/引擎里的真实 kind 是 `build_extractor` / `build_structure`；`observations.py:309,311` 又各写一种（`build_extractor` / `build_factory`）。同一个动作在三处有三个名字 | `micro_config.py:103`；`protocol.py:33-34`；`mock_engine.py:40-41`；`observations.py:309,311` |
| 4-2 | `src/ironzero/observations.py:314,322` | **占位注释冒充实现**：`# ... (keep existing group logic)` / `# ... rest of the function remains similar` 后面并没有"保留原有逻辑"——原逻辑（`group_id is None` 时暴露全部 group 的兜底，见 git diff 被删除的 12 行）被**真删了**。grep 显示这两行注释是本仓库唯一一处该句式 | `git diff src/ironzero/observations.py`（删除 `group_ids_known` 分支）；可复现失败：`build_masks({'own_entities':[{...}×8], 'global_state':{'funds':0}}, DEFAULT_CONFIG)["group"] == [True, False, False, False, True]` |
| 4-3 | `observations.py:306` | **魔法数字硬编码** `kind_mask = np.ones(7)`，没有用 `len(ACTION_KINDS)` 也没有从 config 派生；`micro_config.py:103` 的另一份 7 是字面量。双份硬编码，下次改头数必然再漏一处 | `observations.py:306`、`micro_config.py:103` |
| 4-4 | `env.py:170-203`（`decode_actions`） | **功能断链的垃圾实现**：`kind_index = int(action[0]) % len(ACTION_KINDS)`（10 种）现在可以解出 `build_extractor`/`build_structure`，但函数里**没有对应分支**，落到末尾的 `return [{"kind": kind, "unit_group": group} ...]`——而 `build_extractor` 需要 `slot_id`、`build_structure` 需要 `x/y`，于是引擎**100% 返回 `INVALID_MESSAGE`**。实测：连发 35 步 `kind=5` 得到 **140 条 rejected**，`funds` 仍在涨（说明确实什么都没发生） | 实测 receipts：`{'kind': 'build_extractor', 'accepted': False, 'rejected_reason': 'INVALID_MESSAGE'}`；`mock_engine.py:340-343, 378-381` |
| 4-5 | `rwpp_env.py:702` | **一行塞两个 key + 语义挪用**：`"income": float(self.micro.extractor_reward) * ... , "bonus_build": 5.0 if ...`。① 把"造矿奖励系数"当成"收入系数"用；② `bonus_build` 用**硬编码 5.0**（比 `potential_coef=0.5` 的势能项大 10 倍，是事实上的主导奖励）；③ 两个 key 挤在同一行，可读性为零 | `rwpp_env.py:702`；对照 `micro_config.py:90`、`mock_engine.py:1099` |
| 4-6 | `rwpp_env.py:702` 的 `"income"` 分量 | **恒为 0 的死分量**：真机 Java Bridge 把 `income` **硬编码成 `0.0`** 上报，所以这个奖励项在真机永远贡献 0 | `IronZeroBridge.java:281` `+ "\"income\":0.0,"` |
| 4-7 | `mock_engine.py:1109-1110` | **魔数阈值**：`if funds > 1500.0` 与 1500 的来源（`starting_funds`? 硬编码?）无关；`idle_funds_penalty_coef` 也没被任何注释解释 | `mock_engine.py:1107-1110` |
| 4-8 | `bc_warmstart_rwpp.py:148-149` | **无意义参数**：`def _grid(size: float, grid: int)` 里 `size` **完全未使用**，函数永远返回网格中心 `int(round(0.5*(grid-1)))`。调用方还认真传了 `micro.map_width` / `micro.map_height`（`:128-129`），暗示"按地图比例选格"但根本没实现 | `bc_warmstart_rwpp.py:148-149, 128-129` |
| 4-9 | `bc_warmstart_rwpp.py:279-291` | **重复实现策略层的 active-head 逻辑**：这里手抄了一遍 `policy._active_matrix` 的 `active[:, 2] = kinds <= 1` 等 5 行。三处同逻辑（`policy.py:108-117`、此处、以及死常量 `ACTIVE_HEADS`）→ 任一处改动都会漂移 | `bc_warmstart_rwpp.py:279-291` vs `policy.py:108-117` |
| 4-10 | `bc_warmstart_rwpp.py:272-297` | **同一指标算两遍**：`per_head` 循环（272-276）已经算过每头 argmax 匹配率，紧接着 285-291 又对同样的 argmax 再算一遍 active 加权。第二次循环完全可以用第一次的结果 | `bc_warmstart_rwpp.py:272-276` 与 `285-291` |
| 4-11 | `baseline_rwpp.py:93-113`、`bc_warmstart_rwpp.py:125-145`、`mock_engine.py:913-989` | **三份 focus-fire 脚本控制器**：都在实现"无敌人→攻击移动到地图中心；有敌人→打最低 HP/最近/质心"。`baseline_rwpp.py` 的注释甚至自称 baseline，与 `mock_engine.scripted_actions` 是同一套算法 | `baseline_rwpp.py:93-113`、`bc_warmstart_rwpp.py:125-145`、`mock_engine.py:913-989` |
| 4-12 | `baseline_rwpp.py:206-208` | **重复实现** `evaluate.summarize` 里的 `mean` 语义（且对 `None` 静默丢弃，与 `summarize` 行为不同） | `baseline_rwpp.py:206-208` vs `evaluate.py:156-159` |
| 4-13 | `train_rwpp.py:207-213` | **`except Exception: pass` 吞掉关停错误**：`finally` 里的 `trainer.vec.close()` 在 `RwppPPOTrainer.close()`→`vec.close()` 已经执行过一次的前提下**必然二次调用**；而 `VecRwppEnv.close()` 第二次会在已 shutdown 的线程池上 `map()` → 实测抛 `RuntimeError: cannot schedule new futures after shutdown`，被这里无声吞掉 | `train_rwpp.py:209-213`；`rwpp_ppo.py:81-86`；`vec_rwpp.py:151-156`；实测线程池二次 `map` 抛错 |
| 4-14 | `rwpp_ppo.py:83-86` | 同上模式的 `except Exception: pass  # shutdown must never mask training errors`——注释说"不掩盖训练错误"，实际把**所有**关停异常都掩盖了 | `rwpp_ppo.py:85-86` |
| 4-15 | `ppo.py:136-137` | `try: self._eval_thread.join(timeout=3.0) / except Exception: pass` —— `Thread.join` 只会抛 `RuntimeError`（不能 join 未启动线程），此处异常语义不明地吞掉 | `ppo.py:132-140` |
| 4-16 | `record.py:162` | **元数据与实际不符**：`meta["vision_radius"] = float(config.vision_radius if fog else args.vision_radius)`——`fog=False` 时把 CLI 的 `--vision-radius`（默认 260.0）写进 meta，而实际雾是**关闭**的。回放录制信息会误导后续分析 | `record.py:143-145, 160-162` |
| 4-17 | `vec_macro_micro.py:37-46` / `train_macro_micro.py:21-30` / `evaluate_macro_micro.py:31-40` | **同一套经济超参抄了三份且互不相同**：`base_income_per_second` 分别是 **4.0 / 5.0 / 5.0**；`evaluate_macro_micro` 独有 `extractor_t2_income_per_second=22.0` 而 `train_macro_micro` 没有。训练/评估超参不一致 | `vec_macro_micro.py:40`、`train_macro_micro.py:24`、`evaluate_macro_micro.py:34,36` |
| 4-18 | `train_macro_micro.py:9-11` | **import 风格不一致**：同目录其它 16 个文件都用相对 import（`from ..micro_config import`），此处用绝对 `from ironzero.micro_config import`，需要 `PYTHONPATH=src` 才能跑 | `train_macro_micro.py:9-11` vs `train.py:10-11`、`vec_macro_micro.py:16-19` |
| 4-19 | `bc_warmstart_rwpp.py:48`、`baseline_rwpp.py:49` | `BROADCAST = 4` 是"魔法常量"，注释还写 `matches MicroDuelConfig.action_nvec[1] - 1`——即硬编码 `groups=4`。一旦 25v25 想改分组数，两个脚本静默发出越界 group | `bc_warmstart_rwpp.py:48`、`baseline_rwpp.py:49` |
| 4-20 | `tests/test_training_pipeline.py:74` | **恒真断言（假测试）**：`self.assertTrue((out / "summary.json").exists() or True)`——`or True` 让该断言永真。而 `summary.json` 实际**由 `train.py:64` 写，不由 `PPOTrainer.train()` 写**，所以这条断言本来会失败，是用 `or True` 糊过去的 | `tests/test_training_pipeline.py:74`；`train.py:62-74` |

---

## 5. 半成品清单（缺什么 / 要补什么）

### 5-1 经济动作（`build_extractor` / `build_structure`）——**最严重的半成品**

- **已改的**：`micro_config.py:101-108`（kind 头 5→7）、`observations.py:306-311`（mask 7 值 + 资金门槛）、`rwpp_env.py:345-349`（真机解码新增两分支）。
- **没跟上的（清单）**：
  1. `env.py:170-203` `decode_actions` **没有 `build_*` 分支** → mock 引擎路径 100% `INVALID_MESSAGE`（实测 140/140 拒绝）。
  2. `mock_engine.py` **没有针对无效动作的惩罚**（真机有 `game.invalid_penalty`，mock 无）→ 策略采到这两个动作得到的 reward 与 `hold` **完全相等**（实测三者都是 `0.0`），学不到"不该点"。
  3. `IronZeroBridge.java:615-642` **只认 5 种 kind**（`move_group/attack_move/attack_target/stop/hold`），没有任何 `build_*`/`produce_unit` 处理 → 真机上下发即被 `rejected++`（`IronZeroBridge.java:645-647`）。
  4. `policy.py:114-116` `_active_matrix`：kind=5/6 时**只激活 head 0/1**，可 `rwpp_env.py:346` 的 `build_extractor` 用 `target_index`（head 4）、`:348-349` 的 `build_structure` 用 `gx/gy`（head 2/3）——**解码用了不参与梯度的头**，这两个动作永远学不会选址。缺的是给 kind 5/6 加上对应的 active head。
  5. `observations.py` 的全局特征块（`observations.py:47-62`，12 维）**不含 funds/income/resource_slots**：实测把 `funds=99999`、`income=42`、`resource_slots` 全改成 occupied 后，`encode_observation` **输出逐元素相同**。即策略**看不到自己有多少钱**，却要决定造不造。
- **要补的**：① `env.py` 加 `build_*` 解码分支；② mock 引擎加无效动作惩罚或直接 mask 掉；③ Bridge 实现 `build_structure`/`produce_unit`（`Unit`/builder API）；④ `_active_matrix` 补 kind 5/6 的 head；⑤ 观测里加 funds/income 维度（会动 `global_dim`，需同步所有 checkpoint 与文档）。

### 5-2 `curriculum.py` —— 空壳（见 §3.1），要补真的阶段调度并接进 `PPOTrainer.train()` 的 update 循环。

### 5-3 `bc_warmstart_rwpp.py --episodes` —— **解析了但从不使用**

- `parse_args()` 注册了 `--episodes`（`bc_warmstart_rwpp.py:68`），但全文件只有 `:21`(docstring)、`:68`(定义) 两处出现，`collect()` 的循环条件是 `while len(observations) < int(args.transitions)`（`:169`），**从不读 `args.episodes`**。
- 要补：要么按 episode 数终止（`while completed < args.episodes`），要么删掉该参数。

### 5-4 宏控状态不重置（`VecMacroMicroEnv`）

- `macro_controllers` 在构造时创建（`vec_macro_micro.py:57-66`），**在 `reset()`/episode 结束时都不调用 `reset_stats()`**；`reset_stats()` 本身全仓库无调用点。实测 `reset()` 后 `get_stats()` 已非零（`{'extractors_built': 1, 'factories_built': 1, ...}`），跨 episode 累积 → 任何"本局建造数"统计都是脏的。
- 要补：在 `vec_macro_micro.py:116`（`env.reset()` 处）与 `reset()` 里 `self.macro_controllers[index].reset_stats()`。

### 5-5 `VecMacroMicroEnv` 无动作宽度校验

- `step()`（`vec_macro_micro.py:84-96`）直接 `micro_actions[index]` 后 `env.decode_actions(...)`，**没有 `num_envs × 头数` 校验**（对比 `vec_env.py:69-72`、`vec_rwpp.py:90-92` 都有）。实测：传宽度 **7 被静默接受**（多余列被丢弃），传 3/4 则抛 `ValueError: not enough values to unpack (expected 4, got N)` —— 报错位置离真正原因很远。
- 要补：照抄 `vec_env.py:69-72` 的三行校验。

### 5-6 `evaluate_macro_micro` 未接入训练评估，且丢弃 checkpoint 契约

- `PPOTrainer.evaluate`（`ppo.py:154-169`）默认走 `evaluate_policy`（纯 5 类战术动作、**不跑 MacroController**）；`train_macro_micro.py:47-51` 构造 `PPOTrainer` 时**没有传 `eval_fn`** → 分层训练过程中每次 eval 用的都是"没有经济系统的环境"，与训练环境不是同一个 POMDP。
- `evaluate_macro_micro.py:106` `policy, _ = load_policy(...)` **把 checkpoint 的 micro_config 丢掉了**，随后用自己硬编码的默认 cfg 跑环境；若加载 25v25（768 维）checkpoint 会直接 obs 维度不匹配。
- 要补：`train_macro_micro` 传 `eval_fn=lambda t, policy=None: evaluate_macro_micro_policy(policy or t.policy, config=micro_cfg)`；`evaluate_macro_micro.main()` 用 `load_policy` 返回的 micro 而不是丢弃。

### 5-7 `--init-checkpoint` 的"兼容加载"实际丢掉了 kind 头

- `train_rwpp.py:284-319` `_load_compatible_weights` 只对 `trunk.0.weight`、`heads.4.weight`、`heads.4.bias` 做了形状适配，**`heads.0`（kind 头）既没有等形匹配（5≠7）也没有适配分支，直接 `continue` 跳过**。
- 实测加载 `runs/bc_ironzero25/checkpoint.pt`（旧 5 头）：`heads.0 in loaded? False`，`kind head changed by init? False`，且报告 `"initialized": true, "loaded_parameters": 14` —— **用户以为 warm start 成功，实际最重要的动作类型头是随机初始化**。
- 同类问题：`load_policy`（`ppo.py:615-616`）走严格 `load_state_dict`，实测遍历 `runs/**/checkpoint*.pt` 共 **158 个文件，143 个已无法加载**（`size mismatch for heads.0.weight: 5 vs 7`）。
- 要补：`heads.0` 增加 `min()` 截断适配（旧 5 类权重拷进新 7 类前 5 行），并在 `load_policy` 里对不匹配给出明确迁移报错/自动升级。

### 5-8 `PPOConfig` 的 `opponent` 与分层环境不联动

- `PPOTrainer` 把 `config.opponent` 传给 `VecMicroEnv`（`ppo.py:98-103`），但 `train_macro_micro` 自建 `VecMacroMicroEnv(opponent="scripted")`（`train_macro_micro.py:34`）硬编码，`PPOConfig` 里的 `opponent` 对该路径**完全无效**（且 `train_macro_micro` 也没暴露 CLI）。属"开关存在但两条路径无差别"。

---

## 6. 契约断裂清单

### 6-1【高】动作头 5→7 的契约变更只改了一半

**已改（7）**：`micro_config.py:101-108`、`observations.py:298,306,309,311`
**未改（仍硬编码 5 / 仍按 5 断言）**：

| 文件:行 | 内容 | 性质 |
| :--- | :--- | :--- |
| `src/ironzero/training/policy.py:6` | docstring `[kind(5), group(G), ...]` | 陈旧文档 |
| `src/ironzero/training/policy.py:38` | `nvec=(5, 4, 8, 8, 9)` 默认值 | **运行期陷阱**：省略 nvec 即 5 |
| `src/ironzero/training/ppo.py:61` | 注释 `# (T, B, 5)`（头数，正确但歧义） | 陈旧注释 |
| `src/ironzero/training/train_rwpp.py:18` | docstring `action heads (5, 5, 8, 8, 26)` | 陈旧文档 |
| `tests/test_env.py:40` | `masks["kind"].shape == (5,)` | 失败源 |
| `tests/test_env.py:48` | `flat[:5].all()` | 失败源 |
| `tests/test_env.py:56` | `len(message.masks.action_mask) == 5` | 失败源 |
| `tests/test_mock_engine.py:143` | `action_nvec[0] == 5` | 失败源 |
| `tests/test_rwpp_25v25.py:138` | `action_nvec == (5,5,8,8,26)` | 失败源 |
| `tests/test_rwpp_25v25.py:381` | `action_nvec == (5,5,8,8,26)` | 失败源 |
| `tests/test_rwpp_25v25.py:502` | `action_nvec == (5,5,8,8,26)` | 失败源 |
| `tests/test_rwpp_25v25.py:662` | `loaded_micro.action_nvec == (5,5,8,8,26)` | 失败源 |
| `tests/test_rwpp_25v25.py:665` | `loaded.nvec == (5,5,8,8,26)` | 失败源 |
| `tests/test_training_pipeline.py:39` | 手搓 `"kind": torch.ones(2, 5)` | 失败源（RuntimeError 5 vs 7） |
| `tests/test_training_pipeline.py:101` | `kinds.issubset(set(range(5)))` | 失败源 |

> 说明：`tests/test_training_pipeline.py:28`（`actions.shape == (1,5)`）、`tests/test_rwpp_env.py:583-585`（`np.zeros((2,5))`）**不是**错误——动作向量长度=头数=5 未变，只有 kind 取值域变了。已实测确认 `sample_masked_action` 返回的 5 元素向量仍是合法长度。

**11 个失败测试与本范围的关系**：11 条中有 **2 条**落在本范围的 `tests/test_training_pipeline.py`（`:39`、`:101`），另 9 条属 env/observations 范围但**根因同源**（`action_nvec` / `kind_mask` 5→7）。

### 6-2【高】分层训练环境 ↔ 评估环境 ↔ 真机 Bridge 三方动作空间不一致

| 环节 | 支持的动作 | 证据 |
| :--- | :--- | :--- |
| mock PPO 训练（`MicroDuelEnv`） | kind 0-6 都能"采"，但 5/6 被引擎拒 | `env.py:170-203`；`mock_engine.py:340-381` |
| 真机 PPO 训练（`RwppDuelEnv`） | 解码能产 5 种 + build_extractor/build_structure | `rwpp_env.py:342-362` |
| 真机 Bridge（真正执行的） | **只执行 5 种**：`move_group/attack_move/attack_target/stop/hold` | `IronZeroBridge.java:619-642` |
| 宏微观环境（`VecMacroMicroEnv`） | 宏控额外产 `produce_unit`/`upgrade_building` | `macro_controller.py:98-120` |
| `ACTION_KINDS` 枚举 | 10 种，其中 `produce_unit`/`upgrade_building`/`repair_target`（index 7/8/9）**永远不会被 kind 头选中**（头只有 7 个值） | `mock_engine.py:34-45` vs `micro_config.py:103` |

→ 同一条"策略输出"，在训练、评估、真机三处**执行语义不同**：mock 里 build 是空操作，真机里 build 是"被拒+罚分"（`invalid_penalty=0.05`，`rwpp_env.py:703`）。策略在 mock 学到的"点 build 没事"在真机是负奖励。

### 6-3【高】经济状态不进观测，但动作依赖经济状态

- `observations.py:47-62` 的 12 维全局块**无 funds/income/building_count/资源点占用**（实测改 funds 后 obs 逐元素相同）。
- 但动作正确性依赖它：`observations.py:308-311` 用 funds 决定 mask；`rwpp_env.py:346` 用 `target_index` 选资源点。
- 结论：**策略在"看不见钱"的观测上被要求做经济决策**——POMDP 断裂。且 mask 随 funds 变化而观测不变，等价于"用外部信息改了动作空间"，`log_prob` 归因会漂移。

### 6-4【高】`build_extractor` 的资源点索引在真机永远为 0

- `rwpp_env.py:346`：`slot_id = target_index % len(state.get("resource_slots", [{}]))`。
- 真机的 `observation_to_state`（`rwpp_env.py:296-317`）**不产出 `resource_slots` 键**，`Bridge` 的 obs 消息也**不含 `resource_slots`**（`IronZeroBridge.java:274-290` 只有 global_state/entities）。
- 实测：`resource_slots in state? False`，`target_index ∈ {0,1,2,3,7}` **全部映射为 `slot_id=0`**。→ "选哪个矿点"这个决策维度在真机是**退化的**，但 target 头仍占 26 个取值，白学。

### 6-5【中】`enable_economy` 开关：两条分支的观测完全相同

- 实测（同 seed、同 20 步动作序列、`opponent='idle'`）：`enable_economy=True` 与 `False` 的 **observation 序列逐元素相同**（`np.array_equal(...) == True`）。
- 唯一差异：`masks["kind"]`（True 版本 `[T,T,T,T,T,T,T]`，False 版本 `[T,T,T,T,T,F,F]`）与 funds/资源点数量。
- 由于这两个 kind 在 mock 里都是无效动作（§4-4），**该开关对训练行为的实际影响 ≈ 只有"多了两个空动作"**。
- 且 `rwpp_env.py` **完全不读 `enable_economy`**（`git grep enable_economy` 在 `src/ironzero` 只命中 `micro_config.py` 与 `mock_engine.py`）→ 真机路径无此开关。
- 附带风险：工作区已把默认值从 `False` 改成 `True`（`git diff`），**所有沿用默认值的离线训练/评估/回放都会多出两个空动作**，历史 baseline 数字不再可比。

### 6-6【中】`PPOConfig` 默认 rollout 与真机吞吐不匹配（未修复的已知问题，代码仍在）

- `ppo.py:26-30`：`rollout_steps=128 / minibatch_size=256`（离线量级）；`train_rwpp.py:41,45`：`8 / 64`。`train_macro_micro` 又用 `64 / 64`（`train_macro_micro.py:18,41`）。
- 后果：单 update 样本数 < minibatch → 实际每 epoch 只有 1 个 batch。属"同一语义参数三套默认值"的契约漂移。

### 6-7【中】`train_rwpp.py` docstring 与默认参数自相矛盾

- docstring 声称默认场景是 "Artillery 22 player units against 25 AI units"、`obs_dim 768`、`action heads (5, 5, 8, 8, 26)`（`train_rwpp.py:4-18`）。
- 实测 `RwppGameConfig()` 默认：`units_per_team=8, max_entities=8, target_slots=9 → obs_dim=364, nvec=(7,5,8,8,9)`。
- 即：文档写的默认值与代码默认值**完全不同**，且 `train_rwpp.py:64-65` 的 `--map-width 2700` 与 "8v8" 的默认组合本身就不自洽。

### 6-8【低】`Rollout.masks` / episode 字段在两套 vec 环境里不一致

- `VecMicroEnv` 写入 `episode["eval"]`（`vec_env.py:86-91`）；`VecRwppEnv` 硬编码 `"eval": None`（`vec_rwpp.py:124`）但多写 `tick/port/components`；`VecMacroMicroEnv` 写 `components` 不写 `tick`（`vec_macro_micro.py:108-114`）。
- `PPOTrainer.train` 只用 `return/result/components`（`ppo.py:365-372`），所以 `eval` 字段在真机路径永远是 None——不崩，但下游（前端曲线）拿到的是空。

---

## 7. 修复优先级建议

### P0 —— 会让"默认配置"产出错误模型，必须先定夺

1. **决定 5→7 是否真的上线**。若上线：把 `policy.py:38` 默认 `nvec` 改成从 `MicroDuelConfig` 派生（或必须显式传入）、给 kind 5/6 补 active heads（`policy.py:114-116`）、给 `env.py:170-203` 补 `build_*` 解码分支、给 mock 引擎补无效动作惩罚。若不上线：把 `micro_config.py:101-108` 回滚到 5，并把 `observations.py` 一起回滚——**当前"改一半 + 默认 `enable_economy=True`"是最差组合**。
2. **修 `heads.0` 的兼容加载**（`train_rwpp.py:284-319`）：最小 3 行——对 `heads.0.weight/bias` 加 `min()` 截断分支。否则 143/158 个 checkpoint 报错、warm start 静默丢头。
3. **补经济状态进观测**（或明确放弃经济动作，回到 P0-1 的回滚选项）。当前"看不到钱却要花钱"是纯噪声，且 `income` 分量在真机恒 0（`IronZeroBridge.java:281`）会误导奖励分解曲线。

### P1 —— 契约与测试收口（1~2 天）

4. 同步 15 处 5/7 断言（§6-1 表格），让 `pytest` 回到全绿；给 `micro_config.action_nvec` 加一个契约测试（`len(action_nvec)==5` 且 `action_nvec[0]==len(ACTION_KINDS_AVAILABLE)`）。
5. 恢复 `observations.py:313-317` 被删掉的 `group_id is None` 兜底（现在 `# ... (keep existing group logic)` 是假的），修 `tests/test_rwpp_25v25.py:178`。
6. 给 `VecMacroMicroEnv.step` 加动作宽度校验（`vec_macro_micro.py:84-86`）。
7. 修 `bc_warmstart_rwpp.py --episodes`（要么接线要么删），修 `_grid(size)` 的无用参数。

### P2 —— 清理（低风险，可批量做）

8. 删 `training/curriculum.py`（死代码）或在 README 明确它未实现；同步 README:24/218 的"课程学习"措辞。
9. 删 `policy.py:23-29 ACTIVE_HEADS`、`policy.py:151-153 greedy_action`、`ppo.py:11 field`、`ppo.py:110 history`、`ppo.py:247-251` 的 `last_masks`、`evaluate_cli.py:29-30` 不可达分支、`micro_config.py:87 resource_slots_count`（或接线到 `mock_engine.py:188-193`）。
10. 合并 `bc_warmstart_rwpp.py` 的重复 per-head 计算（272-297）与手抄 active matrix（279-291），改为调用 `policy._active_matrix`。
11. 删 `ppo.py:580-589` 的 `checkpoint_latest.pt` 重复写盘，或让它承载差异信息（如 optimizer state）。
12. 统一三处 `base_income_per_second`（4.0/5.0/5.0）到一处常量；把 `idle_funds_penalty_coef` 的 1500 阈值提成 config 字段。
13. 修 `record.py:162` 的 `vision_radius` 元数据语义；修 `tests/test_training_pipeline.py:74` 的 `or True` 假断言。
14. 把 `rwpp_env.py:702` 拆成多行，`bonus_build` 的 5.0 换成 config 字段；`income` 分量在 Bridge 上报真实 income 之前直接删掉（避免恒 0 分量污染奖励曲线）。

---

## 附录 A：本报告涉及的 11 个失败测试归因

| # | 测试 | 失败断言位置 | 根因 | 归属 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | `test_env.py::test_mask_shapes_and_minimum_validity` | `test_env.py:40` | `kind` 5→7 | env 范围（同源） |
| 2 | `test_env.py::test_masks_serialize_into_protocol_message` | `test_env.py:56` | 同上 | env 范围（同源） |
| 3 | `test_env.py::test_masked_random_actions_are_legal` | `test_env.py:146` | 随机采到 `build_structure` | **本报告 §4-4**（同源） |
| 4 | `test_mock_engine.py::test_custom_config_keeps_shapes_consistent` | `test_mock_engine.py:143` | `action_nvec[0]` 5→7 | mock 范围（同源） |
| 5 | `test_rwpp_25v25.py::test_25v25_observation_and_action_contract` | `:138` | 5→7 | rwpp 范围（同源） |
| 6 | `test_rwpp_25v25.py::test_masks_without_group_id_expose_all_groups` | `:178` | `observations.py:313-317` 兜底被删 | **本报告 §4-2** |
| 7 | `test_rwpp_25v25.py::test_game_config_builds_25v25_terrain_micro` | `:381` | 5→7 | 同源 |
| 8 | `test_rwpp_25v25.py::test_ironzero25_flags_build_the_25v25_terrain_contract` | `:502` | 5→7（`train_rwpp.build_game`） | **本范围 §6-7 相关** |
| 9 | `test_rwpp_25v25.py::test_terrain_checkpoint_rebuilds_the_768_dim_policy` | `:662` | 5→7（`load_policy`） | **本范围 §5-7 相关** |
| 10 | `test_training_pipeline.py::test_masked_logits_cannot_select_masked_group` | `policy.py:73`（RuntimeError 5 vs 7） | 测试手搓 5 值 mask | **本范围 §3.3** |
| 11 | `test_training_pipeline.py::test_random_legal_actions_cover_all_kinds` | `:101` | `range(5)` 与 7 值 mask 冲突 | **本范围 §6-1** |

## 附录 B：可直接复现的关键实验

```powershell
$env:PYTHONPATH = "src"

# 1) 全量测试基线
python -m pytest -q                     # => 11 failed, 190 passed

# 2) 经济动作在 mock 里 100% 被拒
python -c "import numpy as np; from ironzero.env import MicroDuelEnv
env=MicroDuelEnv(seed=4); o,i=env.reset(); o,r,t,tr,info=env.step(np.array([5,4,3,3,0]))
print(info['receipts'])"                # => rejected_reason: INVALID_MESSAGE

# 3) kind=5/6 的 active heads 只有 (0,1)
python -c "from ironzero.training.policy import MaskedActorCritic as M
from ironzero.micro_config import DEFAULT_CONFIG as C; import torch
p=M(C.flat_obs_dim, nvec=C.action_nvec, hidden=8)
print([p._active_matrix(torch.tensor([k]), torch.device('cpu'))[0].nonzero().flatten().tolist() for k in range(7)])"

# 4) 旧 checkpoint 143/158 无法加载
python -c "import glob; from ironzero.training.ppo import load_policy
ps=sorted(glob.glob('runs/**/checkpoint*.pt', recursive=True)); bad=0
for p in ps:
    try: load_policy(p, device='cpu')
    except Exception: bad+=1
print(len(ps), bad)"

# 5) enable_economy 两分支 obs 完全相同
python -c "import numpy as np; from ironzero.micro_config import MicroDuelConfig as M
from ironzero.env import MicroDuelEnv as E
def run(f):
    e=E(config=M(enable_economy=f, episode_ticks=1200), opponent='idle', seed=0); o,i=e.reset()
    seq=[o.copy()]
    for _ in range(20):
        o,r,t,tr,i=e.step(np.array([4,4,3,3,0])); seq.append(o.copy())
        if t or tr: break
    return np.array(seq)
print(np.array_equal(run(True), run(False)))"   # => True

# 6) 工程 5→7 之后 load_policy 对旧 5 头权重必定失败
python -c "from ironzero.training.ppo import load_policy
load_policy('runs/bc_ironzero25/checkpoint.pt', device='cpu')"
```
