# `ironzero.rwpp` —— 真机 RWPP 环境分层说明

本包由原先的单文件 `src/ironzero/rwpp_env.py`（1051 行）拆分而来。
**只搬结构、不改行为**：所有数值常量、奖励系数、分支判断、正则、提示文案逐字节保留，
唯一的改动是「把代码搬到更内聚的文件里」+「新增 re-export」。

旧 import 路径 `ironzero.rwpp_env` 仍然可用，它是本包的**薄 shim**（只做 re-export）。

---

## 1. 模块边界

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| `config.py` | 149 | `RwppGameConfig` + 默认常量（`REPO_ROOT`/`DEFAULT_RUNTIME`/`DEFAULT_JAVA_HOME`/`DEFAULT_MAP`/`DEFAULT_MAP_PX_*`/`DECISION_TICKS`）。纯数据 + `micro_config()` 派生，无 I/O。 |
| `errors.py` | 11 | `RwppEnvError(RuntimeError)`：进程/桥连接失败的统一异常类型。 |
| `process.py` | 120 | `_ProcLike`、`launch_rwpp_process()`、`effective_frame_ms()`。全仓库唯一 spawn `rwpp-server.bat` 的地方，`import subprocess` 在这里。 |
| `transport.py` | 54 | `_LineReader`：socket 上的 NDJSON 行协议读取（带缓冲 + `closed` 标志）。 |
| `decoding.py` | 169 | `observation_to_state()`（桥 obs → 编码器 state）、`decode_action()`（策略动作 → `ironzero/1` step 指令）、`_hold_action()`、`merge_done_message()`（`done` 消息并入终局 state）。 |
| `vision.py` | 38 | `apply_vision()`：观测级战争迷雾，按 `vision_radius` 裁掉视野外敌人。 |
| `reward.py` | 108 | `hp_potential()` / `hp_fractions()` / `army_hp_reference()` / `army_distance_norm()`：奖励塑形用到的**势函数**（纯函数）。 |
| `reward_shaping.py` | 149 | `compute_step_shaping()`：单步奖励分解 + 终局裁决。原 `step()` 中间那一大段，原样抽出。 |
| `strategy.py` | 89 | 可选战略层（LLM/规则）：`strategy_summary()` / `refresh_strategy()` / `strategy_potential()`。**只影响奖励**，从不产出底层动作。 |
| `spaces.py` | 48 | `_BoxSpace` / `_MultiDiscreteSpace`：仅形状描述，供 `observation_space`/`action_space` 暴露。 |
| `stepping.py` | 145 | 单步机械部件：`send_step()`（解码+发送 step 消息）、`absorb_message()`（回执并入 state/masks）、`build_step_info()`（打包 `info`）、`strategy_payload()`、`env_fracs()`。 |
| `session.py` | 111 | 桥连接生命周期：`start_game()`（拉起进程 + 连端口）、`await_observation()`（读到下一条 obs/done）、`shutdown_game()`（发 shutdown、关 socket、kill）。 |
| `env.py` | 290 | `RwppDuelEnv`：`reset`/`step`/`_observed`/`_refresh_strategy`/`action_masks`/`close`，以及转发到 `stepping`/`session`/`strategy` 的同名私有方法。 |
| `__init__.py` | 71 | 包级 re-export + `__all__`。 |

硬约束：单文件 ≤300 行，最大文件 `env.py` 290 行，全部低于 450 硬上限。

---

## 2. 分层与数据流

```
        ┌─────────────────────────────────────────────────────────┐
        │ config.RwppGameConfig  ──micro_config()──▶ MicroDuelConfig │
        └─────────────────────────────────────────────────────────┘
                              │
   reset()                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ session.start_game(env)            ← env._start_game() 转发   │
   │   process.launch_rwpp_process(game, port)                    │
   │     → subprocess.Popen(rwpp-server.bat --bridge-port …)      │
   │   socket.create_connection(127.0.0.1:port)                   │
   │     → transport._LineReader(sock)                            │
   └──────────────────────────────────────────────────────────────┘
                              │  NDJSON (hello / obs / receipt / done)
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ session.await_observation(env, timeout)  ← 同名私有方法转发    │
   │   obs     → return                                           │
   │   receipt → env.receipts.append(...)                         │
   │   done    → env.done = True; return                          │
   │   hello   → 跳过                                              │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
   decoding.observation_to_state(msg)   →  state（真实全量）
   vision.apply_vision(state, radius)   →  obs_state（策略可见）
   observations.build_masks(obs_state)  →  masks
   observations.encode_observation(...) →  flat obs 向量
```

`step()` 的一步完整链路：

```
action ──stepping.send_step(env)──▶ decoding.decode_action(obs_state) ──▶ actions[]
                                        └── json ──▶ socket.sendall
                                                                  │
                                                     下一条 obs / done
                                                                  ▼
                     stepping.absorb_message(env, msg)
                       ├─ done → decoding.merge_done_message → state
                       ├─ 否则 → decoding.observation_to_state → state
                       ├─ env.obs_state = vision.apply_vision(...)
                       └─ env.masks = observations.build_masks(obs_state)
                                                                  │
                       ├─ reward.army_distance_norm(state)        → distance
                       ├─ reward.hp_potential(state, refs)        → potential
                       ├─ strategy.refresh_strategy / potential   → strategy_component
                                                                  ▼
                    reward_shaping.compute_step_shaping(...)
                      → components{} + reward + terminated + truncated + last_result
                                                                  ▼
                    stepping.build_step_info(env, ctx) → info
                    obs, reward, terminated, truncated, info
```

**两个 state 的区别（很重要）**：
`self.state` 是桥上报的**真实全量**状态，用于终局判定 / HP 归一 / 奖励；
`self.obs_state` 是迷雾过滤后的**策略可见**状态，用于构建观测与掩码。
动作解码也走 `obs_state`，否则雾下 `target_index` 会指向看不见的敌人。

---

## 3. `subprocess` monkeypatch 契约（关键约束）

`tests/test_rwpp_25v25.py` 里有：

```python
with mock.patch("ironzero.rwpp_env.subprocess.Popen") as popen:
    launch_rwpp_process(game, 5425)
```

`mock.patch` 的语义是：解析 `ironzero.rwpp_env` → 取属性 `subprocess` → 在其中把 `Popen`
替换成 Mock。由于 `subprocess` 是**模块单例**，`ironzero.rwpp_env.subprocess` 与
`ironzero.rwpp.process.subprocess` 是**同一个对象**，替换发生在该对象上，所以
`process.launch_rwpp_process()` 里 `subprocess.Popen(...)` 拿到的就是 Mock。

因此有两处必须同时成立：

1. `src/ironzero/rwpp_env.py`（shim）里必须有 `import subprocess  # noqa: F401`
   ——否则 `patch` 解析属性路径时直接 `AttributeError`；
2. 真正的 `subprocess.Popen` 调用必须在 **`rwpp/process.py`**（它自己 `import subprocess`）。

已实测验证：patch 后 `popen.called is True`，且 `rwpp.process.subprocess.Popen is popen`
（见交付说明）。若未来有人把 `import subprocess` 从 shim 删掉，测试会立刻红。

---

## 4. 奖励计算顺序（不可变）

`reward_shaping.compute_step_shaping()` 里 `components` 的**键顺序**就是历史顺序：

```
potential, engagement, strategy, kills, losses, income, bonus_build,
invalid, idle, terminal, timeout
```

原因：`reward = float(sum(components.values()))` 是浮点累加，**顺序影响末位**，
`tests/test_rwpp_env.py` 的 reward 断言可能敏感。同理：

- `episode_components` 的累加发生在**终局裁决之前**；
- 截断裁决改写 `components["terminal"]` 之后会**重新** `sum()` 一次，并单独补记
  `episode_components["terminal"]` —— 这两点都原样保留。

拆分时用 4000 组随机输入做了「旧内联代码 vs 新函数」的逐位对比（`==` 与 `repr(reward)`），
`components`、`reward`、`episode_components`、`terminated`/`truncated`/`last_result` 全部一致。

---

## 5. 兼容 shim

`src/ironzero/rwpp_env.py` 只做 re-export，导出：

`DEFAULT_RUNTIME`, `DEFAULT_JAVA_HOME`, `DECISION_TICKS`, `DEFAULT_MAP`, `DEFAULT_MAP_NAME`,
`DEFAULT_MAP_PX_W`, `DEFAULT_MAP_PX_H`, `REPO_ROOT`, `RwppDuelEnv`, `RwppEnvError`,
`RwppGameConfig`, `apply_vision`, `army_hp_reference`, `army_distance_norm`, `decode_action`,
`hp_fractions`, `hp_potential`, `launch_rwpp_process`, `observation_to_state`,
`effective_frame_ms`, `_LineReader`, `_ProcLike`, `_BoxSpace`, `_MultiDiscreteSpace`,
`compute_step_shaping`, `StepShaping`, `_hold_action`, `FRAME_MS_WARNED`。

下游仍按旧路径 import 的地方（无需改动）：`scripts/{baseline,end,play,eval}_rwpp*.py`、
`scripts/bc_warmstart_rwpp.py`、`src/ironzero/training/{vec_rwpp,evaluate_rwpp,rwpp_ppo,train_rwpp}.py`、
`tests/test_rwpp_{env,25v25,scripts}.py`。

`RwppDuelEnv` 的历史私有方法（`_observed`/`_refresh_strategy`/`_strategy_potential`/
`_strategy_summary`/`_start_game`/`_await_observation`/`_shutdown_game`）都仍然存在，
只是实现转发到 `session.py` / `strategy.py` / `stepping.py` 里的同名自由函数。

---

## 6. 自测

```powershell
python -m pytest -q tests/test_rwpp_env.py tests/test_rwpp_25v25.py tests/test_rwpp_scripts.py
python -m pytest -q          # 209 passed
python -X utf8 -c "import sys; sys.path.insert(0,'src'); import ironzero.rwpp_env as m; assert m.subprocess; from ironzero.rwpp_env import RwppDuelEnv, RwppGameConfig, RwppEnvError, launch_rwpp_process; print('ok')"
```
