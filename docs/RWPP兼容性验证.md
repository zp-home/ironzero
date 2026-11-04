# RWPP 兼容性验证

> 文档状态：M1 验证手册
> 作者：周培智
> 目标：确认当前 RWPP 版本、Java 版本和游戏资源能够启动无头服务端并正常退出。

---

## 1. 验证目标

M1 只验证运行环境，不训练模型：

```text
Java 25
  → RWPP 服务端
  → 原版 assets
  → 固定地图
  → --ticks 180
  → 正常退出
  → 端口释放
```

通过后才能进入只读 Bridge 和动作桥。

---

## 2. 前置条件

需要一个受控的 RWPP 运行目录，目录结构至少包含：

```text
<RWPP_RUNTIME_DIR>/
  bin/
    rwpp-server.bat
  assets/
  extensions/
```

注意：

- `assets/` 来自本机原版游戏，不能提交到 Git。
- `extensions/` 只放受信任扩展。
- RWPP 版本必须记录为具体 commit 或 release。
- Java 25 必须可执行。

---

## 3. 环境变量

推荐在本地 PowerShell 会话中设置：

```powershell
$env:RWPP_RUNTIME_DIR = "D:\rwpp-runtime"
$env:JAVA_HOME = "D:\jdks\jdk-25"
```

不要把包含本机路径的配置提交到仓库。

---

## 4. 检查环境

```powershell
.\scripts\check_environment.ps1
```

也可以通过参数指定：

```powershell
.\scripts\check_environment.ps1 `
  -RuntimeDir "D:\rwpp-runtime" `
  -JavaHome "D:\jdks\jdk-25" `
  -Port 5123
```

必须全部通过：

- Java 可执行。
- Java 主版本至少 25。
- RWPP 运行目录存在。
- `rwpp-server.bat` 存在。
- `assets/` 存在。
- `extensions/` 存在。
- 回环端口可用。

---

## 5. 执行冒烟测试

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir "D:\rwpp-runtime" `
  -JavaHome "D:\jdks\jdk-25" `
  -Port 5123 `
  -Ticks 180 `
  -TimeoutSeconds 180
```

脚本会：

1. 启动 `rwpp-server.bat`。
2. 使用 `--no-console` 和 `--ticks 180`。
3. 将 stdout/stderr 写入 `artifacts/smoke/<timestamp>/`。
4. 等待进程退出。
5. 检查退出码。
6. 扫描明显的异常日志。
7. 检查端口是否释放。
8. 写出 `report.json`。

---

## 6. 通过标准

- [ ] 进程退出码为 0。
- [ ] 日志没有致命异常。
- [ ] tick 正常推进到指定上限。
- [ ] 固定地图加载成功。
- [ ] 端口在退出后释放。
- [ ] 没有残留 Java 子进程。
- [ ] 报告文件已生成。

建议额外记录：

```text
启动耗时
地图加载耗时
内存峰值
RWPP commit
Java 版本
游戏资源版本
```

---

## 7. 常见失败

| 现象 | 优先检查 |
|---|---|
| Java 版本过低 | 安装 JDK 25 并设置 `JAVA_HOME` |
| 找不到 assets | 确认运行目录下存在原版 `assets/` |
| 地图加载失败 | 检查地图路径、地图版本和模组 |
| 端口占用 | 更换端口并检查旧进程 |
| 启动后无输出 | 检查 stderr、Java 参数和运行目录 |
| 退出后进程残留 | 检查子进程树和脚本启动方式 |
| 模组不匹配 | 固定 RWPP、游戏资源和模组版本 |

---

## 8. 当前状态

当前工作站已确认：

- Python 3.12 可用。
- Java 25 便携运行时可位于 `.runtime/jdk25/`。
- 尚未发现可用的 RWPP 运行目录和游戏 assets。

因此 M1 的真实服务端启动尚未通过。必须先提供 RWPP 运行目录、Java 25 和合法游戏资源，才能继续完成兼容性验证。

在缺少真实资源时，可以使用 mock runtime 验证脚本本身：

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir ".\tests\fixtures\mock_rwpp_runtime" `
  -JavaHome ".\.runtime\jdk25\jdk-25.0.5.0.2+1" `
  -Ticks 180
```

mock 通过只说明脚本和进程清理逻辑正常，不代表 RWPP 兼容。
