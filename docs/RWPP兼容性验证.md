<!-- SPDX-License-Identifier: CC-BY-4.0 -->
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
- 已找到一份 1.15 PC 客户端，Steam App ID 为 `647960`。
- 客户端自报 `Game Version: 1.15`、`Game Code: 176`、`Build Number: #28`。
- 客户端包含完整 `assets/`，约 21.4 MiB。
- 客户端自带 `game-lib.jar` 和 Java 13 运行时；RWPP 服务端仍使用项目自己的 Java 25。
- RWPP 源码已下载到纯英文路径 `.rwrl-runtime/RWPP-main`，用于避免 Android Gradle 插件拒绝中文路径。
- Gradle 下载已显式配置本地 HTTP/HTTPS 代理，不再依赖直连。

M1 的服务端构建和真实启动尚未通过。当前构建已经进入 `:rwpp-api:compileKotlinJvm`，说明 Gradle、Java 25 和主源码解析基本可用。

## 9. 已发现的构建阻塞

### 9.1 中文路径

Android Gradle Plugin 会拒绝非 ASCII 项目路径。RWPP 构建目录必须放在英文路径，例如：

```text
D:\qwq\.rwrl-runtime\RWPP-main
```

不要直接使用包含中文的仓库目录作为构建根目录。

### 9.2 Gradle 代理

Windows 系统代理不会自动传给 Java/Gradle。需要显式设置，例如：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:JAVA_TOOL_OPTIONS = "-Dhttp.proxyHost=127.0.0.1 -Dhttp.proxyPort=7897 -Dhttps.proxyHost=127.0.0.1 -Dhttps.proxyPort=7897"
$env:GRADLE_OPTS = $env:JAVA_TOOL_OPTIONS
```

端口和地址应按本机代理实际配置修改。

### 9.3 JetBrains Runtime 工具链

RWPP 仓库的 `gradle/gradle-daemon-jvm.properties` 默认要求 JetBrains Runtime 25。若使用 Microsoft OpenJDK 25，需要在本机构建副本中使用兼容的工具链配置，或安装对应 JBR。不要把这类本机适配提交到上游仓库。

### 9.4 `AI.aiUpdatePaused` 映射差异

当前 RWPP `main` 的 AI Controller 代码直接引用 `AI.aiUpdatePaused`，但 `game-lib:v2-1.15` 映射中不存在该字段，导致 API 编译失败。

这意味着“同为 1.15”并不等于接口完全一致。后续必须：

- 固定 RWPP commit。
- 固定 `game-lib` 映射版本。
- 记录游戏 Build Code。
- 对 AI Controller 字段做能力探测。
- 缺少字段时只允许 `AUGMENT_NATIVE`，不要强行启用 `REPLACE_NATIVE`。

本地可用于验证的临时兼容方案是反射读取字段，但这不应替代正式版本适配。

### 9.5 `rwml-desktop:1.0.2` 依赖不可匿名获取

服务端依赖 `io.github.rwpp-team:rwml-desktop:1.0.2`，JitPack 当前返回 `401 Unauthorized`。RWML 仓库本身是公开的，因此下一步选项是：

1. 从 RWML 公开源码构建并发布到本机 Maven 仓库。
2. 使用已验证的本地构件。
3. 等待上游提供可匿名下载的正式构件。

在依赖可用前，`rwpp-server` 不能完成构建。

## 10. 当前结论

M1 尚未通过真实运行验证。已经确认的不是“RWPP 不能跑”，而是“当前 RWPP main、映射库、工具链和依赖发布状态需要固定版本后才能跑”。

下一步应优先解决 `rwml-desktop` 本地构建，然后用同一份 1.15 客户端 assets 启动服务端。构建成功前不要开始 RL 训练。

在缺少真实资源时，可以使用 mock runtime 验证脚本本身：

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir ".\tests\fixtures\mock_rwpp_runtime" `
  -JavaHome ".\.runtime\jdk25\jdk-25.0.5.0.2+1" `
  -Ticks 180
```

mock 通过只说明脚本和进程清理逻辑正常，不代表 RWPP 兼容。
