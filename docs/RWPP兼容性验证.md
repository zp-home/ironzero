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

## 8. 已验证兼容矩阵

2026-11-04 的 M1 实测组合：

| 组件 | 已验证版本 |
|---|---|
| 游戏 | Rusted Warfare 1.15，Game Code `176`，Build `#28`，Steam App `647960` |
| 游戏资源 | 完整 `assets/`，约 21.4 MiB |
| 游戏映射库 | `io.github.rwpp-team:game-lib:v3-1.15` |
| 映射 JAR SHA-256 | `1C7B864C3B2E78C30903BFB94F7FF4C89B31230D04C8321734D66F025522B230` |
| RWPP | `0c8325fb4eeb98ceb0d93e279c61d05f784538f4`（2026-10-23） |
| RWML | `1.0.2`，commit `c5b7c8dc57da25d667f1536a7badb6d3e21182b6` |
| RWML Desktop | 从 RWML `1.0.2` 源码本地构建 |
| Java | Microsoft OpenJDK `25.0.5.0.2` |
| Gradle（RWPP） | `9.6.1` |
| Gradle（RWML） | `9.6.0` |
| 默认测试地图 | `maps/skirmish/[p2]Lake (2p).tmx` |
| 地图 SHA-256 | `251273E30DEBFF8BB3927A99A6D668EE0B5D03D936EA991DE8866F638358FA6E` |

这组版本已经完成真实无头启动，不能只根据游戏显示版本“1.15”推断其他构建也兼容。

## 9. 版本差异与兼容补丁

### 9.1 `game-lib` 必须使用 v3 映射

`game-lib:v2-1.15` 中 `AI` 没有 `aiUpdatePaused`，而当前 RWPP AI Controller 需要该字段。`game-lib:v3-1.15` 已包含：

```text
public boolean aiUpdatePaused;
```

因此本机构建使用：

```toml
game-lib = "v3-1.15"
```

这是版本对齐，不是功能改写。

### 9.2 `LocaleEngine` 方法改名

`v3-1.15` 将旧映射方法：

```text
LocaleEngine.b(String)
```

命名为：

```text
LocaleEngine.translateMapName(String)
```

RWPP main 中对应调用需要改为：

```kotlin
LocaleEngine.translateMapName(plainName)
```

这是 API 名称适配，不改变地图名称翻译逻辑。

### 9.3 RWML Desktop 本地构建

`rwml-desktop:1.0.2` 当前没有可从 Maven Central 匿名获取的 JAR，JitPack 返回 `401`。

已验证的处理方式：

1. 从公开的 RWML `1.0.2` 源码构建。
2. 使用 `game-lib:v3-1.15` 的本地 Maven 布局。
3. 执行：

```powershell
.\gradlew.bat :desktop:publishToMavenLocal --no-daemon
```

生成：

```text
C:\Users\<user>\.m2\repository\io\github\rwpp-team\rwml-desktop\1.0.2\rwml-desktop-1.0.2.jar
```

本地构建的 JAR SHA-256：

```text
B205278494D17CBAF5E50391672CCFE097D3D7C5D1A461D92C8BD988BD97AB7B
```

RWPP 的 `settings.gradle.kts` 已允许从 `mavenLocal()` 解析 `rwml-desktop`。

### 9.4 构建环境要求

- RWPP 和 RWML 构建目录必须使用纯英文路径。
- Windows 系统代理需要显式传给 Java/Gradle。
- RWPP 仓库默认要求 JetBrains Runtime 25；使用 Microsoft OpenJDK 时，本机构建副本需调整工具链供应商标记。
- 不要把本机代理、路径和工具链适配提交到上游仓库。

## 10. M1 冒烟结果

实际运行：

```text
[rwml] 100% 2 loaded, 0 disabled, 0 skipped
Version:RAW - v1.15
network:server started
[rwml-server] Listening on port 5123; map maps/skirmish/[p2]Lake (2p).tmx
[rwpp-server] Completed 180 frames
```

结果：

| 项目 | 结果 |
|---|---|
| Java 环境检查 | 通过 |
| RWPP runtime 目录 | 通过 |
| assets / extensions | 通过 |
| 服务端启动 | 通过 |
| RWML 扩展加载 | 2 loaded |
| 地图加载 | 通过 |
| 180 tick 执行 | 通过 |
| 进程退出码 | 0 |
| 端口释放 | 通过 |
| 明显异常日志 | 未发现 |

测试报告：

```text
artifacts/smoke/20261104-160400/report.json
```

## 10.1 一键可重复构建

以下脚本已经从全新构建目录完整验证：

```powershell
.\scripts\build_rwpp_compat.ps1 `
  -GameDir "D:\path\to\Rusted Warfare" `
  -BuildRoot "$env:LOCALAPPDATA\RWRL" `
  -ProxyUrl "http://127.0.0.1:7897"
```

它执行：

```text
下载固定 RWPP/RWML/game-lib
  -> 应用兼容补丁
  -> 构建 RWML Desktop
  -> 构建 RWPP Server
  -> 组装独立 runtime
  -> 运行环境检查
  -> 执行 180 tick 冒烟测试
```

全新目录验证结果：

```text
BUILD SUCCESSFUL
Environment check passed
RWPP smoke test passed
exit_code: 0
port_available_after_exit: true
```

## 11. 自维护兼容策略

可以作为开源项目自行维护兼容层，但必须把补丁限制在“API 适配”而不是修改游戏行为。

建议维护：

```text
vendor/rwpp/
  pinned-versions.toml
  patches/
    0001-game-lib-v3.patch
    0002-locale-engine-translate-map-name.patch
  build-runtime.ps1
  smoke-test.ps1
```

原则：

- 固定上游 commit，不跟随 main 自动构建。
- 每个补丁记录目标版本、原因、影响和测试结果。
- 只做字段存在性、方法改名和依赖获取适配。
- 不修改单位数值、战斗逻辑、奖励机制或可见性规则。
- 升级 RWPP、RWML、game-lib 或游戏 Build 后重新跑完整 M1。
- 上游修复后删除本地补丁，不长期分叉。

当前两处源码修改都属于可提交上游的通用兼容修复：

1. `game-lib` 依赖从 `v2-1.15` 升级到 `v3-1.15`。
2. `LocaleEngine.b` 更新为 `translateMapName`。

如果上游接受，优先提交 PR 并删除本地补丁。

固定版本清单见 `vendor/rwpp/pinned-versions.json`，可重复构建入口见 `scripts/build_rwpp_compat.ps1`。

## 12. 结论

M1 已经通过真实服务端启动验证。之前的两个问题都是版本/发布状态问题，不是游戏引擎不可接入：

- `AI.aiUpdatePaused`：升级到 `game-lib:v3-1.15` 解决。
- `rwml-desktop:1.0.2`：从公开 RWML 源码本地构建并发布到本机 Maven 解决。

下一阶段可以进入 M2：只读 Bridge、状态快照和固定 episode 生命周期。

在缺少真实资源时，可以使用 mock runtime 验证脚本本身：

```powershell
.\scripts\smoke_rwpp.ps1 `
  -RuntimeDir ".\tests\fixtures\mock_rwpp_runtime" `
  -JavaHome ".\.runtime\jdk25\jdk-25.0.5.0.2+1" `
  -Ticks 180
```

mock 通过只说明脚本和进程清理逻辑正常，不代表 RWPP 兼容。
