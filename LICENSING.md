<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 许可证说明

Copyright 2026 周培智

本项目采用分层许可证。不同目录和文件类型适用不同许可证，第三方组件继续适用其原许可证。

## 1. 许可证范围

| 路径或类型 | 许可证 | 说明 |
|---|---|---|
| `src/` | Apache-2.0 | Python 协议与运行时代码 |
| `tests/` | Apache-2.0 | 测试代码和 mock fixture |
| `tools/` | Apache-2.0 | 生成、校验和辅助工具 |
| `scripts/` | Apache-2.0 | PowerShell 脚本 |
| `schemas/` | Apache-2.0 | 机器可读协议 schema |
| `vendor/rwpp/pinned-versions.json` | Apache-2.0 | 第三方版本固定清单 |
| `pyproject.toml` | Apache-2.0 | 项目构建配置 |
| `site/`（除 README） | Apache-2.0 | 静态展示页 HTML、CSS、JS 和 SVG |
| `site/README.md` | CC BY 4.0 | 展示页使用说明 |
| `docs/` | CC BY 4.0 | 架构文档、规格和检查表 |
| 根目录 Markdown 文档 | CC BY 4.0 | README、总体架构和配套架构文档 |

完整文本：

- 代码许可证：[LICENSE](./LICENSE)
- 文档许可证：[docs/LICENSE](./docs/LICENSE)

## 2. 第三方组件

项目自身许可证不覆盖第三方依赖。第三方依赖、参考项目和可选集成保留各自许可证。

详见：

- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)
- [NOTICE](./NOTICE)

## 3. RWPP 与 AGPLv3

当前仓库**包含一个 RWPP 服务器扩展**：`src/ironzero_bridge/java/ironzero/IronZeroBridge.java`。
它 `implements io.github.rwpp.server.ServerExtension`，并通过 `io.github.rwpp.api.*` 与
`com.corrodinggames.rts.*`（game-lib）交互，在运行时加载进 RWPP 宿主进程。

- 该扩展**不复制** RWPP 上游源码（见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)），
  但它依赖并链接 RWPP / game-lib，属于衍生作品。
- 因此 **`src/ironzero_bridge/` 不适用 Apache-2.0**，单独按 **AGPL-3.0** 分发；其分发需满足
  AGPLv3 的对应源代码与网络使用义务。仓库其余部分（`src/ironzero/` 等）保持 Apache-2.0。
- 发布前请核对 RWPP 的 LICENSE、具体 API 使用方式与实际分发方式。

当前实际的许可证边界：

```text
src/ironzero/                  Apache-2.0
src/ironzero_bridge/           AGPL-3.0（RWPP 扩展，链接 RWPP / game-lib）
web/  scripts/  tools/  tests/ Apache-2.0
schemas/  configs/  docs/  site/  Apache-2.0
vendor/rwpp/                   仅锁定第三方版本信息，不包含其代码
```

发布前应再次核对 RWPP 的 LICENSE、具体 API 使用方式和分发方式。

## 4. 游戏本体与游戏资源

本项目不包含、也不授权分发：

- 《铁锈战争》游戏本体。
- 原版 `assets/`。
- 反编译产物。
- 未授权的贴图、音频、地图或模组。

这些内容的使用受游戏 EULA、版权法和其他适用条款约束。

展示页通过 Steam 官方 CDN 远程引用《铁锈战争》宣传截图。这些图片不属于本项目代码或文档许可证范围，仓库也不保存这些图片副本。详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

## 5. 模型、数据和运行产物

当前仓库不包含训练好的模型权重、训练数据集、回放或日志。

未来如果加入模型权重或数据集，必须单独记录：

```text
来源
许可证
是否允许再分发
是否允许商用
是否包含真人数据
是否包含游戏资源
```

不能默认把代码许可证套用到模型权重或训练数据上。

## 6. 贡献

除非贡献者明确书面说明，提交到本项目的代码贡献按 Apache-2.0 授权，文档贡献按 CC BY 4.0 授权。

提交贡献前，应确保贡献者有权提交相关代码、文本和数据，并且没有把第三方受限制内容混入仓库。

## 7. 免责声明

本说明用于整理项目许可证边界，不构成法律意见。公开发布、商业使用、分发修改版本或提供网络服务前，应根据实际情况核对所有许可证原文并咨询专业人士。
