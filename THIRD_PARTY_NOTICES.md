<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 第三方组件与参考项目

本文件记录项目当前使用、依赖或参考的第三方项目。项目自身许可证不覆盖这些内容。

## 1. 运行依赖

| 组件 | 用途 | 许可证 | 来源 |
|---|---|---|---|
| Pydantic | 协议模型和校验 | MIT | https://github.com/pydantic/pydantic |
| Python | 运行环境 | PSF License | https://www.python.org |
| Java / OpenJDK | RWPP 运行环境 | GPLv2 with Classpath Exception | https://openjdk.org |

Pydantic 通过 `pyproject.toml` 声明为依赖，当前仓库不包含其源代码副本。

## 2. 开发依赖

| 组件 | 用途 | 许可证 |
|---|---|---|
| jsonschema | JSON Schema 校验 | MIT |
| pytest | 可选测试运行器 | MIT |
| setuptools | 构建后端 | MIT |

开发依赖不随运行时分发。

## 3. 可选集成与参考项目

### RWPP

- 项目：https://github.com/RWPP-Team/RWPP
- 用途：未来可选的无头服务端和 API 集成
- 许可证：AGPL-3.0
- 当前状态：未复制代码，未随本项目分发

如果未来使用或修改 RWPP 代码，必须单独遵守 AGPL-3.0，并在相关目录保留许可证和来源说明。

### Rusted Warfare LLM Pilot

- 项目：https://github.com/Crystalhihihi/rusted-warfare-llm-pilot
- 用途：Java Agent 和状态读取思路参考
- 仓库 LICENSE 文件：Apache-2.0
- GitHub 自动识别状态：NOASSERTION
- 当前状态：仅作为参考资料，未复制其代码

复用其代码前必须重新检查具体文件、LICENSE 和 NOTICE。

## 4. 游戏与商店

- 《铁锈战争》由 Corroding Games 开发和发行。
- PC 版本通过 Steam 分发。
- 本仓库不包含游戏本体、assets、商店安装包或 Steam 客户端。
- “Rusted Warfare”名称和相关商标归其权利人所有。

## 5. 维护要求

新增依赖或复制第三方代码前，更新本文件并记录：

```text
名称
版本或 commit
来源 URL
许可证
使用的文件
是否修改
```

发布前应重新生成依赖清单并检查许可证兼容性。
