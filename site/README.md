<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# IronZero Lab 展示页

这是一个无外部依赖的静态单页站点，入口为 `index.html`。

本地预览：

```powershell
python -m http.server 8000 --directory site
```

然后访问 `http://127.0.0.1:8000/`。

页面通过 Steam 官方 CDN 远程引用《铁锈战争》宣传截图，用于非官方项目的游戏识别与介绍。图片不存入仓库，也不属于项目开源许可证范围。HUD、排版和交互层由本项目原创实现。
