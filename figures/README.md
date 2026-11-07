# 出图脚本（code/figures/）

论文插图的生成脚本。原先是 `Papers/figures/` 里与图片混放的，整理时抽到代码区，图片资产仍在 [`../../papers/figures/`](../../papers/figures/README.md)。

| 脚本 | 作用 | 输出 |
|---|---|---|
| `generate_framework_v2_svg.py` | 图 1：加宽版、可编辑 SVG 框架图（含问题识别环节） | `framework_v2.svg` |
| `create_hot_vector_svg.py` | 主题 × 站区热力图矢量图（Word 兼容） | `hot.svg` |

已归档的两个（旧版图 1 生成、格式转换工具）见 [`../archive/README.md`](../archive/README.md)。

## 运行

```bash
cd code/figures
python generate_framework_v2_svg.py
python create_hot_vector_svg.py
```

## 注意

- 这两个脚本原本与图片放在同一目录，靠相对路径读取 `Papers/figures/` 下的素材。目录拆分后，**运行时需要把工作目录切到 `papers/figures/Papers_figures/`**，或先改脚本里的相对路径。
- 图 3、图 6、图 7 的数据驱动出图脚本不在 `figures/`，在 [`../experiments/plot_paper_figures.py`](../experiments/plot_paper_figures.py)；主题命名的五栏重绘在 [`../experiments/plot_bertopic_293_named.py`](../experiments/plot_bertopic_293_named.py)。
- 图 1 的文字标签已过时（「主题级影响力建模」「影响分数」「基于 BERT 的负向文本识别」），SVG 里是活字，改字即可。
