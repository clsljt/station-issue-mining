# 归档脚本（code/archive/）

这里是**一次性用完、已被取代、或与论文主线无关**的脚本。保留只为追溯，不影响当前流程，确认不需要后可以整目录删除。

| 文件 | 原位置 | 为什么归档 | 产出 |
|---|---|---|---|
| `figures_generate_chapter3_framework.py` | `figures/generate_chapter3_framework.py` | 旧版图 1（`chapter3_framework.svg/png`），已被 `figures/generate_framework_v2_svg.py` 及 `fig_all/framework_journal_five_v*.svg` 取代 | `papers/figures/Papers_figures/chapter3_framework.*` |
| `figures_convert_figures_to_eps.py` | `figures/convert_figures_to_eps.py` | 一次性格式转换（PDF/SVG/HTML → EPS/EMF），投稿时用过一轮 | `Papers_figures/eps/`、`emf/` |
| `tools_benchmark_qwen3vl_batch.py` | `tools/benchmark_qwen3vl_batch.py` | 纯性能基准，测 Qwen3-VL 的 batch 吞吐，不参与业务 | 无 |
| `topic_modeling_word_frequency.py` | `topic_modeling/word_frequency.py` | 早期词频统计，主题建模上线后不再使用 | `outputs/*/word_frequency_top200.csv` |
| `topic_modeling_clean_gongzhonghao.py` | `topic_modeling/clean_gongzhonghao.py` | 公众号文章清洗，与论文数据集无关 | `data/articles/gongzhonghao/` |
| `paper_build_revise_sections_31_32.py` | `paper_build/revise_sections_31_32.py` | 3.1–3.2 节的批量文本替换，改完即弃 | 无（直接改 docx） |

## 注意

- 文件名前缀（`figures_`、`topic_modeling_` 等）标明了它们原来的目录，避免同名冲突。
- `topic_modeling_word_frequency.py` 内部用 `from word_frequency import compute_word_frequency` 自引用（模块 + CLI 一体的写法），归档后这段引用不再有效，需要运行时改回原目录名。
- 这些脚本的文件路径都指向整理前的旧位置（`/Users/clsljt/Documents/hhgh/station/...`），直接运行前需要先改路径。
