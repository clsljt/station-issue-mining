# 代码（code/）

本项目全部脚本，按流程阶段分目录。**这是本仓库唯一纳入 Git 的部分**；数据、模型、文档、论文都留在本地对应目录。

- 语言：Python 3.11
- 环境：conda `station`
- 依赖：[`requirements.txt`](requirements.txt)
- 代码量：43 个文件（38 个脚本 + 2 个配置 JSON + 5 个 notebook + 3 个 README/依赖清单）

---

## 一、目录与职责

| 目录 | 职责 | 关键脚本 |
|---|---|---|
| [`pipeline/`](pipeline) | 多模态大模型分析流水线：评论归位 → 图文理解 → 情感判定 → 报告生成 | `analyze_notes.py`、`run_sentiment.py` |
| [`topic_modeling/`](topic_modeling) | 数据清洗、主题建模、指标计算、论文实验重跑 | `rerun_0905.py`、`paper_common.py` |
| [`figures/`](figures) | 论文插图生成（框架图、热力图、格式转换） | `generate_framework_v2_svg.py` |
| [`tools/`](tools) | 图片描述（image captioning）与吞吐基准 | `caption_xizhan_qwen3vl.py` |
| [`daily_report/`](daily_report) | 舆情日报的批量处理与统计分析 | `process_reports.py` |
| [`paper_build/`](paper_build) | 论文 docx 排版生成、章节改写脚本 | `build_paper_docx.py` |
| [`notebooks/`](notebooks) | 探索过程 notebook（保留原始实验痕迹） | `exp.ipynb`、`bis.ipynb` |
| [`config/`](config) | 站点分析任务口径与关键词过滤规则 | `station_analysis_tasks.json` |

---

## 二、脚本清单

### pipeline/ — 多模态分析流水线

| 脚本 | 作用 |
|---|---|
| `process_comments.py` | 把评论 xlsx 按 note_id 拆进各帖子文件夹（`comment.json`） |
| `analyze_notes.py` | 调 Qwen3.6-27B 读「标题+正文+配图+高赞评论」，产出 `description` 与 `category`；支持断点续跑 |
| `run_sentiment.py` | 用 `senModel` 的 BERT 给每条记录打 `sentiment_label` / `sentiment_score`，并映射 5 类业务口径 |
| `generate_report.py` | 按站点生成 Word 舆情分析报告 |
| `generate_statistics.py` | 生成各站点与全站的统计 Markdown |
| `gen_md_airport.py` | 机场专项（大兴、首都）Markdown 报告 |

### topic_modeling/ — 清洗、建模与实验

| 脚本 | 作用 |
|---|---|
| `clean_station_data.py` | 小红书导出 Excel 清洗：去重、抽文本、过滤非站区内容 |
| `sentiment_filter.py` | 按情感筛出负向评论 |
| `word_frequency.py` | 词频分析（清洗后 / 情感筛选后） |
| `build_xizhan_negative_texts.py` | 北京西站负向文本的帖子级聚合 |
| `clean_gongzhonghao.py` | 公众号文章正文清洗 |
| `run_bertopic.py` | BERTopic 主流程（可作库调用） |
| `run_bertopic_experiments.py` | 多组嵌入/降维/聚类组合的对比实验 |
| `run_bertopic_293_default.py` | 293 条论文样本上的标准中文 BERTopic 基线 |
| `paper_common.py` | **公共组件**：数据载入、分词、NPMI 等主题指标、语义网络与 PageRank |
| `rerun_0905.py` | 0905 重跑：PageRank 方案 A + 单变量消融 + 多种子 + 敏感性检验 |
| `rerun_0905_supplement.py` | 补充实验：填补 4.5 / 4.7 节空缺，绘制图 3、6、7 |
| `rerun_0905_followup.py` | 后续一致性检验 |
| `rerun_0905_node_sens.py` | 节点级参数敏感性 + 各主题代表性帖子 |
| `plot_paper_figures.py` | 论文图 3、图 6、图 7 |
| `plot_bertopic_293_named.py` | 17 个主题命名后按五栏重绘 |
| `replot_original_topics_5cols.py` | 用已保存的 Plotly 数值重排 10 主题图 |

### figures/ — 插图

| 脚本 | 作用 |
|---|---|
| `generate_chapter3_framework.py` | 图 1：多模态舆情分析框架 |
| `generate_framework_v2_svg.py` | 加宽版、可编辑 SVG 框架图（含问题识别环节） |
| `create_hot_vector_svg.py` | 主题 × 站区热力图矢量图 |
| `convert_figures_to_eps.py` | PDF / SVG / Plotly HTML → EPS / EMF / SVG 批量转换 |

### tools/ — 图片描述

| 脚本 | 作用 |
|---|---|
| `caption_xizhan_qwen3vl.py` | 双卡、可断点续跑的 Qwen3-VL 图片描述生成 |
| `caption_xizhan_media.py` | 输出「图片相对路径 → 中文描述」JSON |
| `benchmark_qwen3vl_batch.py` | 单卡不同 batch size 的吞吐基准 |

### daily_report/ — 舆情日报

| 脚本 | 作用 |
|---|---|
| `process_reports.py` | 日报原始表格的批量处理 |
| `analyze_daily_reports.py` | 日报数据统计与整改措施汇总 |

### paper_build/ — 论文排版

| 脚本 | 作用 |
|---|---|
| `build_paper_docx.py` | 按期刊体例生成论文 docx（样式、表注、公式编号） |
| `revise_sections_31_32.py` | 3.1–3.2 节批量改写与替换 |

---

## 三、运行顺序

```bash
conda activate station
cd code

# ① 评论归位（每站一次）
python pipeline/process_comments.py --station xizhan

# ② 图文理解（GPU，可断点续跑）
python pipeline/analyze_notes.py --station xizhan

# ③ 情感判定（全站批量）
python pipeline/run_sentiment.py

# ④ 报告
python pipeline/generate_report.py --station xizhan
python pipeline/generate_statistics.py

# ⑤ 清洗 → 主题建模 → 指标
python topic_modeling/clean_station_data.py
python topic_modeling/rerun_0905.py          # 论文主结果，输出到 data/results/output/0905
```

`analyze_notes.py --station` 支持站点：`xizhan`、`nanzhan`、`beizhan`、`fengtai`、`chaoyang`、`qinghe`、`tongzhou`、`beijing`、`daxingjichang`、`shoudujichang`。

---

## 四、⚠ 路径与依赖注意

1. **脚本里写死了旧绝对路径**。整理前工作区根是 `~/Documents/hhgh/station/`，数据已迁到 `data/`、`docs/`、`papers/`，共有 12 个脚本、61 处 `/Users/clsljt/...` 或 `/home/lu.junting/...` 需要按新位置改：

   ```
   daily_report/analyze_daily_reports.py   pipeline/analyze_notes.py        topic_modeling/clean_station_data.py
   daily_report/process_reports.py         pipeline/gen_md_airport.py       topic_modeling/run_bertopic.py
   pipeline/generate_report.py             pipeline/process_comments.py     topic_modeling/sentiment_filter.py
   pipeline/run_sentiment.py               topic_modeling/clean_gongzhonghao.py   topic_modeling/word_frequency.py
   ```

   典型映射（旧 → 新）：

   | 旧路径 | 新路径 |
   |---|---|
   | `station/data/` | `station/data/raw/station_data/` |
   | `station/xlsx/` | `station/data/raw/station_xlsx/` |
   | `station/llmOutput/` | `station/data/llm_output/llmOutput/` |
   | `station/llmPart/outputs/` | `station/data/llm_output/llmPart_outputs/` |
   | `station/output/` | `station/data/results/output/` |
   | `station/scripts/` | `station/code/topic_modeling/` |
   | `station/senModel/`（远程机 `/home/lu.junting/station/senModel`） | `station/data/models/senModel/` |

2. **`paper_common.py` 是共享模块**，`rerun_0905*.py` 都依赖它。原来 `scripts/` 下的脚本用 `sys.path.append(".../scripts")` 引它，目录改名后需改成 `code/topic_modeling`（或从该目录直接运行）。

3. **notebooks 需要手动设根目录**：`a.ipynb`、`comp.ipynb`、`exp.ipynb`、`bis.ipynb`、`bertop.ipynb` 里的 `sys.path.append` 同样指向旧的 `scripts/`。

4. **`analyze_notes.py` 的三分类情感标签不在这里产出**，它只输出 `description` 与 `category`；Positive/Negative 由 `run_sentiment.py` 的 BERT 模型给出。

5. **配置口径**：`config/station_analysis_tasks.json` 定义 10 类服务环节、任务分组与影响力权重（点赞 1.0 / 收藏 1.5 / 评论 2.0 / 分享 3.0），只用于结果解释与占比统计，不参与 embedding 聚类。

---

## 五、Git

```bash
git status       # 只跟踪 code/、根 README.md、.gitignore
```

`data/`、`docs/`、`papers/` 已由根目录 `.gitignore` 排除。`__pycache__/`、`*.pyc`、`.ipynb_checkpoints/` 也已忽略。
