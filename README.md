# 代码（code/）

本项目全部脚本。**这是本仓库唯一纳入 Git 的部分**；数据、模型、文档、论文都留在本地对应目录。

- 语言：Python 3.11（conda 环境 `station`）
- 依赖：[`requirements.txt`](requirements.txt)
- 规模：33 个脚本 + 2 个配置 JSON + 5 个 notebook

---

## 一、目录与职责

按「是不是每次都要用」分三层：**核心流程** → **论文一次性实验** → **归档**。

| 目录 | 数量 | 定位 | 常用度 |
|---|---:|---|---|
| [`pipeline/`](pipeline) | 6 | 多模态分析主流水线：评论归位 → LLM 图文理解 → 情感 → 报告 | ✅ 每次分析都要跑 |
| [`topic_modeling/`](topic_modeling) | 5 | 数据清洗、情感筛选、BERTopic 聚类 | ✅ 主流程 |
| [`experiments/`](experiments) | 9 | 论文实验：0905 重跑、K 值选择、消融、图 3/6/7 | ⚠ 一次性，但论文数字靠它回溯 |
| [`figures/`](figures) | 2 | 现用插图生成（框架图、热力图） | ✅ 改图时用 |
| [`tools/`](tools) | 2 | Qwen3-VL 图片描述支线 | 支线，未进论文主线 |
| [`daily_report/`](daily_report) | 2 | 舆情日报批量处理 | 另一条业务线 |
| [`paper_build/`](paper_build) | 1 | 论文 docx 排版生成 | ✅ 排版时用 |
| [`notebooks/`](notebooks) | 5 | 探索过程记录 | 历史存档 |
| [`archive/`](archive/README.md) | 6 | 一次性、已被取代、与论文无关的脚本 | ❌ 确认后可整目录删除 |

`config/` 放站点分析口径（10 类服务环节、影响力权重）与关键词过滤规则，被 `experiments/` 引用。

---

## 二、大模型（LLM）部分

这是整条流水线的核心：**用 Qwen3.6-27B 多模态模型逐帖读图+读文+读评论，产出结构化描述与分类**。

### 涉及的文件

| 文件 | 角色 |
|---|---|
| [`pipeline/analyze_notes.py`](pipeline/analyze_notes.py) | **主推理脚本**（264 行）：加载 Qwen3.6-27B，构造多模态输入，逐帖生成 `description` + `category`，每 5 条落盘、支持断点续跑 |
| [`pipeline/process_comments.py`](pipeline/process_comments.py) | 前置：把评论 xlsx 按 note_id 拆进各帖子目录 `comment.json` |
| [`pipeline/run_sentiment.py`](pipeline/run_sentiment.py) | 后置：BERT 情感模型打 Positive/Negative + 5 类业务口径 |
| [`tools/caption_xizhan_qwen3vl.py`](tools/caption_xizhan_qwen3vl.py) | 支线：用 **Qwen3-VL** 给配图生成中文描述（双卡、断点续跑） |
| [`tools/caption_xizhan_media.py`](tools/caption_xizhan_media.py) | 上者的封装：输出「图片相对路径 → 中文描述」JSON |

### 输入格式（每个帖子一个目录）

```
<station>_media/<note_id>/
├── info.json        # 帖子元数据：title / desc / liked_count / tags / upload_time
├── detail.txt       # 帖子详情快照
├── image_0.jpg …    # 配图，最多取 4 张
└── comment.json     # 评论，由 process_comments.py 生成
```

`analyze_notes.py` 的 `STATION_CONFIG` 为 10 个站点写死了 `data_dir` 与 `output`（旧 GPU 机路径 `/home/lu.junting/station/...`），换机器要改这一处。

### 模型与 Prompt

- 模型：`AutoProcessor` + `AutoModelForImageTextToText`，bf16，`device_map="auto"` 多卡自动切分，`enable_thinking=False` 关思考链，`do_sample=False` 保证确定性。
- 路径：`MODEL_PATH = "/home/lu.junting/station/qwen-3.6"`，**约 52 GB**。
- Prompt：System + User 两段式。System 定义 17 个分类（10 个站区服务环节 + 4 个小红书内容类 + 3 个无信息类）；User 拼「图片 + 标题 + 正文 + 互动数 + 标签 + 发布时间 + 点赞前 15 条评论」。完整文案见 [`../docs/guides/llmPart_README.md`](../docs/guides/llmPart_README.md)。
- 解析：三级正则容错，解析失败落 `parse_error` 并用 `xhs_other` 兜底。

### 输出

`<station>_analysis.json`，之后 `run_sentiment.py` 再补 `sentiment_label` / `sentiment_score` / `business_category_v2`。现有结果在 [`../data/llm_output/llmOutput/`](../data/llm_output/llmOutput/)：10 站、每站 4 个文件。

### ⚠ 重跑前必须自备

| 需要什么 | 说明 |
|---|---|
| **Qwen3.6-27B 权重（约 52 GB）** | 不在本机，原在 GPU 服务器 `/home/lu.junting/station/qwen-3.6`。`data/models/` 只有嵌入模型与 BERT 情感模型 |
| **GPU** | A100-80GB（52 GB 权重 bf16 + 4 张图的多模态输入） |
| **输入素材** | ✅ 已齐全：10 个站在 `data/raw/station_data/`（1208 帖 / 4084 图） |
| **爬虫脚本** | ❌ 不在仓库，只有爬虫产物；`info.json` + `image_*.jpg` 由外部工具生成 |

> 2026-08-29 的代码快照（`Desktop/station_代码_20260829/`）与输出快照（`Desktop/outputs/`）已逐字节比对，**31/31 代码、46/46 输出与本目录完全一致**，无独有内容，未重复收录。

---

## 三、脚本清单

### pipeline/ — 多模态分析流水线（核心）

| 脚本 | 作用 |
|---|---|
| `process_comments.py` | 评论 xlsx → 各帖子目录 `comment.json` |
| `analyze_notes.py` | Qwen3.6-27B 逐帖生成 `description` + `category`，可断点续跑 |
| `run_sentiment.py` | BERT 情感判定 + 业务口径映射 |
| `generate_report.py` | 按站点生成 Word 舆情分析报告 |
| `generate_statistics.py` | 各站点与全站统计 Markdown |
| `gen_md_airport.py` | 机场专项（大兴、首都）Markdown 报告 |

### topic_modeling/ — 清洗与聚类（核心）

| 脚本 | 作用 |
|---|---|
| `clean_station_data.py` | 小红书导出 Excel 清洗：去重、抽文本、过滤非站区内容 |
| `sentiment_filter.py` | 情感过滤（中性/负向），BERT 打分 |
| `build_xizhan_negative_texts.py` | 西站负向文本的帖子级聚合（依赖图片描述 JSON） |
| `run_bertopic.py` | BERTopic 主流程（可作库调用） |
| `run_bertopic_experiments.py` | 嵌入 / 降维 / 聚类组合的对比实验 |

### experiments/ — 论文实验（一次性，保留以便回溯）

| 脚本 | 作用 | 产出 |
|---|---|---|
| `rerun_0905.py` | **论文主实验**：PageRank 方案 A + 单变量消融 + 多种子 + 敏感性检验 | `data/results/output/0905/` |
| `rerun_0905_supplement.py` | 补充实验：填补 4.5、4.7 节空缺，绘 图 3/6/7 | 同上 |
| `rerun_0905_followup.py` | 一致性检验（节点级增量信息） | 同上 |
| `rerun_0905_node_sens.py` | 节点级参数敏感性 + 代表性帖子 | 同上 |
| `run_bertopic_293_default.py` | 293 条样本的标准中文 BERTopic 基线 | `output/1020/bertopic_293_default/` |
| `plot_paper_figures.py` | 论文图 3、图 6、图 7 | `output/0905/figures/` |
| `plot_bertopic_293_named.py` | 17 个主题命名后按五栏重绘 | `output/1020/…_more_topics/` |
| `replot_original_topics_5cols.py` | 用已保存的 Plotly 数值重排 10 主题图 | `output/1020/original_10_topics_5cols/` |
| `paper_common.py` | **公共组件**：数据载入、分词、NPMI、语义网络与 PageRank | — |

> `paper_common.py` 被上面 5 个脚本 import，所以放在同一目录；脚本靠自身所在目录找到它，不依赖外部 `sys.path`。

### 其余

| 目录 | 脚本 | 作用 |
|---|---|---|
| `figures/` | `generate_framework_v2_svg.py` | 图 1 框架图（可编辑 SVG） |
| | `create_hot_vector_svg.py` | 主题 × 站区热力图（Word 兼容矢量） |
| `tools/` | `caption_xizhan_qwen3vl.py` | Qwen3-VL 双卡图片描述 |
| | `caption_xizhan_media.py` | 上者的 JSON 封装 |
| `daily_report/` | `process_reports.py` | 日报原始表格批量处理 |
| | `analyze_daily_reports.py` | 日报统计与整改措施汇总 |
| `paper_build/` | `build_paper_docx.py` | 按期刊体例生成论文 docx |
| `notebooks/` | `a`、`bertop`、`bis`、`comp`、`exp` | 探索过程（主题数、K 值、对比实验草稿） |

已归档的 6 个脚本见 [`archive/README.md`](archive/README.md)。

---

## 四、运行顺序

```bash
conda activate station
cd ~/Documents/hhgh/station        # 在项目根执行

# ① 评论归位（每站一次）
python code/pipeline/process_comments.py --station xizhan

# ② 图文理解（GPU，可断点续跑）
python code/pipeline/analyze_notes.py --station xizhan

# ③ 情感判定（全站批量）
python code/pipeline/run_sentiment.py

# ④ 报告
python code/pipeline/generate_report.py --station xizhan
python code/pipeline/generate_statistics.py

# ⑤ 清洗 → 主题建模 → 论文实验
python code/topic_modeling/clean_station_data.py
python code/experiments/rerun_0905.py        # 输出到 data/results/output/0905
```

站点 key：`xizhan`、`nanzhan`、`beizhan`、`fengtai`、`chaoyang`、`qinghe`、`tongzhou`、`beijing`、`daxingjichang`、`shoudujichang`。

---

## 五、路径与依赖

### 已修好的部分

本次整理把 `experiments/`、`topic_modeling/` 里的路径常量全部改成**基于脚本自身位置**推导，不再依赖旧绝对路径：

| 常量 | 现在指向 |
|---|---|
| `paper_common.ROOT` | 项目根 `~/Documents/hhgh/station` |
| `paper_common.MODEL_PATH` | `data/models/embModel/Qwen3-Embedding-0.6B` |
| `paper_common.KEYWORD_FILTER_CONFIG_PATH` | `code/config/station_keyword_filter.json` |
| `paper_common.STOPWORD_PATHS` | `data/dictionaries/stopword.txt`、`stopinthis.txt` |
| `paper_common.LLM_OUTPUT_DIR` | `data/llm_output/llmOutput` |
| `rerun_0905*.py` 的 `OUT` | `data/results/output/0905` |
| `run_bertopic_293_default.py` 的 `OUT` / `SOURCE` | `data/results/output/1020/…`、`data/results/output/0905/…` |

全部 33 个脚本已通过 `py_compile` 语法检查。

### 仍需手工改的部分

`pipeline/` 6 个脚本里写死了 GPU 服务器的绝对路径（`/home/lu.junting/station/...`），集中在 `STATION_CONFIG` 的 `data_dir` / `output`。换机器时改这几处：

| 旧（GPU 机） | 新（本机） |
|---|---|
| `/home/lu.junting/station/data/<站>/<站>_media` | `data/raw/station_data/<站目录>/<media>` |
| `/home/lu.junting/station/outputs/<站>/<站>_analysis.json` | `data/llm_output/llmOutput/<站>/<站>_analysis.json` |
| `/home/lu.junting/station/qwen-3.6` | 自备的 Qwen3.6-27B 权重路径 |

`daily_report/` 两个脚本也各有几处旧路径，未改（那条业务线依赖另一套目录）。

### 其余注意

1. **`analyze_notes.py` 的三分类情感标签不在这里产出**：它只输出 `description` 与 `category`，Positive/Negative 由 `run_sentiment.py` 的 BERT 模型给出。
2. **配置口径**：`config/station_analysis_tasks.json` 定义 10 类服务环节、任务分组与影响力权重（点赞 1.0 / 收藏 1.5 / 评论 2.0 / 分享 3.0），只用于结果解释与占比统计，不参与 embedding 聚类。
3. **notebooks 需要手动设根目录**：里面的 `sys.path.append` 仍指向旧的 `scripts/`。

---

## 六、Git

```bash
git status       # 只跟踪 code/、根 README.md、.gitignore
```

`data/`、`docs/`、`papers/` 已由根目录 `.gitignore` 排除。`__pycache__/`、`*.pyc`、`.ipynb_checkpoints/` 也已忽略。
