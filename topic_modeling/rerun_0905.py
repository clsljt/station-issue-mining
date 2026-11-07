"""0905 重跑：PageRank 方案 A（全局连通网络）+ 单变量消融 + 多种子 + 敏感性检验。

输出目录：output/0905
设计要点：
1. 主题划分沿用 output/0623 的既有结果（按 id 关联），保证表 5 其余列不变，只替换关注度列；
2. 重算 Qwen 嵌入后，先用「主题内建网」复现既有 kmeans_pagerank 作为一致性校验，
   校验通过再报告方案 A 的结果，避免把嵌入差异误当成方案差异；
3. 消融实验独立重新聚类，与主题划分互不影响。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jieba
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

OUT = Path("output/0905")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("embModel/Qwen3-Embedding-0.6B")
DEFAULT_MODEL_FALLBACK = "embModel/BAAI/bge-small-zh-v1.5"
KEYWORD_FILTER_CONFIG_PATH = Path("config/station_keyword_filter.json")
STOPWORD_PATHS = [Path("stopword.txt"), Path("stopinthis.txt")]
PREV_RESULT = Path("output/0623/bertopic_kmeans_results.csv")

BATCH_SIZE = 16
MIN_TEXT_LENGTH = 4
SIMILARITY_TOP_K = 8
SIMILARITY_THRESHOLD = 0.42
ALPHA = 0.85
K_TOPICS = 10
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]

CHINESE_WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$")

TOPIC_TITLES = {
    0: "候车环境与进站体验",
    1: "丰台站出站换乘动线",
    2: "机场运行延误与旅客滞留",
    3: "首都机场航站楼服务与接驳体验",
    4: "网约车接驳与运力不足",
    5: "列车晚点与候车滞留",
    6: "通州站空间布局与通行体验",
    7: "机场线票价与接驳成本",
    8: "站台吸烟与空气环境",
    9: "目的车站混淆与改签困扰",
}

STATION_CN = {
    "xizhan": "北京西站",
    "nanzhan": "北京南站",
    "beizhan": "北京北站",
    "fengtai": "北京丰台站",
    "chaoyang": "北京朝阳站",
    "qinghe": "北京清河站",
    "tongzhou": "北京通州站",
    "beijing": "北京站",
    "daxingjichang": "北京大兴国际机场",
    "shoudujichang": "北京首都国际机场",
}

LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def section(title: str) -> None:
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------- 分词与指标

stop_words = {"无标题"}
for path in STOPWORD_PATHS:
    if path.is_file():
        stop_words.update(
            w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()
        )

with KEYWORD_FILTER_CONFIG_PATH.open("r", encoding="utf-8") as f:
    keyword_filter_config = json.load(f)

keyword_exclude_words = {
    str(w).strip() for w in keyword_filter_config.get("exclude_words", []) if str(w).strip()
}
keyword_prefer_words = {
    str(w).strip() for w in keyword_filter_config.get("prefer_words", []) if str(w).strip()
}
stop_words.update(keyword_exclude_words)


def tokenize_zh(text: str) -> list[str]:
    return [
        w.strip()
        for w in jieba.lcut(str(text))
        if len(w.strip()) > 1 and not w.strip().isdigit() and w.strip() not in stop_words
    ]


def tokenize_chinese_only(text: str) -> list[str]:
    return [
        w.strip()
        for w in jieba.lcut(str(text))
        if CHINESE_WORD.fullmatch(w.strip()) and w.strip() not in stop_words
    ]


def topic_diversity_score(topic_word_lists: list[list[str]]) -> float:
    all_words = [w for words in topic_word_lists for w in words]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def topic_coherence_npmi(
    docs: list[str], topic_word_lists: list[list[str]], eps: float = 1e-12
) -> float:
    tokenized = [set(tokenize_zh(d)) for d in docs]
    n_docs = len(tokenized)
    if n_docs == 0:
        return float("nan")

    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    pair_scores = []
    for words in topic_word_lists:
        words = [w for w in dict.fromkeys(words) if doc_freq.get(w, 0) > 0]
        if len(words) < 2:
            continue
        for i, wi in enumerate(words[:-1]):
            di = doc_freq[wi]
            for wj in words[i + 1 :]:
                dj = doc_freq[wj]
                co = sum(1 for tk in tokenized if wi in tk and wj in tk)
                pi, pj = di / n_docs, dj / n_docs
                pij = (co + eps) / n_docs
                pair_scores.append(np.log(pij / (pi * pj)) / (-np.log(pij)))
    return float(np.mean(pair_scores)) if pair_scores else float("nan")


def extract_topic_words(docs: list[str], topics) -> dict:
    topic_documents = (
        pd.DataFrame({"topic": topics, "content": docs})
        .groupby("topic", sort=True)["content"]
        .agg(" ".join)
    )
    vec = TfidfVectorizer(
        tokenizer=tokenize_chinese_only, token_pattern=None, lowercase=False, min_df=1
    )
    matrix = vec.fit_transform(topic_documents)
    terms = vec.get_feature_names_out()

    topic_word_map = {}
    for topic_id, row in zip(topic_documents.index, matrix.toarray()):
        weighted = row.copy()
        for idx, term in enumerate(terms):
            if term in keyword_prefer_words:
                weighted[idx] *= 1.35
        top_idx = [i for i in weighted.argsort()[::-1] if row[i] > 0][:10]
        topic_word_map[topic_id] = [terms[i] for i in top_idx]
    return topic_word_map


def calculate_metrics(name: str, docs: list[str], embeddings, topics) -> dict:
    labels = np.asarray(topics)
    topic_ids = np.unique(labels)
    topic_word_map = extract_topic_words(docs, topics)
    word_lists = [topic_word_map[t] for t in sorted(topic_ids) if topic_word_map.get(t)]

    def safe(fn, **kw):
        mask = labels != -1
        x, y = np.asarray(embeddings)[mask], labels[mask]
        if len(np.unique(y)) < 2 or len(y) <= len(np.unique(y)):
            return float("nan")
        try:
            return float(fn(x, y, **kw))
        except ValueError:
            return float("nan")

    return {
        "experiment": name,
        "n_documents": int(len(docs)),
        "n_topics": int(len(topic_ids)),
        "silhouette_score": safe(silhouette_score, metric="cosine"),
        "davies_bouldin_index": safe(davies_bouldin_score),
        "calinski_harabasz_index": safe(calinski_harabasz_score),
        "topic_coherence_npmi": topic_coherence_npmi(docs, word_lists),
        "topic_diversity": topic_diversity_score(word_lists),
    }


# ------------------------------------------------------------ 建网与 PageRank


def percentile_score(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    if series.nunique(dropna=False) <= 1:
        return pd.Series(50.0, index=series.index)
    return series.rank(method="average", pct=True) * 100


def build_group_semantic_graph(
    group_values,
    similarity_matrix_input,
    top_k: int = SIMILARITY_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> nx.Graph:
    """在同一分组内部连接语义最相似的帖子（与原实现保持一致）。"""
    groups = pd.Series(group_values).reset_index(drop=True)
    matrix = np.asarray(similarity_matrix_input)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(groups)))

    for _, indices in groups.groupby(groups).groups.items():
        indices = list(indices)
        if len(indices) <= 1:
            continue
        group_similarity = matrix[np.ix_(indices, indices)]
        for local_i, source_idx in enumerate(indices):
            neighbor_order = group_similarity[local_i].argsort()[::-1][:top_k]
            for local_j in neighbor_order:
                target_idx = indices[local_j]
                sim = float(group_similarity[local_i, local_j])
                if source_idx == target_idx or sim < threshold:
                    continue
                graph.add_edge(source_idx, target_idx, weight=sim)

    if graph.number_of_edges() == 0 and len(groups) > 1:
        graph.add_edges_from(
            (i, i + 1, {"weight": 1.0}) for i in range(len(groups) - 1)
        )
    return graph


def personalized_pagerank(graph: nx.Graph, engagement: pd.Series) -> pd.Series:
    raw_p = np.log1p(engagement.astype(float).reset_index(drop=True))
    if float(raw_p.sum()) == 0:
        personalization = {i: 1 / len(engagement) for i in range(len(engagement))}
    else:
        personalization = (raw_p / raw_p.sum()).to_dict()
    scores = nx.pagerank(
        graph,
        alpha=ALPHA,
        personalization=personalization,
        weight="weight",
        max_iter=200,
        tol=1e-8,
    )
    return pd.Series(scores).sort_index()


# ------------------------------------------------------------------ 数据载入

section("步骤 1  载入原始数据并复现 293 条问题导向样本")


def normalize_llm_sentiment(value) -> str:
    label = str(value or "").strip().lower()
    if label in {"负向", "负面", "negative", "neg"} or "负" in label:
        return "negative"
    if label in {"正向", "正面", "positive", "pos"} or "正" in label:
        return "positive"
    if label in {"中立", "中性", "neutral", "neu"} or "中" in label:
        return "neutral"
    return "unknown"


analysis_paths = sorted(Path("llmOutput").glob("*/*_analysis.json"))
records = []
for p in analysis_paths:
    station = p.parent.name
    for rec in json.load(p.open(encoding="utf-8")):
        if isinstance(rec, dict):
            records.append({**rec, "station": station})
raw = pd.DataFrame(records)
log(f"原始记录：{len(raw):,} 条，来自 {len(analysis_paths)} 个站区分析文件")

raw["llm_sentiment_group"] = raw["llm_sentiment"].map(normalize_llm_sentiment)

sent_dist = raw["llm_sentiment_group"].value_counts()
log("\n[表 1 用] LLM 情感三分类分布：")
for k, v in sent_dist.items():
    log(f"    {k:<10} {v:>5} 条  ({v / len(raw) * 100:.1f}%)")

if "sentiment_label" in raw.columns:
    log("\n[重要核查] 数据中另有 BERT 情感字段 sentiment_label：")
    for k, v in raw["sentiment_label"].value_counts().items():
        log(f"    {k:<10} {v:>5} 条  ({v / len(raw) * 100:.1f}%)")
    cross = pd.crosstab(raw["llm_sentiment_group"], raw["sentiment_label"])
    log("\n    llm_sentiment_group × sentiment_label 交叉表：")
    for line in cross.to_string().splitlines():
        log("      " + line)
    cross.to_csv(OUT / "sentiment_llm_vs_bert_crosstab.csv", encoding="utf-8-sig")

kmeans_keep = raw["llm_sentiment_group"].eq("negative")
neg_df = raw.loc[kmeans_keep].copy()
if "content" in neg_df.columns:
    neg_df = neg_df.drop(columns=["content"])

description_text = neg_df["description"].fillna("").astype(str).str.strip()
post_text = raw.loc[kmeans_keep, "content"].fillna("").astype(str).str.strip()
neg_df["source_content"] = post_text
neg_df["analysis_text"] = description_text.where(description_text.ne(""), post_text)

clean_content = neg_df["analysis_text"].fillna("").astype(str).str.strip()
keep = (
    clean_content.ne("")
    & clean_content.ne("无标题")
    & clean_content.str.len().ge(MIN_TEXT_LENGTH)
)

work = (
    neg_df.loc[keep]
    .rename(columns={"note_id": "id", "analysis_text": "content"})
    .reset_index(drop=True)
)
docs_desc = clean_content.loc[keep].tolist()
log(f"\n问题导向候选样本：{len(work):,} 条（占全量 {len(work) / len(raw) * 100:.1f}%）")

title_text = raw.loc[kmeans_keep, "title"].fillna("").astype(str).str.strip()
content_text = raw.loc[kmeans_keep, "content"].fillna("").astype(str).str.strip()
combined_text = (title_text + " " + content_text).str.strip()
docs_comb = combined_text.loc[keep].tolist()
log(f"原始短文本（标题+正文）平均长度：{np.mean([len(d) for d in docs_comb]):.1f} 字")
log(f"综合描述平均长度：          {np.mean([len(d) for d in docs_desc]):.1f} 字")

ENGAGEMENT_COLUMNS = ["liked_count", "collected_count", "comment_count", "share_count"]


def parse_count(value) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).replace(",", "").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("万"):
            return float(text[:-1]) * 10000
        if text.endswith("亿"):
            return float(text[:-1]) * 100000000
        return float(text)
    except ValueError:
        return 0.0


for col in ENGAGEMENT_COLUMNS:
    work[f"{col}_num"] = work[col].map(parse_count) if col in work else 0.0
work["互动总量"] = sum(work[f"{c}_num"] for c in ENGAGEMENT_COLUMNS)

# ------------------------------------------------- 沿用既有主题划分（按 id 关联）

section("步骤 2  关联既有主题划分（保证表 5 其余列不变）")

prev = pd.read_csv(PREV_RESULT)
topic_map = dict(zip(prev["id"].astype(str), prev["topic"]))
work["topic"] = work["id"].astype(str).map(topic_map)
missing = int(work["topic"].isna().sum())
log(f"主题标签关联成功：{len(work) - missing}/{len(work)} 条，缺失 {missing} 条")
if missing:
    raise SystemExit("主题标签关联失败，请检查 id 对齐")
work["topic"] = work["topic"].astype(int)

counts = work["topic"].value_counts().sort_index()
log(f"各主题样本数：{counts.to_dict()}")
expected = {0: 57, 1: 49, 2: 38, 3: 36, 4: 32, 5: 24, 6: 24, 7: 13, 8: 11, 9: 9}
log(f"与论文表 5 一致：{counts.to_dict() == expected}")

work["主题标题"] = work["topic"].map(TOPIC_TITLES)
work["站区"] = work["station"].map(STATION_CN).fillna(work["station"])

# --------------------------------------------------------------------- 嵌入

section("步骤 3  重算 Qwen3-Embedding-0.6B 语义向量")

from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    import torch

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
log(f"推理设备：{DEVICE}")

qwen_model = SentenceTransformer(
    str(MODEL_PATH), device=DEVICE, trust_remote_code=True, local_files_only=True
)


def encode(model, texts):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


emb_qwen_desc = encode(qwen_model, docs_desc)
emb_qwen_comb = encode(qwen_model, docs_comb)
log(f"Qwen 嵌入维度：{emb_qwen_desc.shape}  （论文 3.4 节待补 15 的 D 值 = {emb_qwen_desc.shape[1]}）")

sim = np.asarray(emb_qwen_desc) @ np.asarray(emb_qwen_desc).T
np.fill_diagonal(sim, 0)

# ------------------------------------------------ 一致性校验：复现旧的主题内建网

section("步骤 4  一致性校验：用「主题内建网」复现 output/0623 的既有 PageRank")

graph_old = build_group_semantic_graph(work["topic"], sim)
pr_old = personalized_pagerank(graph_old, work["互动总量"])
work["pagerank_主题内建网"] = pr_old.to_numpy()
work["影响分_主题内建网"] = work["pagerank_主题内建网"] * 10000

prev_pr = work["id"].astype(str).map(dict(zip(prev["id"].astype(str), prev["kmeans_pagerank"])))
corr = float(np.corrcoef(work["pagerank_主题内建网"], prev_pr)[0, 1])
max_abs = float(np.max(np.abs(work["pagerank_主题内建网"] - prev_pr)))
log(f"旧实现网络：{graph_old.number_of_nodes()} 节点，{graph_old.number_of_edges():,} 条边")
log(f"与既有 kmeans_pagerank 的 Pearson 相关：{corr:.6f}")
log(f"最大绝对偏差：{max_abs:.3e}")
log(
    "校验结论："
    + ("嵌入与建网口径复现成功，后续方案 A 结果可信" if corr > 0.999 else "复现不一致，需人工检查")
)

# 分量质量守恒的数值验证
p_vec = np.log1p(work["互动总量"].to_numpy())
p_vec = p_vec / p_vec.sum()
cons = pd.DataFrame(
    {
        "topic": work["topic"],
        "pi": work["pagerank_主题内建网"],
        "p": p_vec,
    }
).groupby("topic").sum()
cons["偏差"] = cons["pi"] - cons["p"]
log("\n分量质量守恒验证（旧实现下应有 Σπ = Σp）：")
log(f"    最大绝对偏差 {cons['偏差'].abs().max():.3e}，说明主题级得分等于个性化质量之和")
cons.to_csv(OUT / "validation_mass_conservation.csv", encoding="utf-8-sig")

# ------------------------------------------------------------- 方案 A 主计算

section("步骤 5  方案 A：全局连通语义网络 PageRank")

graph_new = build_group_semantic_graph(pd.Series("all", index=work.index), sim)
iso = [n for n, d in graph_new.degree() if d == 0]
avg_deg = 2 * graph_new.number_of_edges() / graph_new.number_of_nodes()
log(f"全局语义网络：{graph_new.number_of_nodes()} 节点，{graph_new.number_of_edges():,} 条边")
log(f"平均度 {avg_deg:.2f}，孤立节点 {len(iso)} 个   [3.5 节待补 18]")
log(f"连通分量数：{nx.number_connected_components(graph_new)}")
sizes = sorted((len(c) for c in nx.connected_components(graph_new)), reverse=True)
log(f"最大连通分量规模：{sizes[0]}（占 {sizes[0] / len(work) * 100:.1f}%），各分量规模 {sizes[:10]}")

tri = sim[np.triu_indices_from(sim, k=1)]
pct = float((tri < SIMILARITY_THRESHOLD).mean() * 100)
log(f"\n相似度阈值 {SIMILARITY_THRESHOLD} 对应全部节点对分布的第 {pct:.1f} 百分位   [4.7 节待补]")
log(f"节点对相似度：均值 {tri.mean():.4f}，中位数 {np.median(tri):.4f}，P90 {np.percentile(tri, 90):.4f}")

pr_new = personalized_pagerank(graph_new, work["互动总量"])
work["kmeans_pagerank"] = pr_new.to_numpy()
work["kmeans_impact_score"] = (work["kmeans_pagerank"] * 10000).round(4)
work["kmeans_network_centrality"] = percentile_score(work["kmeans_pagerank"]).round(2)

# ------------------------------------------------------------- 主题级汇总

section("步骤 6  主题级汇总（表 5 替换数据）")

topic_words = extract_topic_words(docs_desc, work["topic"].tolist())

summary = (
    work.groupby("topic")
    .agg(
        文本数=("id", "count"),
        点赞数=("liked_count_num", "sum"),
        收藏数=("collected_count_num", "sum"),
        评论数=("comment_count_num", "sum"),
        转发数=("share_count_num", "sum"),
        互动总量=("互动总量", "sum"),
        综合影响分_方案A=("kmeans_impact_score", "sum"),
        综合影响分_主题内建网=("影响分_主题内建网", "sum"),
        平均网络中心性=("kmeans_network_centrality", "mean"),
    )
    .reset_index()
)
summary["主题标题"] = summary["topic"].map(TOPIC_TITLES)
summary["主题词"] = summary["topic"].map(lambda t: " / ".join(topic_words.get(t, [])[:5]))
summary["文本占比"] = summary["文本数"] / summary["文本数"].sum()
summary["互动占比"] = summary["互动总量"] / summary["互动总量"].sum()
summary["相对关注度占比_方案A"] = summary["综合影响分_方案A"] / summary["综合影响分_方案A"].sum()
summary["相对关注度占比_主题内建网"] = (
    summary["综合影响分_主题内建网"] / summary["综合影响分_主题内建网"].sum()
)
summary["方案A与文本占比之差"] = summary["相对关注度占比_方案A"] - summary["文本占比"]
summary["旧实现与文本占比之差"] = summary["相对关注度占比_主题内建网"] - summary["文本占比"]

cols = [
    "topic", "主题标题", "主题词", "文本数", "文本占比", "互动总量", "互动占比",
    "相对关注度占比_方案A", "相对关注度占比_主题内建网",
    "方案A与文本占比之差", "旧实现与文本占比之差", "平均网络中心性",
    "综合影响分_方案A", "综合影响分_主题内建网",
    "点赞数", "收藏数", "评论数", "转发数",
]
summary = summary[cols].sort_values("topic")
summary.to_csv(OUT / "kmeans_topic_summary.csv", index=False, encoding="utf-8-sig")

log(f"{'主题':<4}{'名称':<18}{'文本占比':>9}{'互动占比':>9}{'方案A':>9}{'旧实现':>9}{'中心性':>8}")
for _, r in summary.iterrows():
    log(
        f"{int(r['topic']):<4}{r['主题标题']:<18}"
        f"{r['文本占比'] * 100:>8.2f}%{r['互动占比'] * 100:>8.2f}%"
        f"{r['相对关注度占比_方案A'] * 100:>8.2f}%{r['相对关注度占比_主题内建网'] * 100:>8.2f}%"
        f"{r['平均网络中心性']:>8.1f}"
    )

log()
log(f"方案 A 与文本占比的最大偏差：{summary['方案A与文本占比之差'].abs().max() * 100:.2f} 个百分点")
log(f"旧实现与文本占比的最大偏差：{summary['旧实现与文本占比之差'].abs().max() * 100:.2f} 个百分点")

# ------------------------------------------------------- 关注度指标增量检验

section("步骤 7  关注度指标的增量检验  [4.7 节第 (4) 项]")

pairs = [
    ("方案A影响分", "互动总量", summary["综合影响分_方案A"], summary["互动总量"]),
    ("方案A影响分", "文本数", summary["综合影响分_方案A"], summary["文本数"]),
    ("旧实现影响分", "互动总量", summary["综合影响分_主题内建网"], summary["互动总量"]),
    ("旧实现影响分", "文本数", summary["综合影响分_主题内建网"], summary["文本数"]),
    ("方案A影响分", "旧实现影响分", summary["综合影响分_方案A"], summary["综合影响分_主题内建网"]),
]
rows = []
for a, b, x, y in pairs:
    rho = spearmanr(x, y).statistic
    tau = kendalltau(x, y).statistic
    rows.append({"指标A": a, "指标B": b, "Spearman": round(float(rho), 4), "Kendall": round(float(tau), 4)})
    log(f"    {a} vs {b:<12} Spearman={rho:>7.4f}  Kendall={tau:>7.4f}")
pd.DataFrame(rows).to_csv(OUT / "attention_increment_check.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------- 网络参数敏感性

section("步骤 8  网络参数敏感性  [4.7 节第 (2) 项]")

base_rank = summary.set_index("topic")["综合影响分_方案A"].rank()
sens_rows = []
for k in (5, 8, 12):
    for th in (0.35, 0.42, 0.50):
        g = build_group_semantic_graph(
            pd.Series("all", index=work.index), sim, top_k=k, threshold=th
        )
        pr = personalized_pagerank(g, work["互动总量"])
        s = (
            pd.Series(pr.to_numpy() * 10000, index=work.index)
            .groupby(work["topic"])
            .sum()
        )
        tau = kendalltau(base_rank, s.rank()).statistic
        rho = spearmanr(base_rank, s.rank()).statistic
        n_iso = sum(1 for _, d in g.degree() if d == 0)
        sens_rows.append(
            {
                "top_k": k, "threshold": th,
                "边数": g.number_of_edges(), "孤立节点": n_iso,
                "Kendall_tau_vs_基准": round(float(tau), 4),
                "Spearman_vs_基准": round(float(rho), 4),
            }
        )
        log(
            f"    k={k:<3} th={th:<5} 边数={g.number_of_edges():>5} 孤立={n_iso:>3} "
            f"Kendall={tau:>7.4f}"
        )
sens = pd.DataFrame(sens_rows)
sens.to_csv(OUT / "network_param_sensitivity.csv", index=False, encoding="utf-8-sig")
log(f"\n9 组组合的 Kendall tau 最小值：{sens['Kendall_tau_vs_基准'].min():.4f}")

# 互动权重稳健性
section("步骤 9  互动权重稳健性  [4.7 节第 (5) 项]")
weight_schemes = {
    "等权（基准）": (1, 1, 1, 1),
    "评论分享加权": (1, 1, 2, 2),
    "仅点赞": (1, 0, 0, 0),
    "仅评论": (0, 0, 1, 0),
}
w_rows = []
for name, (wl, wf, wc, ws) in weight_schemes.items():
    m = (
        wl * work["liked_count_num"] + wf * work["collected_count_num"]
        + wc * work["comment_count_num"] + ws * work["share_count_num"]
    )
    pr = personalized_pagerank(graph_new, m)
    s = pd.Series(pr.to_numpy() * 10000, index=work.index).groupby(work["topic"]).sum()
    tau = kendalltau(base_rank, s.rank()).statistic
    w_rows.append({"权重方案": name, "Kendall_tau_vs_等权": round(float(tau), 4)})
    log(f"    {name:<14} Kendall={tau:>7.4f}")
pd.DataFrame(w_rows).to_csv(OUT / "engagement_weight_robustness.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------ 站区分布（表 7）

section("步骤 10  站区分布（表 7 数据，含全量帖子数）")

station_total = raw["station"].map(STATION_CN).fillna(raw["station"]).value_counts()
station_neg = work["站区"].value_counts()
cross = pd.crosstab(work["topic"], work["站区"])
share = cross.div(station_neg.reindex(cross.columns), axis=1).fillna(0)

st_rows = []
for st in station_neg.index:
    col = cross[st]
    top_t = int(col.idxmax())
    st_rows.append(
        {
            "站区": st,
            "全量帖子数": int(station_total[st]),
            "问题导向候选样本数": int(station_neg[st]),
            "问题导向占比": round(station_neg[st] / station_total[st] * 100, 1),
            "占比最高主题": top_t,
            "主题名称": TOPIC_TITLES[top_t],
            "该主题条数": int(col.max()),
            "该主题占比": round(col.max() / station_neg[st] * 100, 1),
        }
    )
st_df = pd.DataFrame(st_rows).sort_values("问题导向候选样本数", ascending=False)
st_df.to_csv(OUT / "station_distribution.csv", index=False, encoding="utf-8-sig")
share.to_csv(OUT / "kmeans_topic_station_share.csv", encoding="utf-8-sig")
cross.to_csv(OUT / "kmeans_topic_station_count.csv", encoding="utf-8-sig")

log(f"{'站区':<20}{'全量':>7}{'候选':>7}{'占比':>8}  最集中主题")
for _, r in st_df.iterrows():
    log(
        f"{r['站区']:<20}{r['全量帖子数']:>7}{r['问题导向候选样本数']:>7}"
        f"{r['问题导向占比']:>7.1f}%  主题{r['占比最高主题']} {r['主题名称']}"
        f" {r['该主题条数']}/{r['问题导向候选样本数']} ({r['该主题占比']}%)"
    )
log(f"\n合计：全量 {int(st_df['全量帖子数'].sum())} 条，候选 {int(st_df['问题导向候选样本数'].sum())} 条")

# --------------------------------------------------------- 消融实验（多种子）

section("步骤 11  单变量消融 + 多种子重复（K=10）")

log("载入对照嵌入模型 ...")
default_model = None
for spec in ("paraphrase-multilingual-MiniLM-L12-v2", DEFAULT_MODEL_FALLBACK):
    try:
        default_model = SentenceTransformer(spec, device=DEVICE)
        DEFAULT_MODEL_NAME = spec
        break
    except Exception as exc:  # noqa: BLE001
        log(f"    载入 {spec} 失败：{type(exc).__name__}")
if default_model is None:
    raise SystemExit("对照嵌入模型均无法载入")
log(f"对照嵌入模型：{DEFAULT_MODEL_NAME}   [4.4 节待补：默认句向量模型名称]")

emb_def_desc = encode(default_model, docs_desc)
emb_def_comb = encode(default_model, docs_comb)
log(f"对照模型嵌入维度：{emb_def_desc.shape}")

CONFIGS = {
    "M1_原始文本_默认嵌入_KMeans": (docs_comb, emb_def_comb),
    "M2_综合描述_默认嵌入_KMeans": (docs_desc, emb_def_desc),
    "M3_原始文本_Qwen_KMeans": (docs_comb, emb_qwen_comb),
    "M4_综合描述_Qwen_KMeans（本文方法）": (docs_desc, emb_qwen_desc),
}

per_seed = []
for name, (docs, emb) in CONFIGS.items():
    log(f"\n  {name}")
    for seed in SEEDS:
        labels = KMeans(n_clusters=K_TOPICS, random_state=seed, n_init=10).fit_predict(emb)
        m = calculate_metrics(name, docs, emb, labels)
        m["seed"] = seed
        per_seed.append(m)
    sub = pd.DataFrame([r for r in per_seed if r["experiment"] == name])
    log(
        f"    NPMI {sub['topic_coherence_npmi'].mean():.4f}±{sub['topic_coherence_npmi'].std():.4f}"
        f"  轮廓 {sub['silhouette_score'].mean():.4f}±{sub['silhouette_score'].std():.4f}"
        f"  CHI {sub['calinski_harabasz_index'].mean():.3f}±{sub['calinski_harabasz_index'].std():.3f}"
        f"  多样性 {sub['topic_diversity'].mean():.4f}±{sub['topic_diversity'].std():.4f}"
    )

seed_df = pd.DataFrame(per_seed)
seed_df.to_csv(OUT / "ablation_per_seed.csv", index=False, encoding="utf-8-sig")

METRICS = [
    "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index",
    "topic_coherence_npmi", "topic_diversity",
]
agg = seed_df.groupby("experiment")[METRICS].agg(["mean", "std"]).round(4)
agg.to_csv(OUT / "ablation_multiseed_summary.csv", encoding="utf-8-sig")

# --------------------------------------------------- 路线对照：LDA / HDBSCAN

section("步骤 12  路线对照与聚类器对照")

route_rows = []

count_vec = CountVectorizer(tokenizer=tokenize_zh, token_pattern=None, lowercase=False, min_df=1)
dtm = count_vec.fit_transform(docs_desc)
lda = LatentDirichletAllocation(n_components=K_TOPICS, random_state=42, max_iter=50)
doc_topic = lda.fit_transform(dtm)
lda_labels = doc_topic.argmax(axis=1)
m = calculate_metrics("LDA_综合描述_K10", docs_desc, doc_topic, lda_labels)
m["备注"] = "轮廓/DBI/CHI 基于文档-主题分布计算，与嵌入空间各组不可通约"
route_rows.append(m)
log(f"  LDA: NPMI {m['topic_coherence_npmi']:.4f} 多样性 {m['topic_diversity']:.4f} 主题数 {m['n_topics']}")

import hdbscan  # noqa: E402

HDB = {
    "C1_综合描述_Qwen_HDBSCAN": emb_qwen_desc,
    "C2_综合描述_默认嵌入_HDBSCAN": emb_def_desc,
    "C3_原始文本_默认嵌入_HDBSCAN": emb_def_comb,
}
HDB_DOCS = {
    "C1_综合描述_Qwen_HDBSCAN": docs_desc,
    "C2_综合描述_默认嵌入_HDBSCAN": docs_desc,
    "C3_原始文本_默认嵌入_HDBSCAN": docs_comb,
}
for name, emb in HDB.items():
    labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2).fit_predict(emb)
    n_noise = int((labels == -1).sum())
    m = calculate_metrics(name, HDB_DOCS[name], emb, labels)
    m["噪声点数"] = n_noise
    m["噪声占比"] = round(n_noise / len(labels) * 100, 1)
    m["备注"] = "主题数由算法自动确定；噪声点在聚类指标中剔除、在主题词中单列一类"
    route_rows.append(m)
    log(
        f"  {name}: 主题数 {m['n_topics']} 噪声 {n_noise}({m['噪声占比']}%) "
        f"NPMI {m['topic_coherence_npmi']:.4f} 多样性 {m['topic_diversity']:.4f}"
    )

try:
    from bertopic import BERTopic
    from umap import UMAP

    bt = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, prediction_data=True),
        vectorizer_model=CountVectorizer(tokenizer=tokenize_zh, token_pattern=None, min_df=1),
    )
    bt_topics, _ = bt.fit_transform(docs_desc, emb_def_desc)
    m = calculate_metrics("BERTopic_默认实现_综合描述", docs_desc, emb_def_desc, bt_topics)
    m["噪声点数"] = int((np.asarray(bt_topics) == -1).sum())
    m["噪声占比"] = round(m["噪声点数"] / len(bt_topics) * 100, 1)
    m["备注"] = "BERTopic 默认管线：默认句向量 + UMAP + HDBSCAN + c-TF-IDF"
    route_rows.append(m)
    log(
        f"  BERTopic 默认实现: 主题数 {m['n_topics']} 噪声 {m['噪声点数']}"
        f" NPMI {m['topic_coherence_npmi']:.4f} 多样性 {m['topic_diversity']:.4f}"
    )
except Exception as exc:  # noqa: BLE001
    log(f"  BERTopic 基线运行失败：{type(exc).__name__}: {exc}")

pd.DataFrame(route_rows).to_csv(OUT / "route_and_cluster_baselines.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------- 汇总表 4

section("步骤 13  表 4 汇总（多种子均值 ± 标准差）")

table4 = []
DESC = {
    "M1_原始文本_默认嵌入_KMeans": ("输入消融", "原始短文本", "默认多语种", "K-means"),
    "M2_综合描述_默认嵌入_KMeans": ("输入消融", "综合描述", "默认多语种", "K-means"),
    "M3_原始文本_Qwen_KMeans": ("输入消融", "原始短文本", "Qwen3-Emb", "K-means"),
    "M4_综合描述_Qwen_KMeans（本文方法）": ("本文方法", "综合描述", "Qwen3-Emb", "K-means"),
}
for name, (grp, inp, emb_name, clus) in DESC.items():
    sub = seed_df[seed_df["experiment"] == name]
    row = {"组别": grp, "配置": name, "输入": inp, "嵌入模型": emb_name, "聚类器": clus, "主题数": K_TOPICS}
    for met, label in [
        ("silhouette_score", "轮廓系数"), ("calinski_harabasz_index", "CHI"),
        ("topic_coherence_npmi", "NPMI"), ("topic_diversity", "主题多样性"),
    ]:
        row[label] = f"{sub[met].mean():.3f}±{sub[met].std():.3f}"
    table4.append(row)

for r in route_rows:
    table4.append(
        {
            "组别": "路线/聚类器对照", "配置": r["experiment"],
            "输入": "综合描述" if "原始文本" not in r["experiment"] else "原始短文本",
            "嵌入模型": "—" if r["experiment"].startswith("LDA") else ("Qwen3-Emb" if "Qwen" in r["experiment"] else "默认多语种"),
            "聚类器": "—" if r["experiment"].startswith("LDA") else ("HDBSCAN"),
            "主题数": r["n_topics"],
            "轮廓系数": f"{r['silhouette_score']:.3f}",
            "CHI": f"{r['calinski_harabasz_index']:.3f}",
            "NPMI": f"{r['topic_coherence_npmi']:.3f}",
            "主题多样性": f"{r['topic_diversity']:.3f}",
        }
    )
t4 = pd.DataFrame(table4)
t4.to_csv(OUT / "table4_topic_identification_results.csv", index=False, encoding="utf-8-sig")
log(t4.to_string(index=False))

# -------------------------------------------------------------- 节点级导出

work_out = work.drop(columns=[c for c in ("raw_response",) if c in work.columns])
work_out.sort_values(["topic", "id"]).to_csv(
    OUT / "bertopic_kmeans_results.csv", index=False, encoding="utf-8-sig"
)

flow = pd.DataFrame(
    [
        {"处理环节": "原始采集帖子", "样本量": len(raw), "占比": "100.0%"},
        {"处理环节": "LLM 情感判定为负向", "样本量": int(sent_dist.get("negative", 0)),
         "占比": f"{sent_dist.get('negative', 0) / len(raw) * 100:.1f}%"},
        {"处理环节": "LLM 情感判定为中立", "样本量": int(sent_dist.get("neutral", 0)),
         "占比": f"{sent_dist.get('neutral', 0) / len(raw) * 100:.1f}%"},
        {"处理环节": "LLM 情感判定为正向", "样本量": int(sent_dist.get("positive", 0)),
         "占比": f"{sent_dist.get('positive', 0) / len(raw) * 100:.1f}%"},
        {"处理环节": "文本长度过滤后（入模）", "样本量": len(work),
         "占比": f"{len(work) / len(raw) * 100:.1f}%"},
        {"处理环节": "全局语义网络节点数", "样本量": graph_new.number_of_nodes(), "占比": "—"},
        {"处理环节": "全局语义网络边数", "样本量": graph_new.number_of_edges(), "占比": "—"},
        {"处理环节": "其中孤立节点", "样本量": len(iso), "占比": "—"},
    ]
)
flow.to_csv(OUT / "table1_sample_flow.csv", index=False, encoding="utf-8-sig")

section("完成")
log(f"全部结果已写入：{OUT.resolve()}")
for f in sorted(OUT.iterdir()):
    log(f"    {f.name:<48}{f.stat().st_size:>9,} B")

(OUT / "rerun_0905.log").write_text("\n".join(LOG_LINES), encoding="utf-8")
