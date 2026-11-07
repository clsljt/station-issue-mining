"""论文实验的公共组件：数据载入、分词、主题指标、语义网络与 PageRank。

由 rerun_0905_supplement.py 等脚本共享，避免各脚本重复实现同一套口径。
所有函数的口径与 comp.ipynb / exp.ipynb 保持一致。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jieba
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "embModel/Qwen3-Embedding-0.6B"
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL_FALLBACK = str(ROOT / "embModel/BAAI/bge-small-zh-v1.5")
KEYWORD_FILTER_CONFIG_PATH = ROOT / "config/station_keyword_filter.json"
STOPWORD_PATHS = [ROOT / "stopword.txt", ROOT / "stopinthis.txt"]
LLM_OUTPUT_DIR = ROOT / "llmOutput"

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

ENGAGEMENT_COLUMNS = ["liked_count", "collected_count", "comment_count", "share_count"]


# ------------------------------------------------------------------ 分词资源


def _load_stopwords() -> tuple[set[str], set[str]]:
    stop_words = {"无标题"}
    for path in STOPWORD_PATHS:
        if path.is_file():
            stop_words.update(
                w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()
            )
    with KEYWORD_FILTER_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    exclude = {str(w).strip() for w in config.get("exclude_words", []) if str(w).strip()}
    prefer = {str(w).strip() for w in config.get("prefer_words", []) if str(w).strip()}
    stop_words.update(exclude)
    return stop_words, prefer


STOP_WORDS, PREFER_WORDS = _load_stopwords()


def tokenize_zh(text: str) -> list[str]:
    return [
        w.strip()
        for w in jieba.lcut(str(text))
        if len(w.strip()) > 1 and not w.strip().isdigit() and w.strip() not in STOP_WORDS
    ]


def tokenize_chinese_only(text: str) -> list[str]:
    return [
        w.strip()
        for w in jieba.lcut(str(text))
        if CHINESE_WORD.fullmatch(w.strip()) and w.strip() not in STOP_WORDS
    ]


# ------------------------------------------------------------------ 主题指标


def topic_diversity_score(topic_word_lists: list[list[str]]) -> float:
    all_words = [w for words in topic_word_lists for w in words]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


class NpmiScorer:
    """预先缓存文档词集与词频，避免在多次调用间重复分词。

    直接复用 comp.ipynb 的 topic_coherence_npmi 逻辑，仅把分词结果缓存下来；
    在 K 值扫描等需要上百次调用的场景下，这一缓存是必要的。
    """

    def __init__(self, docs: list[str], eps: float = 1e-12) -> None:
        self.eps = eps
        self.tokenized = [set(tokenize_zh(d)) for d in docs]
        self.n_docs = len(self.tokenized)
        self.doc_freq: dict[str, int] = {}
        for tokens in self.tokenized:
            for t in tokens:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        self._co_cache: dict[tuple[str, str], int] = {}

    def _co_doc_count(self, wi: str, wj: str) -> int:
        key = (wi, wj) if wi <= wj else (wj, wi)
        if key not in self._co_cache:
            self._co_cache[key] = sum(
                1 for tk in self.tokenized if key[0] in tk and key[1] in tk
            )
        return self._co_cache[key]

    def score(self, topic_word_lists: list[list[str]]) -> float:
        if self.n_docs == 0:
            return float("nan")
        pair_scores = []
        for words in topic_word_lists:
            words = [w for w in dict.fromkeys(words) if self.doc_freq.get(w, 0) > 0]
            if len(words) < 2:
                continue
            for i, wi in enumerate(words[:-1]):
                for wj in words[i + 1 :]:
                    pi = self.doc_freq[wi] / self.n_docs
                    pj = self.doc_freq[wj] / self.n_docs
                    pij = (self._co_doc_count(wi, wj) + self.eps) / self.n_docs
                    pair_scores.append(np.log(pij / (pi * pj)) / (-np.log(pij)))
        return float(np.mean(pair_scores)) if pair_scores else float("nan")


def extract_topic_words(docs: list[str], topics, top_n: int = 10) -> dict:
    """c-TF-IDF 主题词提取，口径与 comp.ipynb 的 calculate_metrics 一致。"""
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
            if term in PREFER_WORDS:
                weighted[idx] *= 1.35
        top_idx = [i for i in weighted.argsort()[::-1] if row[i] > 0][:top_n]
        topic_word_map[topic_id] = [terms[i] for i in top_idx]
    return topic_word_map


def cluster_metrics(embeddings, labels) -> dict:
    """轮廓系数、DBI、CHI，噪声点（标签 -1）在计算前剔除。"""
    labels = np.asarray(labels)
    mask = labels != -1
    x, y = np.asarray(embeddings)[mask], labels[mask]
    out = {}
    if len(np.unique(y)) < 2 or len(y) <= len(np.unique(y)):
        return {k: float("nan") for k in ("silhouette_score", "davies_bouldin_index", "calinski_harabasz_index")}
    for name, fn, kw in (
        ("silhouette_score", silhouette_score, {"metric": "cosine"}),
        ("davies_bouldin_index", davies_bouldin_score, {}),
        ("calinski_harabasz_index", calinski_harabasz_score, {}),
    ):
        try:
            out[name] = float(fn(x, y, **kw))
        except ValueError:
            out[name] = float("nan")
    return out


def evaluate_partition(
    name: str, docs: list[str], embeddings, labels, npmi_scorer: NpmiScorer
) -> dict:
    """对一次聚类划分计算全部五项指标。"""
    labels = np.asarray(labels)
    topic_ids = np.unique(labels)
    word_map = extract_topic_words(docs, labels)
    word_lists = [word_map[t] for t in sorted(topic_ids) if word_map.get(t)]
    metrics = {
        "experiment": name,
        "n_documents": int(len(docs)),
        "n_topics": int(len(topic_ids)),
        **cluster_metrics(embeddings, labels),
        "topic_coherence_npmi": npmi_scorer.score(word_lists),
        "topic_diversity": topic_diversity_score(word_lists),
    }
    return metrics


# -------------------------------------------------------------- 网络与 PageRank


def percentile_score(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    if series.nunique(dropna=False) <= 1:
        return pd.Series(50.0, index=series.index)
    return series.rank(method="average", pct=True) * 100


def build_global_graph(
    similarity_matrix,
    top_k: int = SIMILARITY_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> nx.Graph:
    """全局语义网络（方案 A）：不以主题为分组，跨主题连边同样保留。"""
    matrix = np.asarray(similarity_matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(matrix)))
    for i in range(len(matrix)):
        for j in matrix[i].argsort()[::-1][:top_k]:
            sim = float(matrix[i, j])
            if i == int(j) or sim < threshold:
                continue
            graph.add_edge(i, int(j), weight=sim)
    return graph


def personalized_pagerank(
    graph: nx.Graph, engagement: pd.Series, alpha: float = ALPHA
) -> pd.Series:
    raw_p = np.log1p(engagement.astype(float).reset_index(drop=True))
    if float(raw_p.sum()) == 0:
        personalization = {i: 1 / len(engagement) for i in range(len(engagement))}
    else:
        personalization = (raw_p / raw_p.sum()).to_dict()
    scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
        max_iter=200,
        tol=1e-8,
    )
    return pd.Series(scores).sort_index()


# ------------------------------------------------------------------ 数据载入


def normalize_llm_sentiment(value) -> str:
    label = str(value or "").strip().lower()
    if label in {"负向", "负面", "negative", "neg"} or "负" in label:
        return "negative"
    if label in {"正向", "正面", "positive", "pos"} or "positive" in label or "正" in label:
        return "positive"
    if label in {"中立", "中性", "neutral", "neu"} or "中" in label:
        return "neutral"
    return "unknown"


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


def load_raw() -> pd.DataFrame:
    """载入 llmOutput 下全部站区的分析结果，附加 station 与情感三分类字段。"""
    paths = sorted(LLM_OUTPUT_DIR.glob("*/*_analysis.json"))
    if not paths:
        raise FileNotFoundError(f"未找到分析文件：{LLM_OUTPUT_DIR}")
    records = []
    for path in paths:
        station = path.parent.name
        for record in json.load(path.open(encoding="utf-8")):
            if isinstance(record, dict):
                records.append({**record, "station": station})
    raw = pd.DataFrame(records)
    raw["llm_sentiment_group"] = raw["llm_sentiment"].map(normalize_llm_sentiment)
    return raw


def build_sample(raw: pd.DataFrame, keep_mask: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    """按给定筛选条件构造入模样本，返回 (样本表, 综合描述文本, 标题+正文文本)。

    口径与 comp.ipynb 一致：综合描述缺失时回退为原帖正文，再施加长度下限。
    """
    subset = raw.loc[keep_mask].copy()
    if "content" in subset.columns:
        subset = subset.drop(columns=["content"])

    description = subset["description"].fillna("").astype(str).str.strip()
    post_text = raw.loc[keep_mask, "content"].fillna("").astype(str).str.strip()
    subset["source_content"] = post_text
    subset["analysis_text"] = description.where(description.ne(""), post_text)

    clean = subset["analysis_text"].fillna("").astype(str).str.strip()
    keep = clean.ne("") & clean.ne("无标题") & clean.str.len().ge(MIN_TEXT_LENGTH)

    work = (
        subset.loc[keep]
        .rename(columns={"note_id": "id", "analysis_text": "content"})
        .reset_index(drop=True)
    )
    for col in ENGAGEMENT_COLUMNS:
        work[f"{col}_num"] = work[col].map(parse_count) if col in work else 0.0
    work["互动总量"] = sum(work[f"{c}_num"] for c in ENGAGEMENT_COLUMNS)
    work["站区"] = work["station"].map(STATION_CN).fillna(work["station"])

    docs_desc = clean.loc[keep].tolist()

    title = raw.loc[keep_mask, "title"].fillna("").astype(str).str.strip()
    body = raw.loc[keep_mask, "content"].fillna("").astype(str).str.strip()
    docs_comb = (title + " " + body).str.strip().loc[keep].tolist()

    return work, docs_desc, docs_comb


def load_embedder(spec: str | None = None, device: str | None = None):
    """载入句向量模型；spec 为 None 时使用本文的 Qwen3-Embedding-0.6B。"""
    from sentence_transformers import SentenceTransformer

    if device is None:
        try:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if spec is None:
        return SentenceTransformer(
            str(MODEL_PATH), device=device, trust_remote_code=True, local_files_only=True
        ), device

    try:
        return SentenceTransformer(spec, device=device), device
    except Exception:
        return SentenceTransformer(DEFAULT_MODEL_FALLBACK, device=device), device


def encode(model, texts: list[str]):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def setup_cjk_font() -> str:
    """配置 matplotlib 的中文字体与负号显示。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("PingFang SC", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "STHeiti"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    raise RuntimeError("未找到可用的中文字体，请先安装 PingFang SC 或 Heiti TC")
