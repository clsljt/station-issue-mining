"""Run the standard Chinese BERTopic baseline on the existing 293 paper samples.

Uses BERTopic's multilingual embedding backend and default UMAP/HDBSCAN/c-TF-IDF.
Only the vectorizer is adapted for Chinese word segmentation and stopwords.
Run with /opt/anaconda3/envs/py312/bin/python scripts/run_bertopic_293_default.py.
"""
from __future__ import annotations

import hashlib
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/station_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/station_mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jieba
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/1020/bertopic_293_default"
SOURCE = ROOT / "output/0905/bertopic_kmeans_results.csv"
STOPWORDS = set()
for filename in ("stopword.txt", "stopinthis.txt"):
    STOPWORDS.update((ROOT / filename).read_text(encoding="utf-8").splitlines())


def tokenize(text):
    return [w for w in jieba.lcut(text) if len(w.strip()) > 1
            and not w.isdigit() and w not in STOPWORDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--cluster-selection-method", choices=("eom", "leaf"), default="eom")
    args = parser.parse_args()
    output_dir = args.output_dir
    run(output_dir, args)


def run(OUT, args):
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    data = pd.read_csv(SOURCE, dtype={"id": str}, keep_default_na=False)
    docs = data["content"].astype(str).tolist()
    assert len(docs) == 293 and all(d.strip() for d in docs)
    print(f"Input: {len(docs)} rows, {data.id.nunique()} unique IDs; preserve existing sample.", flush=True)

    extra = {}
    tuned = args.min_topic_size != 10 or args.min_samples is not None or args.cluster_selection_method != "eom"
    if tuned:
        from hdbscan import HDBSCAN
        extra["hdbscan_model"] = HDBSCAN(
            min_cluster_size=args.min_topic_size, min_samples=args.min_samples,
            cluster_selection_method=args.cluster_selection_method, prediction_data=True,
        )
    model = BERTopic(
        language="multilingual",
        min_topic_size=args.min_topic_size,
        vectorizer_model=CountVectorizer(tokenizer=tokenize, token_pattern=None),
        verbose=True,
        **extra,
    )
    topics, probabilities = model.fit_transform(docs)
    info = model.get_topic_info()
    ids = sorted(t for t in set(topics) if t != -1)
    n_noise = sum(t == -1 for t in topics)
    info.to_csv(OUT / "topic_info.csv", index=False, encoding="utf-8-sig")
    assignments = data[["id", "station", "title", "source_content", "content"]].copy()
    assignments.insert(0, "sample_row", np.arange(len(data)))
    assignments["topic"] = topics
    assignments["topic_probability"] = probabilities
    assignments.to_csv(OUT / "topic_assignments.csv", index=False, encoding="utf-8-sig")
    rows = [dict(topic=t, rank=r, keyword=w, weight=float(v))
            for t in sorted(set(topics))
            for r, (w, v) in enumerate(model.get_topic(t), 1)]
    pd.DataFrame(rows).to_csv(OUT / "topic_keywords.csv", index=False, encoding="utf-8-sig")
    reps = [dict(topic=t, rank=r, content=d)
            for t in ids for r, d in enumerate(model.get_representative_docs(t), 1)]
    pd.DataFrame(reps).to_csv(OUT / "representative_docs.csv", index=False, encoding="utf-8-sig")

    native = model.visualize_barchart(topics=ids, n_words=8, title="BERTopic 主题关键词（293条样本）")
    native.write_html(str(OUT / "fig3_bertopic_keywords.html"), include_plotlyjs=True)

    # Export the same c-TF-IDF weights as a paper-ready static figure.
    from paper_common import setup_cjk_font
    font = setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cols = min(4, len(ids))
    nrows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 4.8, nrows * 3.65), squeeze=False)
    colors = ["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73"]
    for ax, t in zip(axes.flat, ids):
        words, scores = zip(*model.get_topic(t)[:8])
        ax.set_facecolor("#E8EDF5")
        ax.barh(range(len(words)), scores, color=colors[t % len(colors)], height=.74)
        ax.set_yticks(range(len(words)), words, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"主题 {t}（n={topics.count(t)}）", fontsize=11, pad=8)
        ax.set_xlabel("c-TF-IDF 权重", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.grid(axis="x", color="white", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    fig.suptitle("BERTopic 主题关键词", fontsize=15, y=.998)
    fig.text(.5, .006, f"293条样本 · {len(ids)}个主题 · 离群点 {n_noise}条（{n_noise / 293:.1%}，未绘入主题面板）",
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, .025, 1, .975), h_pad=2.0, w_pad=2.4)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig3_bertopic_keywords.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    config = dict(
        parameter_variant="finer topics" if tuned else "default",
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        input_text="content: existing integrated descriptions", n_documents=len(docs), unique_ids=data.id.nunique(),
        duplicates="293 existing rows retained, no deduplication", language="multilingual",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        n_topics=len(ids), n_outliers=n_noise, outlier_fraction=n_noise / len(docs),
        umap={k: v for k, v in model.umap_model.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))},
        hdbscan={k: v for k, v in model.hdbscan_model.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))},
        vectorizer="jieba, token length >1, project stopword.txt + stopinthis.txt; unigram; no keyword boosts",
        min_topic_size=model.min_topic_size, nr_topics=model.nr_topics,
        numpy_seed=42, numba_threads=1, font=font,
        versions={p: importlib.metadata.version(p) for p in ("bertopic", "sentence-transformers", "umap-learn", "hdbscan", "scikit-learn", "jieba")},
    )
    def finite_values(value):
        if isinstance(value, dict):
            return {k: finite_values(v) for k, v in value.items()}
        return str(value) if isinstance(value, float) and not math.isfinite(value) else value
    (OUT / "run_config.json").write_text(json.dumps(finite_values(config), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    clustering_note = (
        f"为细分主题调整HDBSCAN：min_cluster_size={args.min_topic_size}，min_samples={args.min_samples}，"
        f"cluster_selection_method={args.cluster_selection_method}；UMAP保持默认。\n"
        if tuned else "UMAP和HDBSCAN使用默认参数。\n"
    )
    (OUT / "README.txt").write_text(
        f"原版 BERTopic 中文基线\n输入：既有论文实验293条综合描述（291个唯一帖子ID），原样保留样本行。\n"
        f"结果：{len(ids)}个主题，离群点{n_noise}条（{n_noise/293:.1%}）。\n"
        "使用BERTopic官方多语言默认嵌入和c-TF-IDF；为中文配置jieba分词与已有停用词。\n"
        + clustering_note +
        "没有固定主题数、合并主题、回填离群点或人工调整关键词。条形图不画离群主题-1，CSV保留。\n"
        "与原图TF-IDF权重的算法口径不同，本图使用BERTopic原生c-TF-IDF权重。\n"
        f"重跑：/opt/anaconda3/envs/py312/bin/python scripts/run_bertopic_293_default.py --output-dir {OUT} "
        f"--min-topic-size {args.min_topic_size} --cluster-selection-method {args.cluster_selection_method}"
        + (f" --min-samples {args.min_samples}" if args.min_samples is not None else "") + "\n", encoding="utf-8")
    print(info[["Topic", "Count", "Name"]].to_string(index=False), flush=True)
    print(f"DONE: {len(ids)} topics, {n_noise} outliers. {OUT}", flush=True)


if __name__ == "__main__":
    main()
