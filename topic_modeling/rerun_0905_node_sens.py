"""0905 第三轮：节点级参数敏感性 + 各主题代表性帖子。

主题级敏感性因质量守恒而恒等于 1，不构成稳健性证据；
真正需要检验的是节点级排序对 top_k / threshold / alpha 的敏感度。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
OUT = Path("output/0905")

SIMILARITY_TOP_K = 8
SIMILARITY_THRESHOLD = 0.42
ALPHA = 0.85

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def section(t: str) -> None:
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


TOPIC_TITLES = {
    0: "候车环境与进站体验", 1: "丰台站出站换乘动线", 2: "机场运行延误与旅客滞留",
    3: "首都机场航站楼服务与接驳体验", 4: "网约车接驳与运力不足", 5: "列车晚点与候车滞留",
    6: "通州站空间布局与通行体验", 7: "机场线票价与接驳成本", 8: "站台吸烟与空气环境",
    9: "目的车站混淆与改签困扰",
}

df = pd.read_csv(OUT / "bertopic_kmeans_results.csv").reset_index(drop=True)
docs = df["content"].fillna("").astype(str).tolist()

from sentence_transformers import SentenceTransformer  # noqa: E402

try:
    import torch

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

model = SentenceTransformer(
    "embModel/Qwen3-Embedding-0.6B", device=DEVICE, trust_remote_code=True, local_files_only=True
)
emb = model.encode(docs, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                   show_progress_bar=False)
sim = emb @ emb.T
np.fill_diagonal(sim, 0)


def build_global_graph(matrix, top_k: int, threshold: float) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(matrix)))
    for i in range(len(matrix)):
        for j in matrix[i].argsort()[::-1][:top_k]:
            s = float(matrix[i, j])
            if i == j or s < threshold:
                continue
            g.add_edge(i, int(j), weight=s)
    return g


def ppr(g: nx.Graph, engagement: pd.Series, alpha: float = ALPHA) -> pd.Series:
    p = np.log1p(engagement.astype(float).reset_index(drop=True))
    pers = ({i: 1 / len(engagement) for i in range(len(engagement))}
            if float(p.sum()) == 0 else (p / p.sum()).to_dict())
    return pd.Series(
        nx.pagerank(g, alpha=alpha, personalization=pers, weight="weight", max_iter=200, tol=1e-8)
    ).sort_index()


# ------------------------------------------------ 节点级参数敏感性

section("检验 3  节点级排序对网络参数的敏感性")

base_g = build_global_graph(sim, SIMILARITY_TOP_K, SIMILARITY_THRESHOLD)
base = ppr(base_g, df["互动总量"])
base_rank = base.rank(ascending=False)
log(f"基准配置 top_k={SIMILARITY_TOP_K}, threshold={SIMILARITY_THRESHOLD}, alpha={ALPHA}："
    f"{base_g.number_of_edges():,} 条边")

rows = []
for k in (5, 8, 12, 20):
    for th in (0.35, 0.42, 0.50):
        g = build_global_graph(sim, k, th)
        s = ppr(g, df["互动总量"])
        tau = float(kendalltau(base_rank, s.rank(ascending=False)).statistic)
        rho = float(spearmanr(base_rank, s.rank(ascending=False)).statistic)
        top20 = len(set(base.nlargest(20).index) & set(s.nlargest(20).index))
        rows.append({"参数": f"top_k={k}, th={th}", "top_k": k, "threshold": th, "alpha": ALPHA,
                     "边数": g.number_of_edges(), "Kendall": round(tau, 4),
                     "Spearman": round(rho, 4), "Top20重合": top20})
        log(f"    top_k={k:<3} th={th:<5} 边数={g.number_of_edges():>5}  "
            f"Kendall={tau:>7.4f}  Spearman={rho:>7.4f}  Top20重合={top20}/20")

log()
for a in (0.70, 0.85, 0.95):
    s = ppr(base_g, df["互动总量"], alpha=a)
    tau = float(kendalltau(base_rank, s.rank(ascending=False)).statistic)
    top20 = len(set(base.nlargest(20).index) & set(s.nlargest(20).index))
    rows.append({"参数": f"alpha={a}", "top_k": SIMILARITY_TOP_K, "threshold": SIMILARITY_THRESHOLD,
                 "alpha": a, "边数": base_g.number_of_edges(), "Kendall": round(tau, 4),
                 "Spearman": round(float(spearmanr(base_rank, s.rank(ascending=False)).statistic), 4),
                 "Top20重合": top20})
    log(f"    alpha={a:<5} Kendall={tau:>7.4f}  Top20重合={top20}/20")

sens = pd.DataFrame(rows)
sens.to_csv(OUT / "node_level_param_sensitivity.csv", index=False, encoding="utf-8-sig")
log(f"\n  全部 {len(sens)} 组配置：Kendall 最小 {sens['Kendall'].min():.4f}，"
    f"中位 {sens['Kendall'].median():.4f}；Top20 重合最小 {sens['Top20重合'].min()}/20")

# PageRank 与网络度中心性的关系：说明它到底在测什么
deg = pd.Series(dict(base_g.degree(weight="weight"))).sort_index()
log()
log("  PageRank 究竟由什么驱动：")
log(f"    与加权度中心性 Spearman = {spearmanr(base, deg).statistic:.4f}")
log(f"    与互动总量     Spearman = {spearmanr(base, df['互动总量']).statistic:.4f}")
log(f"    与所属主题规模 Spearman = {spearmanr(base, df['topic'].map(df['topic'].value_counts())).statistic:.4f}")

# ------------------------------------------------ 各主题代表性帖子

section("检验 4  各主题代表性帖子（供 5.2 节案例分析引用）")

df["_pr"] = base.to_numpy()
reps = []
for t in sorted(df["topic"].unique()):
    sub = df[df["topic"] == t].nlargest(3, "_pr")
    for rank, (_, r) in enumerate(sub.iterrows(), 1):
        reps.append({
            "topic": t, "主题标题": TOPIC_TITLES[t], "代表性排序": rank,
            "id": r["id"], "站区": r["站区"], "互动总量": int(r["互动总量"]),
            "语义代表性百分位": round(float(base.rank(pct=True).iloc[int(r.name)] * 100), 1),
            "摘要": str(r["content"])[:120].replace("\n", " "),
        })
rep_df = pd.DataFrame(reps)
rep_df.to_csv(OUT / "topic_representative_posts.csv", index=False, encoding="utf-8-sig")

for t in sorted(df["topic"].unique()):
    log(f"\n  主题 {t} {TOPIC_TITLES[t]}")
    for _, r in rep_df[rep_df["topic"] == t].iterrows():
        log(f"    [{r['代表性排序']}] {r['站区']}  互动 {r['互动总量']:>5}  {r['摘要'][:60]}")

section("完成")
(OUT / "rerun_0905_node_sens.log").write_text("\n".join(LOG), encoding="utf-8")
log(f"结果已写入 {OUT.resolve()}")
