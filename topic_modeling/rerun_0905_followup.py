"""0905 补充检验：
1. 节点级判断 PageRank 是否比互动量提供增量信息（决定 3.5 节关注度指标的去留）；
2. LLM 情感标签与 BERT 情感标签的一致性（Cohen's Kappa），支撑 3.3 节口径改写。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
OUT = Path("output/0905")

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def section(t: str) -> None:
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


# ------------------------------------------------- 1. 节点级增量信息检验

section("检验 1  节点级：PageRank 相对互动量是否有增量信息")

df = pd.read_csv(OUT / "bertopic_kmeans_results.csv")
log(f"节点数：{len(df)}")

pr = df["kmeans_pagerank"]
eng = df["互动总量"]
log_eng = np.log1p(eng)

rows = []
for name, a, b in [
    ("方案A PageRank", pr, eng),
    ("方案A PageRank", log_eng, pr),
    ("主题内建网 PageRank", df["pagerank_主题内建网"], eng),
    ("方案A vs 主题内建网", pr, df["pagerank_主题内建网"]),
]:
    rho = spearmanr(a, b).statistic
    tau = kendalltau(a, b).statistic
    pear = float(np.corrcoef(a, b)[0, 1])
    rows.append({"对比": name, "Spearman": round(float(rho), 4), "Kendall": round(float(tau), 4), "Pearson": round(pear, 4)})

log("  节点级相关性：")
log(f"    PageRank(方案A) 与 互动总量      Spearman={rows[0]['Spearman']:.4f}  Pearson={rows[0]['Pearson']:.4f}")
log(f"    PageRank(方案A) 与 log(1+互动量) Spearman={rows[1]['Spearman']:.4f}  Pearson={rows[1]['Pearson']:.4f}")
log(f"    PageRank(旧实现) 与 互动总量     Spearman={rows[2]['Spearman']:.4f}  Pearson={rows[2]['Pearson']:.4f}")
log(f"    方案A 与 旧实现 PageRank         Spearman={rows[3]['Spearman']:.4f}  Pearson={rows[3]['Pearson']:.4f}")

log("\n  Top-K 榜单重合度（PageRank 排序 vs 互动量排序）：")
overlap_rows = []
for k in (10, 20, 30, 50):
    top_pr = set(df.nlargest(k, "kmeans_pagerank")["id"].astype(str))
    top_eng = set(df.nlargest(k, "互动总量")["id"].astype(str))
    inter = len(top_pr & top_eng)
    overlap_rows.append({"K": k, "重合条数": inter, "重合率": round(inter / k * 100, 1)})
    log(f"    Top-{k:<3} 重合 {inter:>3}/{k}  ({inter / k * 100:.1f}%)")

# 网络结构究竟改变了谁：找出排名变动最大的帖子
df["互动量排名"] = df["互动总量"].rank(ascending=False, method="min")
df["PageRank排名"] = df["kmeans_pagerank"].rank(ascending=False, method="min")
df["排名变动"] = df["互动量排名"] - df["PageRank排名"]
shift = df.reindex(df["排名变动"].abs().sort_values(ascending=False).index)

log("\n  排名变动最大的 8 条（正值=网络结构使其上升）：")
log(f"    {'主题':<5}{'互动量':>8}{'互动排名':>9}{'PR排名':>8}{'变动':>7}  摘要")
for _, r in shift.head(8).iterrows():
    txt = str(r["content"])[:34].replace("\n", " ")
    log(
        f"    {int(r['topic']):<5}{int(r['互动总量']):>8}{int(r['互动量排名']):>9}"
        f"{int(r['PageRank排名']):>8}{int(r['排名变动']):>+7}  {txt}"
    )

log(f"\n  排名变动绝对值：均值 {df['排名变动'].abs().mean():.1f} 位，中位数 {df['排名变动'].abs().median():.1f} 位，"
    f"最大 {int(df['排名变动'].abs().max())} 位")
log(f"  变动超过 20 位的帖子：{int((df['排名变动'].abs() > 20).sum())} 条 "
    f"（{(df['排名变动'].abs() > 20).mean() * 100:.1f}%）")

pd.DataFrame(rows).to_csv(OUT / "node_level_pagerank_check.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(overlap_rows).to_csv(OUT / "node_level_topk_overlap.csv", index=False, encoding="utf-8-sig")
shift[["id", "topic", "station", "互动总量", "互动量排名", "PageRank排名", "排名变动", "content"]].to_csv(
    OUT / "node_level_rank_shift.csv", index=False, encoding="utf-8-sig"
)

# 结论判定
node_rho = rows[0]["Spearman"]
log()
if abs(node_rho) > 0.95:
    log("  判定：节点级也高度共线，PageRank 未提供增量信息 → 建议弱化或删除该指标")
elif abs(node_rho) > 0.75:
    log("  判定：节点级中度相关，网络结构带来局部重排 → 可保留但须诚实报告共线性")
else:
    log("  判定：节点级相关性较低，网络结构贡献实质信息 → 指标可作为方法要点保留")


# ---------------------------------------- 2. LLM 与 BERT 情感标签一致性

section("检验 2  情感判定一致性（3.3 节口径改写依据）")


def normalize_llm_sentiment(value) -> str:
    label = str(value or "").strip().lower()
    if label in {"负向", "负面", "negative", "neg"} or "负" in label:
        return "negative"
    if label in {"正向", "正面", "positive", "pos"} or "正" in label:
        return "positive"
    if label in {"中立", "中性", "neutral", "neu"} or "中" in label:
        return "neutral"
    return "unknown"


records = []
for p in sorted(Path("llmOutput").glob("*/*_analysis.json")):
    for rec in json.load(p.open(encoding="utf-8")):
        if isinstance(rec, dict):
            records.append({**rec, "station": p.parent.name})
raw = pd.DataFrame(records)
raw["llm_group"] = raw["llm_sentiment"].map(normalize_llm_sentiment)

log(f"全量样本：{len(raw)} 条")
log("\n  BERT 模型仅输出二分类，无中立类：")
log(f"    {raw['sentiment_label'].value_counts().to_dict()}")
log(f"  BERT 置信度：均值 {raw['sentiment_score'].mean():.4f}，中位数 {raw['sentiment_score'].median():.4f}，"
    f"最小 {raw['sentiment_score'].min():.4f}")
low_conf = (raw["sentiment_score"] < 0.65).sum()
log(f"  置信度低于 0.65 的样本：{low_conf} 条（{low_conf / len(raw) * 100:.1f}%）"
    f" → 论文式(5) 的双阈值在本数据上几乎不产生中立类")

# 二值化后算 Kappa：是否负向
llm_bin = raw["llm_group"].eq("negative").astype(int)
bert_bin = raw["sentiment_label"].eq("Negative").astype(int)
kappa = cohen_kappa_score(llm_bin, bert_bin)
agree = float((llm_bin == bert_bin).mean())

log(f"\n  「是否负向」二值一致性：")
log(f"    一致率      {agree * 100:.1f}%")
log(f"    Cohen's Kappa {kappa:.4f}   [可直接填入 4.3 节人工/交叉验证表]")

cm = pd.crosstab(llm_bin.map({1: "LLM判负向", 0: "LLM判非负向"}),
                 bert_bin.map({1: "BERT判负向", 0: "BERT判非负向"}))
log("\n    混淆矩阵：")
for line in cm.to_string().splitlines():
    log("      " + line)

# 若改用 BERT 筛选会得到什么样本量
log(f"\n  两种筛选口径的样本量差异：")
log(f"    按 LLM 负向筛选（现行代码）：{int(llm_bin.sum())} 条")
log(f"    按 BERT 负向筛选（论文描述）：{int(bert_bin.sum())} 条")
log(f"    两者交集：{int((llm_bin & bert_bin).sum())} 条，并集：{int((llm_bin | bert_bin).sum())} 条")

pd.DataFrame(
    [{"一致率": round(agree, 4), "Cohen_Kappa": round(float(kappa), 4),
      "LLM负向数": int(llm_bin.sum()), "BERT负向数": int(bert_bin.sum()),
      "交集": int((llm_bin & bert_bin).sum()), "并集": int((llm_bin | bert_bin).sum())}]
).to_csv(OUT / "sentiment_agreement_kappa.csv", index=False, encoding="utf-8-sig")

# 站区级情感结构（可用于 5.1 节）
st_sent = pd.crosstab(raw["station"], raw["llm_group"], normalize="index").round(4) * 100
st_sent.to_csv(OUT / "station_sentiment_share.csv", encoding="utf-8-sig")
log("\n  各站区负向占比（5.1 节可用）：")
for st, v in st_sent["negative"].sort_values(ascending=False).items():
    log(f"    {st:<16}{v:>6.1f}%")

section("完成")
(OUT / "rerun_0905_followup.log").write_text("\n".join(LOG), encoding="utf-8")
log(f"结果已写入 {OUT.resolve()}")
