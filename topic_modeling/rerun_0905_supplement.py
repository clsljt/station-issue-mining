"""0905 补充实验：填补 4.5、4.7 节剩余空缺并绘制图 3、图 6、图 7。

四项内容：
1. K 值扫描补入 NPMI 与主题多样性（原扫描只有聚类结构指标，与 4.2 节的主要指标口径不一致），
   10 个随机种子重复，并给出可复现的综合得分规则；
2. 聚类稳定性 ARI（4.7 节第 3 项）；
3. 情感筛选口径敏感性（4.7 节第 1 项）：按 MLLM 负向、BERT 负向、两者交集三套样本重跑；
图 3、图 6、图 7 的绘制见 plot_paper_figures.py，该脚本从本脚本产出的结果表读数。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import paper_common as pc

OUT = pc.ROOT / "output/0905"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

K_RANGE = list(range(5, 16))
FRAGMENT_SIZE = 10


LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def section(title: str) -> None:
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ------------------------------------------------------------------ 数据准备

section("准备：载入数据与语义向量")

raw = pc.load_raw()
log(f"原始记录 {len(raw):,} 条")

mask_llm = raw["llm_sentiment_group"].eq("negative")
mask_bert = raw["sentiment_label"].eq("Negative")

work, docs_desc, docs_comb = pc.build_sample(raw, mask_llm)
log(f"基准样本（MLLM 判负向）：{len(work)} 条")

qwen, device = pc.load_embedder()
log(f"推理设备：{device}")
emb_desc = pc.encode(qwen, docs_desc)
emb_comb = pc.encode(qwen, docs_comb)
log(f"Qwen 嵌入：{emb_desc.shape}")

default_model, _ = pc.load_embedder(pc.DEFAULT_MODEL_NAME, device)
emb_def_desc = pc.encode(default_model, docs_desc)
emb_def_comb = pc.encode(default_model, docs_comb)
log(f"对照嵌入：{emb_def_desc.shape}")

npmi_desc = pc.NpmiScorer(docs_desc)
npmi_comb = pc.NpmiScorer(docs_comb)
log("NPMI 共现缓存构建完成")


# ------------------------------------------------- 1. K 值扫描（含 NPMI 与多样性）

section("实验 1  K 值扫描：K ∈ [5, 15]，10 个随机种子，五项指标全评")

scan_rows = []
for k in K_RANGE:
    for seed in pc.SEEDS:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb_desc)
        m = pc.evaluate_partition(f"K={k}", docs_desc, emb_desc, labels, npmi_desc)
        sizes = pd.Series(labels).value_counts()
        m.update(
            {
                "K": k,
                "seed": seed,
                "碎片主题数": int((sizes < FRAGMENT_SIZE).sum()),
                "最小主题占比": float(sizes.min() / len(labels) * 100),
                "最大主题占比": float(sizes.max() / len(labels) * 100),
            }
        )
        scan_rows.append(m)
    sub = pd.DataFrame([r for r in scan_rows if r["K"] == k])
    log(
        f"  K={k:<3} 轮廓 {sub['silhouette_score'].mean():.4f}±{sub['silhouette_score'].std():.4f}"
        f"  DBI {sub['davies_bouldin_index'].mean():.3f}"
        f"  CHI {sub['calinski_harabasz_index'].mean():.3f}"
        f"  NPMI {sub['topic_coherence_npmi'].mean():.4f}±{sub['topic_coherence_npmi'].std():.4f}"
        f"  多样性 {sub['topic_diversity'].mean():.4f}"
        f"  碎片 {sub['碎片主题数'].mean():.1f}"
    )

scan = pd.DataFrame(scan_rows)
scan.to_csv(OUT / "k_selection_per_seed.csv", index=False, encoding="utf-8-sig")

METRICS = [
    ("silhouette_score", "轮廓系数", True),
    ("davies_bouldin_index", "DBI", False),
    ("calinski_harabasz_index", "CHI", True),
    ("topic_coherence_npmi", "NPMI", True),
    ("topic_diversity", "主题多样性", True),
]

agg = scan.groupby("K").agg(
    **{f"{col}_mean": (col, "mean") for col, _, _ in METRICS},
    **{f"{col}_std": (col, "std") for col, _, _ in METRICS},
    碎片主题数=("碎片主题数", "mean"),
    最小主题占比=("最小主题占比", "mean"),
    最大主题占比=("最大主题占比", "mean"),
).reset_index()

agg["碎片主题率"] = (
    scan.assign(有碎片=scan["碎片主题数"] > 0).groupby("K")["有碎片"].mean().to_numpy() * 100
)
agg = agg.sort_values("K")
agg.to_csv(OUT / "k_selection_summary.csv", index=False, encoding="utf-8-sig")

log()
log(f"{'K':<4}{'轮廓':>8}{'DBI':>8}{'CHI':>9}{'NPMI':>9}{'多样性':>9}{'碎片数':>8}{'出现碎片种子':>13}{'最小占比':>10}")
for _, r in agg.iterrows():
    log(
        f"{int(r['K']):<4}{r['silhouette_score_mean']:>8.4f}{r['davies_bouldin_index_mean']:>8.3f}"
        f"{r['calinski_harabasz_index_mean']:>9.3f}{r['topic_coherence_npmi_mean']:>9.4f}"
        f"{r['topic_diversity_mean']:>9.4f}{r['碎片主题数']:>8.1f}{r['碎片主题率']:>12.0f}%"
        f"{r['最小主题占比']:>9.2f}%"
    )

log()
log("各指标单独取极值给出的最优 K（互相冲突，说明单一指标无法定 K）：")
for col, label, higher in METRICS:
    idx = agg[f"{col}_mean"].idxmax() if higher else agg[f"{col}_mean"].idxmin()
    log(f"    {label:<10} K={int(agg.loc[idx, 'K']):<3} 取值 {agg.loc[idx, f'{col}_mean']:.4f}")

# 指标随 K 的单调性检验：若某指标随 K 单调变化，则对其取极值等同于取 K 的端点，
# 该指标就不具备定 K 的能力。这是本文不以指标极值定 K 的直接依据。
log()
log("指标随 K 的单调性（Spearman 秩相关，基于 11 个 K 的均值）：")
trend_rows = []
for col, label, _ in METRICS + [("碎片主题数", "碎片主题数", True), ("最小主题占比", "最小主题占比", True)]:
    key = f"{col}_mean" if f"{col}_mean" in agg.columns else col
    res = spearmanr(agg["K"], agg[key])
    verdict = "单调" if res.pvalue < 0.05 else "无显著趋势"
    trend_rows.append({"指标": label, "Spearman_rho": round(float(res.statistic), 3),
                       "p值": round(float(res.pvalue), 4), "判定": verdict})
    log(f"    {label:<12} rho={res.statistic:+.3f}  p={res.pvalue:.4f}  {verdict}")
pd.DataFrame(trend_rows).to_csv(
    OUT / "k_selection_trend_test.csv", index=False, encoding="utf-8-sig"
)

# K=10 与几何指标最优的 K=12 逐种子配对比较
log()
log("K=10 与 K=12 的逐种子配对比较（同一种子下同批数据，Wilcoxon 符号秩检验）：")
for col, label, _ in METRICS:
    pivot = scan[scan["K"].isin([10, 12])].pivot(index="seed", columns="K", values=col)
    res = wilcoxon(pivot[10], pivot[12])
    log(f"    {label:<10} 差值（K12−K10）{pivot[12].mean() - pivot[10].mean():+.4f}  p={res.pvalue:.4f}")
log(f"    K=10 出现碎片主题的种子 {int((scan[scan['K'] == 10]['碎片主题数'] > 0).sum())}/10，"
    f"K=12 为 {int((scan[scan['K'] == 12]['碎片主题数'] > 0).sum())}/10")


# ----------------------------------------------------- 2. 聚类稳定性 ARI

section("实验 2  聚类稳定性 ARI（4.7 节第 3 项）")

CONFIGS = {
    "M1_原始文本_默认嵌入": (docs_comb, emb_def_comb, npmi_comb),
    "M2_综合描述_默认嵌入": (docs_desc, emb_def_desc, npmi_desc),
    "M3_原始文本_Qwen": (docs_comb, emb_comb, npmi_comb),
    "M4_综合描述_Qwen（本文方法）": (docs_desc, emb_desc, npmi_desc),
}

ari_rows = []
for name, (_, emb, _) in CONFIGS.items():
    partitions = {
        seed: KMeans(n_clusters=pc.K_TOPICS, random_state=seed, n_init=10).fit_predict(emb)
        for seed in pc.SEEDS
    }
    pairs = [
        adjusted_rand_score(partitions[a], partitions[b])
        for a, b in itertools.combinations(pc.SEEDS, 2)
    ]
    ref = [adjusted_rand_score(partitions[42], partitions[s]) for s in pc.SEEDS if s != 42]
    ari_rows.append(
        {
            "配置": name,
            "两两ARI均值": round(float(np.mean(pairs)), 4),
            "两两ARI标准差": round(float(np.std(pairs, ddof=1)), 4),
            "两两ARI最小": round(float(np.min(pairs)), 4),
            "两两ARI最大": round(float(np.max(pairs)), 4),
            "对比数": len(pairs),
            "种子42与其余均值": round(float(np.mean(ref)), 4),
        }
    )
    log(
        f"  {name:<28} 两两 ARI {np.mean(pairs):.4f}±{np.std(pairs, ddof=1):.4f}"
        f"  区间 [{np.min(pairs):.4f}, {np.max(pairs):.4f}]"
    )

ari = pd.DataFrame(ari_rows)
ari.to_csv(OUT / "cluster_stability_ari.csv", index=False, encoding="utf-8-sig")

# 主方法在不同 K 下的稳定性，用于说明 K=10 的选择是否处在稳定区间
log()
log("本文方法在不同 K 下的两两 ARI（说明 K 的选择是否落在稳定区间）：")
ari_k_rows = []
for k in K_RANGE:
    partitions = {
        seed: KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb_desc)
        for seed in pc.SEEDS
    }
    pairs = [
        adjusted_rand_score(partitions[a], partitions[b])
        for a, b in itertools.combinations(pc.SEEDS, 2)
    ]
    ari_k_rows.append(
        {"K": k, "两两ARI均值": round(float(np.mean(pairs)), 4),
         "两两ARI标准差": round(float(np.std(pairs, ddof=1)), 4)}
    )
    log(f"    K={k:<3} ARI {np.mean(pairs):.4f}±{np.std(pairs, ddof=1):.4f}")
ari_k = pd.DataFrame(ari_k_rows)
ari_k.to_csv(OUT / "cluster_stability_ari_by_k.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------- 3. 情感筛选口径敏感性

section("实验 3  情感筛选口径敏感性（4.7 节第 1 项）")

SAMPLE_SETS = {
    "A_MLLM判负向（本文口径）": mask_llm,
    "B_BERT判负向（原稿所述口径）": mask_bert,
    "C_两套判定交集": mask_llm & mask_bert,
}

sens_rows = []
topic_words_by_set = {}
station_share_by_set = {}

for set_name, mask in SAMPLE_SETS.items():
    w, d_desc, _ = pc.build_sample(raw, mask)
    emb = pc.encode(qwen, d_desc)
    scorer = pc.NpmiScorer(d_desc)
    log(f"\n  {set_name}：{len(w)} 条样本")

    per_seed = []
    for seed in pc.SEEDS:
        labels = KMeans(n_clusters=pc.K_TOPICS, random_state=seed, n_init=10).fit_predict(emb)
        per_seed.append(pc.evaluate_partition(set_name, d_desc, emb, labels, scorer))
    df_seed = pd.DataFrame(per_seed)

    ref_labels = KMeans(n_clusters=pc.K_TOPICS, random_state=42, n_init=10).fit_predict(emb)
    topic_words_by_set[set_name] = pc.extract_topic_words(d_desc, ref_labels)
    station_share_by_set[set_name] = (
        w["站区"].value_counts(normalize=True).mul(100).round(1)
    )

    row = {
        "样本集": set_name,
        "样本量": len(w),
        "轮廓系数": f"{df_seed['silhouette_score'].mean():.3f}±{df_seed['silhouette_score'].std():.3f}",
        "CHI": f"{df_seed['calinski_harabasz_index'].mean():.3f}±{df_seed['calinski_harabasz_index'].std():.3f}",
        "NPMI": f"{df_seed['topic_coherence_npmi'].mean():.3f}±{df_seed['topic_coherence_npmi'].std():.3f}",
        "主题多样性": f"{df_seed['topic_diversity'].mean():.3f}±{df_seed['topic_diversity'].std():.3f}",
        "NPMI均值": round(float(df_seed["topic_coherence_npmi"].mean()), 4),
    }
    sens_rows.append(row)
    log(f"    NPMI {row['NPMI']}  轮廓 {row['轮廓系数']}  多样性 {row['主题多样性']}")

sens = pd.DataFrame(sens_rows)
sens.to_csv(OUT / "sentiment_criterion_sensitivity.csv", index=False, encoding="utf-8-sig")

# 主题结构比对：以基准集各主题的 Top-10 词表为准，在对照集中找最佳匹配主题
def best_match_jaccard(base_words: dict, other_words: dict) -> list[dict]:
    out = []
    for bid, bw in sorted(base_words.items()):
        bset = set(bw)
        best_id, best_j = None, 0.0
        for oid, ow in other_words.items():
            oset = set(ow)
            j = len(bset & oset) / len(bset | oset) if bset | oset else 0.0
            if j > best_j:
                best_id, best_j = oid, j
        out.append(
            {
                "基准主题": bid,
                "主题名称": pc.TOPIC_TITLES.get(bid, ""),
                "最佳匹配主题": best_id,
                "Jaccard": round(best_j, 3),
                "共同主题词": " / ".join(
                    sorted(set(bw) & set(other_words.get(best_id, [])))
                ),
            }
        )
    return out


base_name = "A_MLLM判负向（本文口径）"
match_frames = []
log()
log("  主题结构比对（以本文口径的 10 个主题为基准，在对照集中找 Top-10 词表最相似的主题）：")
for set_name in ("B_BERT判负向（原稿所述口径）", "C_两套判定交集"):
    matches = best_match_jaccard(topic_words_by_set[base_name], topic_words_by_set[set_name])
    frame = pd.DataFrame(matches)
    frame.insert(0, "对照集", set_name)
    match_frames.append(frame)
    recovered = int((frame["Jaccard"] >= 0.3).sum())
    strong = int((frame["Jaccard"] >= 0.5).sum())
    log(f"\n    {set_name}")
    log(f"      Jaccard ≥ 0.5 的主题：{strong}/10；≥ 0.3 的主题：{recovered}/10；均值 {frame['Jaccard'].mean():.3f}")
    for _, r in frame.iterrows():
        log(f"        主题{r['基准主题']} {r['主题名称'][:14]:<16} Jaccard={r['Jaccard']:.3f}")

pd.concat(match_frames).to_csv(
    OUT / "sentiment_criterion_topic_match.csv", index=False, encoding="utf-8-sig"
)

st_share = pd.DataFrame(station_share_by_set).fillna(0)
st_share.to_csv(OUT / "sentiment_criterion_station_share.csv", encoding="utf-8-sig")
log()
log("  三套样本的站区构成（%）：")
log("    " + st_share.to_string().replace("\n", "\n    "))


# ------------------------------------------------------------------ 4. 绘图

section("实验 4  图 3、图 6、图 7")

log("绘图已拆分到 scripts/plot_paper_figures.py（从本脚本产出的结果表读数）。")
log("本脚本执行完毕后运行：python scripts/plot_paper_figures.py")


# --------------------------------------- 5. 4.3 节人工核验的抽样与标注模板

section("附加  生成 4.3 节人工核验的抽样文件与标注模板")

# 抽样一：综合描述事实一致性（分层抽样，按站区比例，120 条）
n_desc = 120
by_station = work.groupby("站区")
quota = (by_station.size() / len(work) * n_desc).round().astype(int)
picks = []
for station, grp in by_station:
    take = min(int(quota[station]), len(grp))
    picks.append(grp.sample(n=take, random_state=42))
sample_desc = pd.concat(picks).sample(frac=1, random_state=7).reset_index(drop=True)
sample_desc.insert(0, "核验编号", [f"D{i + 1:03d}" for i in range(len(sample_desc))])
for col in ("站区识别正确", "服务环节识别正确", "无新增事实", "错误类型", "备注"):
    sample_desc[col] = ""
cols = ["核验编号", "id", "站区", "title", "source_content", "content",
        "站区识别正确", "服务环节识别正确", "无新增事实", "错误类型", "备注"]
sample_desc[[c for c in cols if c in sample_desc.columns]].to_csv(
    OUT / "verify_sample_1_description.csv", index=False, encoding="utf-8-sig"
)
log(f"  抽样一（综合描述事实一致性）：{len(sample_desc)} 条，按站区分层")

# 抽样二：情感判定准确性（全量分层，三类各按比例，300 条）
n_sent = 300
raw_pool = raw.copy()
raw_pool["_engagement"] = sum(raw_pool[c].map(pc.parse_count) for c in pc.ENGAGEMENT_COLUMNS)
sent_picks = []
for group, grp in raw_pool.groupby("llm_sentiment_group"):
    take = min(int(round(len(grp) / len(raw_pool) * n_sent)), len(grp))
    sent_picks.append(grp.sample(n=take, random_state=42))
sample_sent = pd.concat(sent_picks).sample(frac=1, random_state=11).reset_index(drop=True)
sample_sent.insert(0, "核验编号", [f"S{i + 1:03d}" for i in range(len(sample_sent))])
sample_sent["人工情感判定"] = ""
sample_sent["人工问题指向判定"] = ""
sample_sent["备注"] = ""
sent_cols = ["核验编号", "note_id", "station", "title", "content", "description",
             "人工情感判定", "人工问题指向判定", "备注"]
sample_sent[[c for c in sent_cols if c in sample_sent.columns]].to_csv(
    OUT / "verify_sample_2_sentiment.csv", index=False, encoding="utf-8-sig"
)
log(f"  抽样二（情感判定准确性）：{len(sample_sent)} 条，按三类情感分层")
log(f"    其中含 BERT 与 MLLM 判定分歧样本 "
    f"{int((sample_sent['llm_sentiment_group'].eq('negative') != sample_sent['sentiment_label'].eq('Negative')).sum())} 条")

# 抽样三：中立样本的问题指向漏检（200 条）
neutral_pool = raw[raw["llm_sentiment_group"].eq("neutral")]
sample_neutral = neutral_pool.sample(n=min(200, len(neutral_pool)), random_state=42).reset_index(drop=True)
sample_neutral.insert(0, "核验编号", [f"N{i + 1:03d}" for i in range(len(sample_neutral))])
sample_neutral["含明确问题指向"] = ""
sample_neutral["问题所属服务环节"] = ""
sample_neutral["备注"] = ""
neu_cols = ["核验编号", "note_id", "station", "title", "content", "description",
            "含明确问题指向", "问题所属服务环节", "备注"]
sample_neutral[[c for c in neu_cols if c in sample_neutral.columns]].to_csv(
    OUT / "verify_sample_3_neutral.csv", index=False, encoding="utf-8-sig"
)
log(f"  抽样三（中立样本漏检）：{len(sample_neutral)} 条")

# 抽样四：主题命名恰当性评分表
rep = pd.read_csv(OUT / "topic_representative_posts.csv")
name_rows = []
kw = pd.read_csv(OUT / "kmeans_topic_summary.csv")[["topic", "主题标题", "主题词"]]
for _, r in kw.iterrows():
    t = int(r["topic"])
    reps = rep[rep["topic"] == t]["摘要"].tolist()
    name_rows.append(
        {
            "主题编号": t,
            "拟定名称": r["主题标题"],
            "Top-5 主题词": r["主题词"],
            "代表帖1": reps[0] if len(reps) > 0 else "",
            "代表帖2": reps[1] if len(reps) > 1 else "",
            "代表帖3": reps[2] if len(reps) > 2 else "",
            "评分者A(1-5)": "", "评分者B(1-5)": "", "评分者C(1-5)": "",
            "建议改名": "",
        }
    )
pd.DataFrame(name_rows).to_csv(
    OUT / "verify_sample_4_topic_naming.csv", index=False, encoding="utf-8-sig"
)
log("  抽样四（主题命名评分表）：10 个主题")

section("完成")
log(f"结果目录：{OUT}")
log(f"图片目录：{FIG}")
for f in sorted(FIG.iterdir()):
    log(f"    figures/{f.name:<36}{f.stat().st_size:>9,} B")

(OUT / "rerun_0905_supplement.log").write_text("\n".join(LOG), encoding="utf-8")
