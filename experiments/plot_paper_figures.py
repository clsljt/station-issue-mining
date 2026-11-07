"""绘制论文图 3、图 6、图 7。

从 output/0905 下的结果表读数，与实验脚本解耦，便于单独调整版式而不必重跑实验。
运行前需先执行 rerun_0905.py 与 rerun_0905_supplement.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import paper_common as pc

OUT = pc.ROOT / "data/results/output/0905"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GRAY = "#2F6DB5", "#E08A2E", "#8A8A8A"

font = pc.setup_cjk_font()
print(f"中文字体：{font}")

import matplotlib.pyplot as plt


def save(fig, stem: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存 {stem}.svg / .png")


# ------------------------------------------------------------------ 图 3

agg = pd.read_csv(OUT / "k_selection_summary.csv").sort_values("K")
k_range = agg["K"].astype(int).tolist()

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
panels = [
    ("silhouette_score", "轮廓系数（余弦）", "越大越好", "随 $K$ 单调上升"),
    ("davies_bouldin_index", "Davies-Bouldin 指数", "越小越好", "随 $K$ 单调下降"),
    ("calinski_harabasz_index", "Calinski-Harabasz 指数", "越大越好", "随 $K$ 单调下降"),
    ("topic_coherence_npmi", "主题一致性 NPMI", "越大越好", "无显著趋势"),
    ("topic_diversity", "主题多样性", "越大越好", "随 $K$ 下降"),
]
for ax, (col, title, hint, trend) in zip(axes.flat, panels):
    mean, std = agg[f"{col}_mean"].to_numpy(), agg[f"{col}_std"].to_numpy()
    ax.plot(agg["K"], mean, "-o", color=BLUE, markersize=4, linewidth=1.6)
    ax.fill_between(agg["K"], mean - std, mean + std, color=BLUE, alpha=0.15)
    ax.axvline(10, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.set_title(f"{title}（{hint}）", fontsize=10.5)
    ax.set_xlabel("主题数 $K$", fontsize=9.5)
    ax.set_xticks(k_range)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.annotate(trend, xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5, color=GRAY)

# 第六格改为碎片主题数与最小主题占比：这两项才是本文最终确定 K 的依据，
# 前五项指标或随 K 单调变化、或无显著趋势，都无法通过取极值定 K。
ax = axes.flat[5]
ax.plot(agg["K"], agg["碎片主题数"], "-o", color=BLUE, markersize=4, linewidth=1.6,
        label="碎片主题数（样本量 < 10）")
ax.axvline(10, color=ORANGE, linestyle="--", linewidth=1.2)
ax.axhline(1, color=GRAY, linestyle=":", linewidth=1.0)
ax.set_ylabel("碎片主题数（个）", fontsize=9.5, color=BLUE)
ax.tick_params(axis="y", labelcolor=BLUE, labelsize=8.5)
ax.set_xlabel("主题数 $K$", fontsize=9.5)
ax.set_xticks(k_range)
ax.tick_params(axis="x", labelsize=8.5)
ax.grid(alpha=0.25, linewidth=0.6)

ax_r = ax.twinx()
ax_r.plot(agg["K"], agg["最小主题占比"], "--s", color="#B5502F", markersize=3.5,
          linewidth=1.3, label="最小主题占比")
ax_r.set_ylabel("最小主题占比（%）", fontsize=9.5, color="#B5502F")
ax_r.tick_params(axis="y", labelcolor="#B5502F", labelsize=8.5)
ax.set_title("主题碎片化程度（定 $K$ 的实际依据）", fontsize=10.5)
handles = ax.get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + ax_r.get_legend_handles_labels()[1]
ax.legend(handles, labels, fontsize=8, frameon=False, loc="upper left")

fig.suptitle(
    "不同主题数下的评价指标变化（10 个随机种子，阴影为 ±1 标准差，橙色虚线为本文取值 $K=10$）",
    fontsize=11.5, y=1.0,
)
fig.tight_layout()
save(fig, "fig3_k_selection")


# ------------------------------------------------------------- 图 6 与图 7

summary = pd.read_csv(OUT / "kmeans_topic_summary.csv").sort_values("topic")
rho = summary["文本占比"].to_numpy() * 100
eta = summary["互动占比"].to_numpy() * 100
n_text = summary["文本数"].to_numpy()
titles = summary["主题标题"].tolist()
tids = summary["topic"].astype(int).tolist()

fig, ax = plt.subplots(figsize=(8.6, 7.2))
lim = max(rho.max(), eta.max()) * 1.22
ax.plot([0, lim], [0, lim], color=GRAY, linestyle="--", linewidth=1.2, zorder=1)
ax.text(lim * 0.055, lim * 0.10, "$\\eta_k=\\rho_k$", color=GRAY, fontsize=10.5, rotation=39)
ax.fill_between([0, lim], [0, lim], [lim, lim], color=ORANGE, alpha=0.05, zorder=0)
ax.fill_between([0, lim], [0, 0], [0, lim], color=BLUE, alpha=0.05, zorder=0)
ax.text(lim * 0.30, lim * 0.93, "传播强度高于表达规模", color="#B5502F", fontsize=10)
ax.text(lim * 0.62, lim * 0.06, "表达规模高于传播强度", color="#2F6DB5", fontsize=10)

colors = [ORANGE if e > r else BLUE for r, e in zip(rho, eta)]
ax.scatter(rho, eta, s=n_text * 11, c=colors, alpha=0.62, edgecolors="white",
           linewidths=1.4, zorder=3)

# 标签偏移逐点微调，避免与气泡、对角线及图边界重叠
OFFSETS = {0: (-116, -4), 1: (16, -4), 2: (11, 11), 3: (-186, -4), 4: (14, -3),
           5: (-108, -4), 6: (13, -4), 7: (11, 3), 8: (12, 2), 9: (10, 2)}
for t, r, e, title in zip(tids, rho, eta, titles):
    ax.annotate(f"{t} {title}", (r, e), textcoords="offset points",
                xytext=OFFSETS.get(t, (10, 4)), fontsize=8.8, color="#333333")

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("文本占比 $\\rho_k$（%）", fontsize=11)
ax.set_ylabel("互动占比 $\\eta_k$（%）", fontsize=11)
ax.set_title("主题文本占比与互动占比的联合分布（气泡面积表示样本数）", fontsize=11.5)
ax.grid(alpha=0.25, linewidth=0.6)
fig.tight_layout()
save(fig, "fig6_share_joint")

order = np.argsort(-(eta - rho))
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7.6), gridspec_kw={"height_ratios": [2.1, 1]}, sharex=True
)
x = np.arange(len(order))
width = 0.38
ax1.bar(x - width / 2, rho[order], width, label="文本占比 $\\rho_k$", color=BLUE, alpha=0.88)
ax1.bar(x + width / 2, eta[order], width, label="互动占比 $\\eta_k$", color=ORANGE, alpha=0.88)
for i, idx in enumerate(order):
    ax1.text(i - width / 2, rho[idx] + 0.35, f"{rho[idx]:.1f}", ha="center", fontsize=8)
    ax1.text(i + width / 2, eta[idx] + 0.35, f"{eta[idx]:.1f}", ha="center", fontsize=8)
ax1.set_ylabel("占比（%）", fontsize=11)
ax1.set_title("各主题的文本占比与互动占比（按差值降序排列）", fontsize=11.5)
ax1.legend(fontsize=10, frameon=False)
ax1.grid(alpha=0.25, axis="y", linewidth=0.6)
ax1.set_ylim(0, max(rho.max(), eta.max()) * 1.16)

diff = (eta - rho)[order]
ax2.bar(x, diff, 0.55, color=[ORANGE if d > 0 else BLUE for d in diff], alpha=0.88)
ax2.axhline(0, color="#444444", linewidth=0.9)
for i, d in enumerate(diff):
    ax2.text(i, d + (0.55 if d > 0 else -1.25), f"{d:+.2f}", ha="center", fontsize=8.5)
ax2.set_ylabel("$\\eta_k-\\rho_k$（百分点）", fontsize=10.5)
ax2.set_ylim(diff.min() * 1.45, diff.max() * 1.45)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{tids[i]} {titles[i]}" for i in order], rotation=32, ha="right", fontsize=9)
ax2.grid(alpha=0.25, axis="y", linewidth=0.6)
fig.tight_layout()
save(fig, "fig7_share_compare")

print(f"图片目录：{FIG}")
