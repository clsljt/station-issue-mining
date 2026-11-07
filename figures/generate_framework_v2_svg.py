"""Draw the widened, editable SVG framework with the added problem-identification step."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUT_SVG = ROOT / "chapter3_framework_v2.svg"
OUT_PNG = ROOT / "chapter3_framework_v2.png"

COLORS = {
    "ink": "#243746", "muted": "#6d7d88", "line": "#9baab5", "outer": "#aebdc7",
    "blue": "#6b91aa", "blue_fill": "#e7eff4", "green": "#739987", "green_fill": "#e9f1ec",
    "orange": "#bd965e", "orange_fill": "#fbf3e4", "white": "#ffffff",
}


def setup_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Hiragino Sans GB", "PingFang SC", "STHeiti", "Arial Unicode MS"):
        if name in available:
            mpl.rcParams["font.family"] = [name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            mpl.rcParams["svg.fonttype"] = "none"
            return name
    raise RuntimeError("No CJK font available")


def panel(ax, x, y, w, h, number, title, *, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=COLORS["white"], edgecolor=edge, linewidth=1.1, zorder=2))
    ax.add_patch(Rectangle((x, y+h-0.42), w, 0.42, facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=3))
    ax.text(x+0.18, y+h-0.21, str(number), ha="center", va="center", fontsize=12,
            color=COLORS["white"], weight="bold", bbox=dict(boxstyle="circle,pad=0.16", fc=edge, ec="none"), zorder=4)
    ax.text(x+0.48, y+h-0.21, title, ha="left", va="center", fontsize=12.2,
            color=COLORS["ink"], weight="bold", zorder=4)


def box(ax, x, y, w, h, text, *, edge, fill, size=10.2, color=None, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05",
                                facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=4))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size,
            color=color or COLORS["ink"], weight=weight, zorder=5)


def arrow(ax, a, b, *, color=COLORS["line"], style="-|>", lw=1.15, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, linewidth=lw,
                                 linestyle=ls, color=color, shrinkA=2, shrinkB=2, zorder=1))


def draw():
    setup_font()
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0.02, 0.035, 0.96, 0.93])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 16, 9, facecolor="white", edgecolor="none", zorder=0))
    ax.text(8, 8.72, "多模态大模型驱动的交通枢纽社交媒体舆情分析框架",
            ha="center", va="center", fontsize=19, weight="bold", color=COLORS["ink"])
    ax.add_patch(FancyBboxPatch((0.35, 0.25), 15.3, 8.12, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="none", edgecolor=COLORS["outer"], linewidth=1.0, linestyle=(0,(4,3)), zorder=1))
    ax.text(0.72, 8.13, "主分析流程", fontsize=12.5, weight="bold", color=COLORS["muted"])

    # Top row
    panel(ax, .75, 5.82, 6.7, 1.95, 1, "多模态语义提取", edge=COLORS["blue"], fill=COLORS["blue_fill"])
    ax.text(1.15, 7.20, "图文与评论", fontsize=12, color=COLORS["ink"], weight="bold")
    ax.text(1.15, 6.78, "标题 · 正文 · 标签\n配图 · 高互动评论", fontsize=10.5, color=COLORS["muted"], linespacing=1.45)
    box(ax, 3.55, 6.74, 2.95, .52, "多模态大模型", edge="#9bb4c4", fill=COLORS["blue_fill"], size=11)
    box(ax, 3.55, 6.05, 2.95, .47, "语义描述 · 情感标签", edge="#9bb4c4", fill=COLORS["blue_fill"], size=10.2)
    arrow(ax, (2.85, 6.99), (3.52, 6.99)); arrow(ax, (5.02, 6.72), (5.02, 6.54), color=COLORS["blue"])
    ax.text(5.24, 6.64, "同步推理", fontsize=9, color=COLORS["muted"])
    panel(ax, 8.55, 5.82, 6.7, 1.95, 2, "情感分析", edge=COLORS["blue"], fill=COLORS["blue_fill"])
    ax.text(11.9, 7.18, "情感标签", ha="center", fontsize=11.5, color=COLORS["ink"], weight="bold")
    box(ax, 9.15, 6.55, 1.55, .46, "负向", edge="#c9aa77", fill=COLORS["orange_fill"], size=10.5)
    box(ax, 11.02, 6.55, 1.55, .46, "中立", edge="#bdc9d0", fill="#f5f7f8", size=10.5)
    box(ax, 12.89, 6.55, 1.55, .46, "正向", edge="#bdc9d0", fill="#f5f7f8", size=10.5)
    ax.text(10.0, 6.30, "保留负向", fontsize=10, color=COLORS["orange"], weight="bold")
    box(ax, 10.65, 5.98, 3.05, .45, "问题导向候选样本", edge="#9bb4c4", fill=COLORS["blue_fill"], size=10.3)
    arrow(ax, (9.92, 6.53), (9.92, 6.33), color=COLORS["orange"]); arrow(ax, (9.92, 6.28), (10.63, 6.28), color=COLORS["orange"])

    # Validation
    box(ax, 6.98, 5.05, 2.05, .48, "人工核验", edge="#c9aa77", fill=COLORS["orange_fill"], size=11, color="#8e6c38", weight="bold")
    arrow(ax, (5.75, 5.82), (7.12, 5.54), color=COLORS["orange"], ls="--", style="-" )
    arrow(ax, (9.75, 5.82), (8.9, 5.54), color=COLORS["orange"], ls="--", style="-")

    # Middle row: topic, problem classification, attention
    panel(ax, .75, 2.23, 4.55, 2.35, 3, "主题识别", edge=COLORS["green"], fill=COLORS["green_fill"])
    box(ax, 1.05, 3.85, 3.95, .45, "语义嵌入", edge="#a8c4b3", fill=COLORS["green_fill"], size=10.8)
    ax.text(1.55, 3.37, "● ●   ● ●", fontsize=17, color=COLORS["green"], ha="center")
    ax.text(3.90, 3.37, "K-means 聚类", fontsize=10.5, color=COLORS["ink"], ha="center")
    box(ax, 1.30, 2.64, 3.45, .45, "c-TF-IDF → 主题词", edge="#a8c4b3", fill=COLORS["green_fill"], size=10.0)
    box(ax, 1.30, 2.34, 3.45, .45, "主题命名与解释", edge="#a8c4b3", fill=COLORS["green_fill"], size=10.0)
    arrow(ax, (2.98, 3.84), (2.98, 3.62), color=COLORS["green"]); arrow(ax, (2.98, 3.08), (2.98, 2.98), color=COLORS["green"])
    panel(ax, 5.73, 2.23, 4.55, 2.35, 4, "问题识别", edge=COLORS["green"], fill=COLORS["green_fill"])
    # The problem-identification input is already preprocessed, so keep this
    # stage intentionally abstract and avoid exposing prompt/category details.
    box(ax, 6.32, 3.74, 3.35, .44, "已处理文本", edge="#b6c6bd", fill="#f7faf8", size=10.5)
    box(ax, 7.18, 3.12, 1.63, .44, "问题识别", edge="#9bb4c4", fill=COLORS["blue_fill"], size=10.2)
    ax.text(8.00, 2.93, "五类问题标签", ha="center", fontsize=9.8, color=COLORS["green"], weight="bold")
    for i in range(5):
        x = 6.18 + (i % 3)*1.35; y = 2.58 - (i//3)*.34
        box(ax, x, y, 1.15, .27, "", edge="#b6c6bd", fill="#f7faf8", size=8.2)
    arrow(ax, (8.00, 3.72), (8.00, 3.59), color=COLORS["green"])
    arrow(ax, (8.00, 3.10), (8.00, 2.98), color=COLORS["green"])
    panel(ax, 10.72, 2.23, 4.53, 2.35, 5, "关注度分析", edge=COLORS["green"], fill=COLORS["green_fill"])
    box(ax, 11.12, 3.63, 1.55, .34, "主题层面", edge="#a8c4b3", fill="#f7faf8", size=9.5, color=COLORS["green"], weight="bold")
    box(ax, 11.12, 2.82, 1.55, .60, "文本占比 · 互动占比\n站区分布", edge="#b6c6bd", fill="#f7faf8", size=8.8)
    box(ax, 13.30, 3.63, 1.55, .34, "帖子层面", edge="#a8c4b3", fill="#f7faf8", size=9.5, color=COLORS["green"], weight="bold")
    box(ax, 13.30, 2.82, 1.55, .60, "语义相似网络\n个性化 PageRank\n代表帖", edge="#b6c6bd", fill="#f7faf8", size=8.8)
    arrow(ax, (10.30, 3.40), (10.68, 3.40));

    # Bottom output
    panel(ax, 5.65, .65, 4.70, 1.05, 6, "治理建议", edge=COLORS["orange"], fill=COLORS["orange_fill"])
    ax.text(8.0, 1.06, "问题分类 · 关注优先级 · 站区画像 → 服务优化与治理核查", ha="center", va="center", fontsize=10.5, color=COLORS["ink"])
    arrow(ax, (8.00, 2.20), (8.00, 1.74), color=COLORS["line"])
    arrow(ax, (2.95, 2.20), (7.55, 1.74), color=COLORS["line"], ls="--", style="-")
    arrow(ax, (12.98, 2.20), (8.45, 1.74), color=COLORS["line"], ls="--", style="-")
    arrow(ax, (11.90, 5.80), (11.90, 4.62), color=COLORS["line"])
    arrow(ax, (3.00, 5.80), (8.00, 4.62), color=COLORS["line"], ls="--", style="-")
    arrow(ax, (8.00, 4.62), (8.00, 4.62), color=COLORS["line"])

    fig.savefig(OUT_SVG, format="svg", bbox_inches="tight", pad_inches=.08)
    fig.savefig(OUT_PNG, format="png", dpi=220, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    draw()
