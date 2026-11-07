"""Generate Figure 1: multimodal public-opinion analysis framework."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent


PALETTE = {
    "navy": "#183A5A",
    "blue": "#2F7EAF",
    "cyan": "#DCEFF7",
    "teal": "#2C8C85",
    "mint": "#DFF2EC",
    "orange": "#DD8A35",
    "cream": "#F9E8D3",
    "purple": "#8064A2",
    "lavender": "#ECE6F4",
    "ink": "#203040",
    "muted": "#627180",
    "line": "#9BA8B2",
    "paper": "#FBFCFD",
}


def set_cjk_font() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in ("Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "Songti SC"):
        if candidate in available:
            mpl.rcParams["font.family"] = [candidate, "Arial", "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            mpl.rcParams["pdf.fonttype"] = 42
            mpl.rcParams["ps.fonttype"] = 42
            mpl.rcParams["svg.fonttype"] = "none"
            return candidate
    raise RuntimeError("No supported CJK font is available.")


def rounded_box(ax, x, y, w, h, title, body, *, edge, fill, title_color=None,
                icon=None, body_size=7.1, title_size=8.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=1.15, edgecolor=edge, facecolor=fill, zorder=2,
    ))
    if icon:
        ax.text(x + 0.018, y + h - 0.043, icon, fontsize=10, color=edge,
                va="center", ha="left", fontweight="bold", zorder=3)
        title_x = x + 0.053
    else:
        title_x = x + 0.018
    ax.text(title_x, y + h - 0.043, title, fontsize=title_size, color=title_color or edge,
            va="center", ha="left", fontweight="bold", zorder=3)
    ax.text(x + 0.018, y + h - 0.085, body, fontsize=body_size, color=PALETTE["ink"],
            va="top", ha="left", linespacing=1.42, zorder=3)


def arrow(ax, start, end, *, color=PALETTE["line"], lw=1.35, rad=0.0):
    style = "arc3" if rad == 0 else f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch(
        start, end, connectionstyle=style, arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2, zorder=1,
    ))


def section_label(ax, x, text, color):
    ax.text(x, 0.925, text, fontsize=8.8, color=color, va="center", ha="left",
            fontweight="bold")
    ax.plot([x, x + 0.165], [0.907, 0.907], color=color, lw=2.2, solid_capstyle="round")


def draw_figure():
    set_cjk_font()
    fig = plt.figure(figsize=(13.3, 6.7), facecolor="white")
    ax = fig.add_axes([0.025, 0.055, 0.95, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background and title
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=PALETTE["paper"], edgecolor="none", zorder=0))
    ax.text(0.5, 0.995, "基于多模态公众感知的站区服务问题识别与影响分析框架",
            fontsize=14.2, fontweight="bold", color=PALETTE["navy"], ha="center", va="top")
    ax.text(0.5, 0.963, "从社交媒体图文表达，到可追溯、可比较、可治理的服务问题证据",
            fontsize=8.5, color=PALETTE["muted"], ha="center", va="top")

    section_label(ax, 0.035, "① 多模态公众感知输入", PALETTE["blue"])
    section_label(ax, 0.270, "② 语义清洗与问题样本构建", PALETTE["teal"])
    section_label(ax, 0.555, "③ 短文本主题发现", PALETTE["orange"])
    section_label(ax, 0.790, "④ 影响解释与治理转化", PALETTE["purple"])

    # Input column
    rounded_box(ax, 0.035, 0.663, 0.185, 0.195, "图文内容", "帖子标题与正文\n话题标签 · 图片/视频\n空间、设施、排队等场景线索",
                edge=PALETTE["blue"], fill=PALETTE["cyan"], icon="▣")
    rounded_box(ax, 0.035, 0.422, 0.185, 0.195, "互动与补充反馈", "高互动评论\n点赞 · 评论 · 收藏 · 分享\n发布时间、站场等元数据",
                edge=PALETTE["blue"], fill=PALETTE["cyan"], icon="◉")
    rounded_box(ax, 0.035, 0.181, 0.185, 0.195, "可追溯数据底座", "以 note_id 为主键\n统一组织原始信息与模型输出\n支持回溯、统计与比较",
                edge=PALETTE["navy"], fill="#E9F0F5", icon="⌘")
    arrow(ax, (0.127, 0.663), (0.127, 0.617), color=PALETTE["blue"])
    arrow(ax, (0.127, 0.422), (0.127, 0.376), color=PALETTE["blue"])

    # Stage 2
    rounded_box(ax, 0.270, 0.662, 0.225, 0.200, "Qwen3.6-27B 多模态语义清洗",
                "图文、标签、评论与互动信息联合解析\n业务相关性分类 + 综合描述生成\n固定标签集合与 JSON 结构化输出",
                edge=PALETTE["teal"], fill=PALETTE["mint"], icon="✦", body_size=6.85)
    rounded_box(ax, 0.270, 0.410, 0.225, 0.185, "BERT 情感识别", "在业务类样本中识别情感倾向\n过滤攻略、正向体验与中性说明\n保留负向问题导向候选帖子",
                edge=PALETTE["teal"], fill=PALETTE["mint"], icon="≈")
    rounded_box(ax, 0.270, 0.164, 0.225, 0.184, "问题导向语料库", "标题 + 正文 + 图片描述 + 高价值评论\n保留站场、业务标签与互动指标\n293 条候选问题帖子",
                edge=PALETTE["navy"], fill="#E9F0F5", icon="✓")
    arrow(ax, (0.220, 0.520), (0.270, 0.756), color=PALETTE["blue"])
    arrow(ax, (0.383, 0.662), (0.383, 0.595), color=PALETTE["teal"])
    arrow(ax, (0.383, 0.410), (0.383, 0.348), color=PALETTE["teal"])
    ax.text(0.507, 0.612, "业务类样本", fontsize=6.6, color=PALETTE["teal"], ha="left", va="center")

    # Stage 3
    rounded_box(ax, 0.555, 0.660, 0.200, 0.202, "语义表征", "Qwen3-Embedding-0.6B\n将口语、简称与隐含诉求\n映射至连续语义向量空间",
                edge=PALETTE["orange"], fill=PALETTE["cream"], icon="↗")
    rounded_box(ax, 0.555, 0.410, 0.200, 0.185, "主题发现与解释", "KMeans 语义聚类\nc-TF-IDF 提取主题词\n代表文本 + 业务构成辅助命名",
                edge=PALETTE["orange"], fill=PALETTE["cream"], icon="◎")
    rounded_box(ax, 0.555, 0.164, 0.200, 0.184, "服务问题主题", "候车与进站 · 出站换乘\n运行延误 · 航站楼服务\n网约车接驳等 10 个主题",
                edge=PALETTE["navy"], fill="#F4EEF9", icon="◆")
    arrow(ax, (0.495, 0.256), (0.555, 0.756), color=PALETTE["teal"])
    arrow(ax, (0.655, 0.660), (0.655, 0.595), color=PALETTE["orange"])
    arrow(ax, (0.655, 0.410), (0.655, 0.348), color=PALETTE["orange"])

    # Stage 4
    rounded_box(ax, 0.790, 0.660, 0.175, 0.202, "综合影响力", "语义相似网络\nPageRank 中心性\n互动量个性化分布",
                edge=PALETTE["purple"], fill=PALETTE["lavender"], icon="◌")
    rounded_box(ax, 0.790, 0.410, 0.175, 0.185, "治理任务映射", "智慧接驳与衔接\n导航导视与指引\n环境、站区服务与城市治理",
                edge=PALETTE["purple"], fill=PALETTE["lavender"], icon="▤", body_size=6.7)
    rounded_box(ax, 0.790, 0.164, 0.175, 0.184, "治理输出", "主题优先级排序\n站场问题画像与横向比较\n服务优化与核查方向",
                edge=PALETTE["navy"], fill="#E9F0F5", icon="★")
    arrow(ax, (0.755, 0.256), (0.790, 0.756), color=PALETTE["orange"])
    arrow(ax, (0.877, 0.660), (0.877, 0.595), color=PALETTE["purple"])
    arrow(ax, (0.877, 0.410), (0.877, 0.348), color=PALETTE["purple"])

    # Cross-cutting traceability line
    ax.plot([0.127, 0.877], [0.085, 0.085], color=PALETTE["navy"], lw=1.0, alpha=0.55)
    for x in (0.127, 0.383, 0.655, 0.877):
        ax.plot([x, x], [0.085, 0.124], color=PALETTE["navy"], lw=1.0, alpha=0.55)
    ax.text(0.5, 0.052, "贯穿全流程的可追溯机制：note_id 关联原始帖子、模型字段、主题标签与影响力结果",
            fontsize=7.3, color=PALETTE["muted"], ha="center", va="center")

    # Scope note
    ax.text(0.5, 0.004, "注：综合影响力反映社交媒体表达与传播表现，不直接等同于问题发生频率或站场服务质量。",
            fontsize=6.5, color=PALETTE["muted"], ha="center", va="bottom")
    return fig


if __name__ == "__main__":
    fig = draw_figure()
    preview = ROOT / "chapter3_framework_preview.png"
    fig.savefig(preview, dpi=180, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(ROOT / "chapter3_framework.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(ROOT / "chapter3_framework.svg", bbox_inches="tight", pad_inches=0.05)
    png = ROOT / "chapter3_framework.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    try:
        from PIL import Image
        Image.open(png).convert("L").save(ROOT / "chapter3_framework_grayscale.png")
    except ImportError:
        pass
    plt.close(fig)
