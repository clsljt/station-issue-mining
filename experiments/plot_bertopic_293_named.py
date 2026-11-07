"""Name the 17 fitted topics and redraw their existing weights in five columns.

This script reads fitted outputs only; topic assignments and weights do not change.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/station_mpl")

import pandas as pd
from paper_common import setup_cjk_font

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results/output/1020/bertopic_293_more_topics"
TITLES = {
    0: "机场安检与行李服务",
    1: "丰台站出站动线设计",
    2: "站内与跨站换乘耗时",
    3: "航班延误与异地备降",
    4: "票证异常与进站受阻",
    5: "夜间运力不足与打车排队",
    6: "出租车与网约车接驳服务",
    7: "上车点定位与路线指引",
    8: "站台吸烟与控烟管理",
    9: "机场线票价与出行成本",
    10: "候车座位与环境舒适度",
    11: "站场餐饮价格与品质",
    12: "高峰客流与候车拥挤",
    13: "列车晚点与旅客滞留",
    14: "车站命名与周边争议",
    15: "车站综合体验与出行困扰",
    16: "站内导视与寻路困难",
}


def main():
    inputs = [OUT / f for f in ("topic_info.csv", "topic_keywords.csv", "topic_assignments.csv")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    info, words, assignments = (pd.read_csv(p, dtype={"id": str}) for p in inputs)
    ids = sorted(int(t) for t in info.Topic if t >= 0)
    assert ids == sorted(TITLES)
    assert len(assignments) == 293 and info.Count.sum() == 293
    count = dict(zip(info.Topic, info.Count))
    names = {-1: "离群点", **TITLES}
    info.insert(2, "主题名称", info.Topic.map(names))
    assignments["主题名称"] = assignments.topic.map(names)
    words["主题名称"] = words.topic.map(names)
    info.to_csv(OUT / "topic_info_named.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(OUT / "topic_assignments_named.csv", index=False, encoding="utf-8-sig")
    words.to_csv(OUT / "topic_keywords_named.csv", index=False, encoding="utf-8-sig")
    (OUT / "topic_titles.json").write_text(json.dumps(TITLES, ensure_ascii=False, indent=2), encoding="utf-8")

    setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cols, nrows = 5, math.ceil(len(ids) / 5)
    fig, axes = plt.subplots(nrows, cols, figsize=(24, 13.9), squeeze=False)
    colors = ["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73"]
    for ax, t in zip(axes.flat, ids):
        top = words.loc[words.topic.eq(t)].sort_values("rank").head(8)
        ax.set_facecolor("#E8EDF5")
        ax.barh(range(len(top)), top.weight, color=colors[t % len(colors)], height=.74)
        ax.set_yticks(range(len(top)), top.keyword, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"主题 {t}：{TITLES[t]}\n（n={count[t]}）", fontsize=11, pad=9, linespacing=1.4)
        ax.set_xlabel("c-TF-IDF 权重", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.grid(axis="x", color="white", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    fig.suptitle("BERTopic 主题关键词", fontsize=17, y=.995)
    fig.text(.5, .007, "293条样本 · 17个主题 · 离群点46条（15.7%，未绘入主题面板）",
             ha="center", fontsize=11, color="#555555")
    fig.tight_layout(rect=(0, .025, 1, .975), h_pad=2.5, w_pad=2.0)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig3_bertopic_keywords_named_5cols.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[p.name] for p in inputs)
    (OUT / "topic_naming_notes.txt").write_text(
        "主题名称根据各主题前10个关键词、3条代表帖及混合主题的全部帖子人工归纳。\n"
        "主题14、15内容较混合，分别采用“车站命名与周边争议”“车站综合体验与出行困扰”的宽口径名称。\n"
        "仅添加名称并重绘为每行5个主题；没有改变聚类、关键词、权重或样本归属。\n",
        encoding="utf-8")
    print(info[["Topic", "Count", "主题名称"]].to_string(index=False))
    print("Verified: original model outputs unchanged; 17 panels in five columns.")


if __name__ == "__main__":
    main()
