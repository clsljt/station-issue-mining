"""Reflow the original ten-topic chart using the exact saved Plotly values."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/station_mpl")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from paper_common import setup_cjk_font

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/results/output/0623/bertopic_chinese_keywords_barchart.html"
OUT = ROOT / "data/results/output/1020/original_10_topics_5cols"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    start = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', source)
    decoder = json.JSONDecoder()
    traces, end = decoder.raw_decode(source[start.end():])
    remainder = source[start.end() + end:].lstrip()
    layout, _ = decoder.raw_decode(remainder[1:].lstrip())
    titles = [a["text"] for a in layout["annotations"]]
    assert len(traces) == len(titles) == 10
    chart = make_subplots(rows=2, cols=5, subplot_titles=titles,
                          horizontal_spacing=.055, vertical_spacing=.22)
    records = []
    for i, trace in enumerate(traces):
        assert len(trace["x"]) == len(trace["y"]) == 8
        bar = {k: v for k, v in trace.items() if k not in ("xaxis", "yaxis")}
        chart.add_trace(go.Bar(**bar), row=i // 5 + 1, col=i % 5 + 1)
        for word, weight in zip(trace["y"], trace["x"]):
            records.append(dict(topic=i, title=titles[i], keyword=word, weight=weight))
    chart.update_layout(title=layout["title"], template="plotly", width=2500, height=780,
                        margin=dict(l=90, r=35, t=110, b=65), showlegend=False,
                        font=dict(family="Arial Unicode MS", size=12))
    chart.update_annotations(font_size=13)
    chart.update_xaxes(title_text="TF-IDF 权重")
    chart.write_html(str(OUT / "bar_5cols.html"), include_plotlyjs=True)
    pd.DataFrame(records).to_csv(OUT / "original_keyword_weights.csv", index=False, encoding="utf-8-sig")

    setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, 5, figsize=(24, 7.2))
    for i, (ax, trace, title) in enumerate(zip(axes.flat, traces, titles)):
        ax.set_facecolor("#E8EDF5")
        ax.barh(range(8), trace["x"], color=trace["marker"]["color"], height=.74)
        ax.set_yticks(range(8), trace["y"], fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=10)
        ax.set_xlabel("TF-IDF 权重", fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.grid(axis="x", color="white", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(layout["title"]["text"], fontsize=16, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=3.0, w_pad=2.0)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"bar_5cols.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    exported = pd.read_csv(OUT / "original_keyword_weights.csv")
    for i, trace in enumerate(traces):
        subset = exported[exported.topic.eq(i)]
        assert subset.keyword.tolist() == trace["y"]
        assert all(abs(a-b) < 1e-14 for a, b in zip(subset.weight, trace["x"]))
    (OUT / "README.txt").write_text(
        f"原10主题图改为2行5列。\n源图：{SOURCE}\n"
        f"源文件SHA256：{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}\n"
        "精确复用源HTML的80个关键词、权重、主题名称和颜色，仅更改版式，没有重跑模型。\n",
        encoding="utf-8")
    print(f"Verified all 80 original keyword weights and labels. Saved {OUT}")


if __name__ == "__main__":
    main()
