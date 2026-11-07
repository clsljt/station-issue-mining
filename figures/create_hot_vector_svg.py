#!/usr/bin/env python3
"""Create a Word-compatible, fully vector SVG for the topic-station heatmap.

Plotly serialises heatmap cells as a base64 PNG inside its SVG output. That
construct is poorly supported by Word and some journal-production systems.
This script instead writes every cell as an SVG ``rect`` and every label as an
SVG ``text`` element.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
INPUT_CSV = PROJECT_ROOT / "output" / "0623" / "kmeans_topic_station_share.csv"
OUTPUT_SVG = SCRIPT_DIR / "svg" / "hot.svg"

STATIONS = [
    ("xizhan", "北京西站", 37),
    ("nanzhan", "北京南站", 25),
    ("beizhan", "北京北站", 16),
    ("fengtai", "北京丰台站", 55),
    ("chaoyang", "北京朝阳站", 20),
    ("qinghe", "北京清河站", 9),
    ("tongzhou", "北京通州站", 24),
    ("beijing", "北京站", 19),
    ("daxingjichang", "北京大兴国际机场", 47),
    ("shoudujichang", "北京首都国际机场", 41),
]

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

# Keep the same topic priority order as the manuscript's analysis, but display
# it from top to bottom (the conventional reading order for a printed table).
TOPIC_ORDER = [0, 1, 3, 2, 4, 5, 6, 7, 8, 9]

WIDTH, HEIGHT = 1380, 720
LEFT, TOP = 300, 80
PLOT_WIDTH, PLOT_HEIGHT = 960, 520
RIGHT = LEFT + PLOT_WIDTH
BOTTOM = TOP + PLOT_HEIGHT
FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',Arial,sans-serif"
LOW = (247, 251, 255)
MID = (158, 202, 225)
HIGH = (8, 81, 156)


def interpolate(first: tuple[int, int, int], second: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * ratio) for a, b in zip(first, second))


def fill_color(value: float) -> str:
    """Match the three-stop blue palette previously used by Plotly."""
    value = max(0.0, min(1.0, value))
    rgb = interpolate(LOW, MID, value * 2) if value <= 0.5 else interpolate(MID, HIGH, (value - 0.5) * 2)
    return "#%02X%02X%02X" % rgb


def text_color(value: float) -> str:
    return "#FFFFFF" if value >= 0.60 else "#444444"


def load_values() -> dict[int, dict[str, float]]:
    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["topic"]): {station: float(row[station]) for station, _, _ in STATIONS}
            for row in rows
        }


def svg_text(x: float, y: float, value: str, *, size: int = 12, anchor: str = "start", fill: str = "#2A3F5F", extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT}" font-size="{size}px" fill="{fill}" {extra}>{escape(value)}</text>'
    )


def build_svg(values: dict[int, dict[str, float]]) -> str:
    column_width = PLOT_WIDTH / len(STATIONS)
    row_height = PLOT_HEIGHT / len(TOPIC_ORDER)
    chunks = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
        svg_text(69, 40, "图4-3  负向舆情主题在各站场的标准化分布", size=17),
        svg_text(LEFT + PLOT_WIDTH * 0.40, 53, "铁路客运站", anchor="middle"),
        svg_text(LEFT + PLOT_WIDTH * 0.90, 53, "机场", anchor="middle"),
        f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" fill="#F7FBFF"/>',
    ]

    for row_index, topic_id in enumerate(TOPIC_ORDER):
        y = TOP + row_index * row_height
        chunks.append(
            svg_text(LEFT - 8, y + row_height / 2 + 4, f"主题{topic_id}：{TOPIC_TITLES[topic_id]}", anchor="end")
        )
        for column_index, (station_key, _, _) in enumerate(STATIONS):
            x = LEFT + column_index * column_width
            value = values[topic_id][station_key]
            chunks.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{column_width:.1f}" height="{row_height:.1f}" '
                f'fill="{fill_color(value)}" stroke="#FFFFFF" stroke-width="1"/>'
            )
            chunks.append(
                svg_text(
                    x + column_width / 2,
                    y + row_height / 2 + 4,
                    f"{value:.1%}",
                    anchor="middle",
                    fill=text_color(value),
                )
            )

    divider_x = LEFT + column_width * 8
    chunks.append(
        f'<line x1="{divider_x:.1f}" x2="{divider_x:.1f}" y1="{TOP + 6}" y2="{BOTTOM}" '
        'stroke="#7F8C8D" stroke-width="1" stroke-dasharray="8 8"/>'
    )

    for column_index, (_, station_label, total) in enumerate(STATIONS):
        x = LEFT + (column_index + 0.5) * column_width
        chunks.append(
            f'<text x="{x:.1f}" y="{BOTTOM + 20}" text-anchor="start" '
            f'transform="rotate(30 {x:.1f} {BOTTOM + 20})" font-family="{FONT}" font-size="12px" fill="#2A3F5F">'
            f'<tspan x="{x:.1f}" dy="0">{escape(station_label)}</tspan>'
            f'<tspan x="{x:.1f}" dy="16">n={total}</tspan></text>'
        )

    colorbar_x, colorbar_y, colorbar_width, colorbar_height = 1289, 130, 30, 460
    chunks.extend(
        [
            '<defs><linearGradient id="heatmap-scale" x1="0" y1="1" x2="0" y2="0">'
            '<stop offset="0%" stop-color="#F7FBFF"/><stop offset="50%" stop-color="#9ECAE1"/>'
            '<stop offset="100%" stop-color="#08519C"/></linearGradient></defs>',
            svg_text(colorbar_x, 103, "站场内", size=14),
            svg_text(colorbar_x, 121, "主题占比", size=14),
            f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_width}" height="{colorbar_height}" fill="url(#heatmap-scale)"/>',
        ]
    )
    for percent in range(0, 101, 10):
        y = colorbar_y + colorbar_height * (1 - percent / 100)
        chunks.append(svg_text(colorbar_x + colorbar_width + 4, y + 4, f"{percent}%"))

    chunks.append("</svg>")
    return "\n".join(chunks)


def main() -> None:
    values = load_values()
    missing = [topic for topic in TOPIC_ORDER if topic not in values]
    if missing:
        raise ValueError(f"CSV缺少主题：{missing}")
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(build_svg(values), encoding="utf-8")
    print(f"已生成纯矢量SVG：{OUTPUT_SVG}")


if __name__ == "__main__":
    main()
