#!/usr/bin/env python3
"""Generate station sentiment analysis Word report based on Qwen analysis results.

Usage:
    python generate_report.py --station <station_key>

Supported stations: xizhan, nanzhan, beizhan, fengtai, chaoyang, qinghe, tongzhou, beijing, daxingjichang, shoudujichang
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

STATION_CONFIG = {
    "xizhan":   {"analysis": "/home/lu.junting/station/outputs/xizhan/xizhan_analysis.json",     "output": "/home/lu.junting/station/outputs/xizhan/西站舆情分析报告.docx",       "name": "北京西站"},
    "nanzhan":  {"analysis": "/home/lu.junting/station/outputs/nanzhan/nanzhan_analysis.json",   "output": "/home/lu.junting/station/outputs/nanzhan/南站舆情分析报告.docx",      "name": "北京南站"},
    "beizhan":  {"analysis": "/home/lu.junting/station/outputs/beizhan/beizhan_analysis.json",   "output": "/home/lu.junting/station/outputs/beizhan/北站舆情分析报告.docx",      "name": "北京北站"},
    "fengtai":  {"analysis": "/home/lu.junting/station/outputs/fengtai/fengtai_analysis.json",   "output": "/home/lu.junting/station/outputs/fengtai/丰台站舆情分析报告.docx",    "name": "北京丰台站"},
    "chaoyang": {"analysis": "/home/lu.junting/station/outputs/chaoyang/chaoyang_analysis.json", "output": "/home/lu.junting/station/outputs/chaoyang/朝阳站舆情分析报告.docx",   "name": "北京朝阳站"},
    "qinghe":   {"analysis": "/home/lu.junting/station/outputs/qinghe/qinghe_analysis.json",     "output": "/home/lu.junting/station/outputs/qinghe/清河站舆情分析报告.docx",     "name": "北京清河站"},
    "tongzhou": {"analysis": "/home/lu.junting/station/outputs/tongzhou/tongzhou_analysis.json", "output": "/home/lu.junting/station/outputs/tongzhou/通州站舆情分析报告.docx",   "name": "北京通州站"},
    "beijing":  {"analysis": "/home/lu.junting/station/outputs/beijing/beijing_analysis.json",   "output": "/home/lu.junting/station/outputs/beijing/北京站舆情分析报告.docx",    "name": "北京站"},
    "daxingjichang":  {"analysis": "/home/lu.junting/station/outputs/daxingjichang/daxingjichang_analysis.json",  "output": "/home/lu.junting/station/outputs/daxingjichang/大兴机场舆情分析报告.docx",    "name": "北京大兴国际机场"},
    "shoudujichang":  {"analysis": "/home/lu.junting/station/outputs/shoudujichang/shoudujichang_analysis.json",  "output": "/home/lu.junting/station/outputs/shoudujichang/首都机场舆情分析报告.docx",    "name": "北京首都国际机场"},
}

CATEGORY_DEFINITIONS = {
    "navigation_guidance": ("business", "导航导视"),
    "transport_connection": ("business", "交通接驳"),
    "environment_hygiene": ("business", "环境卫生"),
    "digital_information": ("business", "智慧信息化"),
    "station_comparison": ("business", "站区比较"),
    "facility_accessibility": ("business", "设施无障碍"),
    "passenger_flow_efficiency": ("business", "客流效率"),
    "commercial_waiting": ("business", "候车商业"),
    "staff_management": ("business", "人员管理"),
    "safety_emergency": ("business", "安全应急"),
    "xhs_guide": ("xhs", "攻略"),
    "xhs_share": ("xhs", "分享"),
    "xhs_recommendation": ("xhs", "推荐"),
    "xhs_other": ("xhs", "其他"),
    "blank_no_information": ("blank", "无信息"),
    "blank_chitchat": ("blank", "纯闲聊"),
    "blank_emoji": ("blank", "表情"),
}

GROUP_NAMES = {
    "business": "业务类",
    "xhs": "小红书内容类",
    "blank": "空白类",
}


def add_heading_run(para, text, size=16, bold=True, color=None):
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run


def build_report(results, station_name, output_docx):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "微软雅黑"
    style.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    add_heading_run(title, f"{station_name}舆情分析报告", size=22, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    add_heading_run(subtitle, "基于小红书平台数据 · Qwen3.6-27B AI分析", size=12, bold=False, color=(128, 128, 128))

    total = len(results)
    cat_counter = Counter(r["category"] for r in results)
    group_counter = Counter(r["category_group"] for r in results)

    by_month = defaultdict(list)
    for r in results:
        t = r.get("upload_time", "")
        if t:
            try:
                dt = datetime.strptime(t[:7], "%Y-%m")
                by_month[dt.strftime("%Y年%m月")].append(r)
            except ValueError:
                by_month["未知"].append(r)

    doc.add_paragraph()
    sec1 = doc.add_paragraph()
    add_heading_run(sec1, "摘要：", size=14, bold=True)

    biz_count = group_counter.get("business", 0)
    xhs_count = group_counter.get("xhs", 0)
    blank_count = group_counter.get("blank", 0)

    summary_points = [
        f"本报告基于小红书平台采集的{total}条{station_name}相关帖子，使用Qwen3.6-27B大语言模型进行自动化舆情分析。",
        f"业务类帖子{biz_count}条（占{biz_count*100//total if total else 0}%），"
        f"小红书内容类{xhs_count}条（占{xhs_count*100//total if total else 0}%），"
        f"空白类{blank_count}条（占{blank_count*100//total if total else 0}%）。",
    ]

    biz_cats = [(k, v) for k, v in cat_counter.most_common() if CATEGORY_DEFINITIONS.get(k, ("", ""))[0] == "business"]
    if biz_cats:
        top3 = biz_cats[:3]
        top3_str = "、".join(f"{CATEGORY_DEFINITIONS[k][1]}（{v}条）" for k, v in top3)
        summary_points.append(f"业务类热点问题TOP3：{top3_str}。")

    for i, pt in enumerate(summary_points):
        p = doc.add_paragraph()
        p.add_run(f"{i+1}、{pt}")

    doc.add_paragraph()
    sec2 = doc.add_paragraph()
    add_heading_run(sec2, "一、数据总结", size=14, bold=True)

    p = doc.add_paragraph()
    add_heading_run(p, "1、分类总览", size=12, bold=True)

    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = "序号", "类别", "条数", "占比"
    for i, (cat, count) in enumerate(sorted(cat_counter.items(), key=lambda x: -x[1])):
        grp, label = CATEGORY_DEFINITIONS.get(cat, ("?", cat))
        row = table.add_row().cells
        row[0].text, row[1].text = str(i + 1), f"[{GROUP_NAMES.get(grp, grp)}] {label}"
        row[2].text, row[3].text = str(count), f"{count*100//total}%"

    doc.add_paragraph()
    p = doc.add_paragraph()
    add_heading_run(p, "2、分组统计", size=12, bold=True)

    table2 = doc.add_table(rows=1, cols=4)
    table2.style = "Light Grid Accent 1"
    hdr2 = table2.rows[0].cells
    hdr2[0].text, hdr2[1].text, hdr2[2].text, hdr2[3].text = "序号", "分组", "条数", "占比"
    for i, (grp, count) in enumerate(group_counter.most_common()):
        row = table2.add_row().cells
        row[0].text, row[1].text = str(i + 1), GROUP_NAMES.get(grp, grp)
        row[2].text, row[3].text = str(count), f"{count*100//total}%"

    doc.add_paragraph()
    sec3 = doc.add_paragraph()
    add_heading_run(sec3, "二、具体内容", size=14, bold=True)

    for mi, month in enumerate(sorted(by_month.keys())):
        notes = by_month[month]
        p = doc.add_paragraph()
        add_heading_run(p, f"{mi+1}、{month}舆情分析（{len(notes)}条）", size=12, bold=True)
        month_cats = Counter(n["category"] for n in notes)
        month_groups = Counter(n["category_group"] for n in notes)
        biz_n, xhs_n = month_groups.get("business", 0), month_groups.get("xhs", 0)
        if biz_n > 0:
            biz_top = [(k, v) for k, v in month_cats.most_common() if CATEGORY_DEFINITIONS.get(k, ("", ""))[0] == "business"]
            if biz_top:
                biz_str = "、".join(f"{CATEGORY_DEFINITIONS[k][1]}（{v}条）" for k, v in biz_top[:3])
                doc.add_paragraph(f"业务类{biz_n}条，主要问题：{biz_str}。")
        if xhs_n > 0:
            doc.add_paragraph(f"小红书内容类{xhs_n}条。")
        for note in sorted(notes, key=lambda x: x.get("upload_time", "")):
            cat = note.get("category", "xhs_other")
            _, label = CATEGORY_DEFINITIONS.get(cat, ("?", cat))
            bullet = doc.add_paragraph(style="List Bullet")
            run_title = bullet.add_run(f"[{label}] {note.get('title', '无标题')}")
            run_title.bold = True
            bullet.add_run(f"\n{note.get('description', '暂无描述')}")

    doc.add_paragraph()
    sec4 = doc.add_paragraph()
    add_heading_run(sec4, "三、业务类问题详细分析", size=14, bold=True)

    biz_by_cat = defaultdict(list)
    for r in results:
        if r.get("category_group") == "business":
            biz_by_cat[r["category"]].append(r)
    for cat in sorted(biz_by_cat.keys(), key=lambda c: -len(biz_by_cat[c])):
        items = biz_by_cat[cat]
        _, label = CATEGORY_DEFINITIONS.get(cat, ("?", cat))
        p = doc.add_paragraph()
        add_heading_run(p, f"● {label}（{len(items)}条）", size=11, bold=True)
        for item in items:
            doc.add_paragraph(f"  - {item.get('title', '')}：{item.get('description', '暂无描述')}")

    doc.save(output_docx)
    print(f"Report saved to {output_docx}")


def main():
    parser = argparse.ArgumentParser(description="Generate Word report for a station")
    parser.add_argument("--station", required=True, choices=STATION_CONFIG.keys())
    args = parser.parse_args()

    cfg = STATION_CONFIG[args.station]
    with open(cfg["analysis"], "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} analysis results for {cfg['name']}.")
    build_report(results, cfg["name"], cfg["output"])


if __name__ == "__main__":
    main()
