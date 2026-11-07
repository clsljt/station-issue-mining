#!/usr/bin/env python3
"""Generate statistics documents for each station and overall.

Produces:
  - outputs/<station>/<station>_统计报告.md  (per-station)
  - outputs/全站舆情统计总览.md              (overall)
"""

import json
import os
from collections import Counter

STATIONS = {
    "xizhan":   {"name": "北京西站",   "file": "outputs/xizhan/xizhan_analysis.json"},
    "nanzhan":  {"name": "北京南站",   "file": "outputs/nanzhan/nanzhan_analysis.json"},
    "beizhan":  {"name": "北京北站",   "file": "outputs/beizhan/beizhan_analysis.json"},
    "fengtai":  {"name": "北京丰台站", "file": "outputs/fengtai/fengtai_analysis.json"},
    "chaoyang": {"name": "北京朝阳站", "file": "outputs/chaoyang/chaoyang_analysis.json"},
    "qinghe":   {"name": "北京清河站", "file": "outputs/qinghe/qinghe_analysis.json"},
    "tongzhou": {"name": "北京通州站", "file": "outputs/tongzhou/tongzhou_analysis.json"},
    "beijing":  {"name": "北京站",     "file": "outputs/beijing/beijing_analysis.json"},
    "daxingjichang":  {"name": "大兴机场", "file": "outputs/daxingjichang/daxingjichang_analysis.json"},
    "shoudujichang":  {"name": "首都机场", "file": "outputs/shoudujichang/shoudujichang_analysis.json"},
}

V2_CATEGORIES = [
    "智慧接驳衔接不畅",
    "导航导视指引不足",
    "室内外环境问题",
    "站区服务和商业配套",
    "城管管理，周边治理",
]

GROUP_NAMES = {"business": "业务类", "xhs": "小红书内容类", "blank": "空白类"}

XHS_CAT_NAMES = {
    "xhs_guide": "攻略",
    "xhs_share": "分享",
    "xhs_recommendation": "推荐",
    "xhs_other": "其他",
}


def pct(n, total):
    return f"{n*100/total:.1f}%" if total else "0%"


def load_all():
    all_data = {}
    for key, cfg in STATIONS.items():
        with open(cfg["file"], "r", encoding="utf-8") as f:
            all_data[key] = json.load(f)
    return all_data


def gen_station_report(key, name, data):
    total = len(data)
    lines = []
    lines.append(f"# {name} 舆情统计报告\n")
    lines.append(f"> 帖子总数：{total}\n")

    # --- 1. 业务5大类 ---
    lines.append("## 一、业务类五大类别统计\n")
    biz = [r for r in data if r.get("category_group") == "business"]
    biz_total = len(biz)
    v2_counter = Counter(r.get("business_category_v2", "未分类") for r in biz)

    lines.append("| 序号 | 类别 | 数量 | 占业务类比例 | 占总体比例 |")
    lines.append("|------|------|------|------------|----------|")
    for i, cat in enumerate(V2_CATEGORIES, 1):
        n = v2_counter.get(cat, 0)
        lines.append(f"| {i} | {cat} | {n} | {pct(n, biz_total)} | {pct(n, total)} |")
    lines.append(f"| | **合计** | **{biz_total}** | **100%** | **{pct(biz_total, total)}** |")
    lines.append("")

    # --- 2. 三组统计 ---
    lines.append("## 二、三组分类统计（业务/小红书/空白）\n")
    gc = Counter(r.get("category_group", "") for r in data)
    lines.append("| 分组 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for grp in ["business", "xhs", "blank"]:
        n = gc.get(grp, 0)
        lines.append(f"| {GROUP_NAMES.get(grp, grp)} | {n} | {pct(n, total)} |")
    lines.append(f"| **合计** | **{total}** | **100%** |")
    lines.append("")

    # --- 3. 正面/负面 ---
    lines.append("## 三、情感倾向统计（正面/负面）\n")
    sc = Counter(r.get("sentiment_label", "Unknown") for r in data)
    pos, neg = sc.get("Positive", 0), sc.get("Negative", 0)
    lines.append("| 情感 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    lines.append(f"| 正面（Positive） | {pos} | {pct(pos, total)} |")
    lines.append(f"| 负面（Negative） | {neg} | {pct(neg, total)} |")
    lines.append(f"| **合计** | **{total}** | **100%** |")
    lines.append("")

    # --- 3b. business-only sentiment ---
    lines.append("### 业务类情感分布\n")
    biz_sc = Counter(r.get("sentiment_label", "Unknown") for r in biz)
    bp, bn = biz_sc.get("Positive", 0), biz_sc.get("Negative", 0)
    lines.append("| 情感 | 数量 | 占业务类比例 |")
    lines.append("|------|------|------------|")
    lines.append(f"| 正面 | {bp} | {pct(bp, biz_total)} |")
    lines.append(f"| 负面 | {bn} | {pct(bn, biz_total)} |")
    lines.append(f"| **合计** | **{biz_total}** | **100%** |")
    lines.append("")

    return "\n".join(lines)


def gen_overall_report(all_data):
    combined = []
    for records in all_data.values():
        combined.extend(records)

    total = len(combined)
    lines = []
    lines.append("# 北京八站两场舆情统计总览\n")
    lines.append(f"> 覆盖站点：北京西站、北京南站、北京北站、北京丰台站、北京朝阳站、北京清河站、北京通州站、北京站、大兴机场、首都机场")
    lines.append(f"> 帖子总数：{total}\n")

    # --- 1. 业务5大类 ---
    lines.append("## 一、业务类五大类别统计\n")
    biz = [r for r in combined if r.get("category_group") == "business"]
    biz_total = len(biz)
    v2_counter = Counter(r.get("business_category_v2", "未分类") for r in biz)

    lines.append("| 序号 | 类别 | 数量 | 占业务类比例 | 占总体比例 |")
    lines.append("|------|------|------|------------|----------|")
    for i, cat in enumerate(V2_CATEGORIES, 1):
        n = v2_counter.get(cat, 0)
        lines.append(f"| {i} | {cat} | {n} | {pct(n, biz_total)} | {pct(n, total)} |")
    lines.append(f"| | **合计** | **{biz_total}** | **100%** | **{pct(biz_total, total)}** |")
    lines.append("")

    # --- 2. 三组统计 ---
    lines.append("## 二、三组分类统计（业务/小红书/空白）\n")
    gc = Counter(r.get("category_group", "") for r in combined)
    lines.append("| 分组 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for grp in ["business", "xhs", "blank"]:
        n = gc.get(grp, 0)
        lines.append(f"| {GROUP_NAMES.get(grp, grp)} | {n} | {pct(n, total)} |")
    lines.append(f"| **合计** | **{total}** | **100%** |")
    lines.append("")

    # --- 3. 正面/负面 ---
    lines.append("## 三、情感倾向统计（正面/负面）\n")
    sc = Counter(r.get("sentiment_label", "Unknown") for r in combined)
    pos, neg = sc.get("Positive", 0), sc.get("Negative", 0)
    lines.append("| 情感 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    lines.append(f"| 正面（Positive） | {pos} | {pct(pos, total)} |")
    lines.append(f"| 负面（Negative） | {neg} | {pct(neg, total)} |")
    lines.append(f"| **合计** | **{total}** | **100%** |")
    lines.append("")

    lines.append("### 业务类情感分布\n")
    biz_sc = Counter(r.get("sentiment_label", "Unknown") for r in biz)
    bp, bn = biz_sc.get("Positive", 0), biz_sc.get("Negative", 0)
    lines.append("| 情感 | 数量 | 占业务类比例 |")
    lines.append("|------|------|------------|")
    lines.append(f"| 正面 | {bp} | {pct(bp, biz_total)} |")
    lines.append(f"| 负面 | {bn} | {pct(bn, biz_total)} |")
    lines.append(f"| **合计** | **{biz_total}** | **100%** |")
    lines.append("")

    # --- 4. 攻略帖分布 ---
    lines.append('## 四、攻略帖分布（信息化不足的"隐性指标"）\n')
    lines.append("攻略帖数量越多，往往反映该站的信息化越不足——旅客不得不通过互联网攻略来弥补现场信息的缺失。\n")

    lines.append("| 站点 | 攻略帖 | 占比 | 分享帖 | 推荐帖 | 其他 | 攻略主要主题 |")
    lines.append("|------|--------|------|--------|--------|------|------------|")

    guide_themes = {
        "xizhan":   "进站路线、快速进站、免安检、接人流程",
        "nanzhan":  "快速进站厅、赶车路线、打车攻略",
        "beizhan":  "西直门换乘、进站流程、跨站换乘时间",
        "fengtai":  "如何不迷路、出站路线、换乘时间",
        "chaoyang": "换乘时间、凌晨出行",
        "qinghe":   "八达岭/崇礼换乘、进站流程",
        "tongzhou": "进站流程、地铁换乘、打车指引",
        "beijing":  "跨站换乘路线、进站流程、周边交通",
        "daxingjichang": "值机流程、航站楼交通、转机攻略",
        "shoudujichang": "航站楼换乘、免税店攻略、交通接驳",
    }

    total_guide = 0
    total_xhs = 0
    for key, cfg in STATIONS.items():
        records = all_data[key]
        xhs_records = [r for r in records if r.get("category_group") == "xhs"]
        xhs_total = len(xhs_records)
        guide = sum(1 for r in xhs_records if r.get("category") == "xhs_guide")
        share = sum(1 for r in xhs_records if r.get("category") == "xhs_share")
        rec = sum(1 for r in xhs_records if r.get("category") == "xhs_recommendation")
        other = sum(1 for r in xhs_records if r.get("category") == "xhs_other")
        station_total = len(records)
        guide_pct = pct(guide, station_total)
        theme = guide_themes.get(key, "")
        lines.append(f"| {cfg['name']} | **{guide}** | {guide_pct} | {share} | {rec} | {other} | {theme} |")
        total_guide += guide
        total_xhs += xhs_total

    total_share = sum(1 for r in combined if r.get("category") == "xhs_share")
    total_rec = sum(1 for r in combined if r.get("category") == "xhs_recommendation")
    total_other_xhs = sum(1 for r in combined if r.get("category") == "xhs_other")
    lines.append(f"| **合计** | **{total_guide}** | **{pct(total_guide, total)}** | {total_share} | {total_rec} | {total_other_xhs} | |")
    lines.append("")

    conclusion = f'> **结论**：攻略帖总量为 {total_guide} 条，占全部帖子的 {pct(total_guide, total)}。攻略帖越多说明现场信息化越不足，旅客被迫通过网络"自救"。'
    lines.append(conclusion)
    lines.append("")

    return "\n".join(lines)


def main():
    all_data = load_all()

    for key, cfg in STATIONS.items():
        report = gen_station_report(key, cfg["name"], all_data[key])
        out_dir = os.path.dirname(cfg["file"])
        out_path = os.path.join(out_dir, f"{key}_统计报告.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Generated: {out_path}")

    overall = gen_overall_report(all_data)
    overall_path = "outputs/全站舆情统计总览.md"
    with open(overall_path, "w", encoding="utf-8") as f:
        f.write(overall)
    print(f"Generated: {overall_path}")


if __name__ == "__main__":
    main()
