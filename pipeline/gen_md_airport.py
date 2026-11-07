#!/usr/bin/env python3
"""Generate Markdown reports for airport analysis results."""

import json
from collections import Counter, defaultdict

def safe_int(v):
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        v = v.strip()
        if v.endswith('万'):
            return int(float(v[:-1]) * 10000)
        try:
            return int(v)
        except:
            return 0
    return 0


def influence(r):
    return safe_int(r.get('liked_count', 0)) * 2 + safe_int(r.get('comment_count', 0)) * 3


CAT_LABELS = {
    "navigation_guidance": "站内导航",
    "transport_connection": "交通接驳",
    "environment_hygiene": "环境卫生",
    "digital_information": "智慧信息化",
    "station_comparison": "站区比较",
    "facility_accessibility": "设施无障碍",
    "passenger_flow_efficiency": "客流效率",
    "commercial_waiting": "候车/商业配套",
    "staff_management": "人员管理",
    "safety_emergency": "安全应急",
}

BIZ_ORDER = [
    "transport_connection", "passenger_flow_efficiency", "navigation_guidance",
    "staff_management", "commercial_waiting", "digital_information",
    "station_comparison", "environment_hygiene", "facility_accessibility",
    "safety_emergency",
]


def gen_report(data, name, total_comments):
    total = len(data)
    biz = [r for r in data if r.get('category_group') == 'business']
    xhs = [r for r in data if r.get('category_group') == 'xhs']
    blank = [r for r in data if r.get('category_group') == 'blank']

    total_inf = sum(influence(r) for r in biz) if biz else 1

    cat_counter = Counter(r['category'] for r in biz)

    lines = []
    lines.append(f"# {name}小红书舆情分析报告\n")
    lines.append(f"> 数据来源：小红书平台 · {total}条帖子 · {total_comments}条评论  ")
    lines.append(f"> 分析工具：Qwen3.6-27B 多模态大语言模型（图文联合分析）  ")
    lines.append(f"> 分析时间：2026年6月\n")
    lines.append("---\n")

    # === 一、摘要 ===
    lines.append("## 一、摘要\n")
    lines.append(f"本报告基于小红书平台采集的 **{total}条** {name}相关帖子及 **{total_comments}条** 用户评论，使用 Qwen3.6-27B 大模型进行多模态分析。\n")
    lines.append("核心发现：\n")

    biz_pct = f"{len(biz)*100/total:.1f}" if total else "0"
    xhs_pct = f"{len(xhs)*100/total:.1f}" if total else "0"
    blank_pct = f"{len(blank)*100/total:.1f}" if total else "0"
    lines.append(f"1. 业务类帖子 **{len(biz)}条**（{biz_pct}%），小红书内容类 **{len(xhs)}条**（{xhs_pct}%）" +
                 (f"，空白类 **{len(blank)}条**（{blank_pct}%）" if blank else "") +
                 f"；业务类占比{biz_pct}%，{'超过' if len(biz) > len(xhs) else '低于'}内容类；")

    top_cats = [(c, cat_counter[c]) for c in BIZ_ORDER if cat_counter.get(c, 0) > 0][:3]
    if top_cats:
        top_str = "、".join(f"**{CAT_LABELS[c]}**（{n}条）" for c, n in top_cats)
        lines.append(f"2. 业务类热点TOP3：{top_str}；")

    biz_neg = [r for r in biz if r.get('sentiment_label') == 'Negative']
    lines.append(f"3. 业务类负面帖子 **{len(biz_neg)}条**，占业务类的 **{len(biz_neg)*100/len(biz):.1f}%**；" if biz else "")

    guide_count = sum(1 for r in xhs if r.get('category') == 'xhs_guide')
    lines.append(f"4. 攻略帖 **{guide_count}条**（占总帖子{guide_count*100/total:.1f}%），反映旅客对{name}出行信息的强烈需求；")

    top_post = max(biz, key=lambda r: influence(r)) if biz else None
    if top_post:
        lines.append(f"5. 单帖最高影响力：「{top_post.get('title','无标题')[:30]}」（{safe_int(top_post.get('liked_count',0))}赞/{safe_int(top_post.get('comment_count',0))}评），综合影响力 {influence(top_post)}。")

    lines.append("\n---\n")

    # === 二、重点领域分析 ===
    lines.append("## 二、重点领域分析\n")

    section_num = 0
    for cat in BIZ_ORDER:
        posts = [r for r in biz if r['category'] == cat]
        if not posts:
            continue
        section_num += 1
        cat_inf = sum(influence(r) for r in posts)
        cat_inf_pct = f"{cat_inf*100/total_inf:.1f}" if total_inf else "0"
        lines.append(f"### {section_num}. {CAT_LABELS[cat]}（{len(posts)}条，综合影响力占比 {cat_inf_pct}%）\n")

        lines.append("| note_id | 帖子 | 赞 | 评 | 藏 | 转 | 核心问题 |")
        lines.append("|--------|------|----|----|----|----|---------|")

        posts_sorted = sorted(posts, key=lambda r: influence(r), reverse=True)
        if len(posts_sorted) <= 8:
            for r in posts_sorted:
                title = r.get('title', '无标题')[:30]
                desc = r.get('description', '')[:40]
                lines.append(f"| {r['note_id']} | {title} | {safe_int(r.get('liked_count',0))} | {safe_int(r.get('comment_count',0))} | {safe_int(r.get('collected_count',0))} | {safe_int(r.get('share_count',0))} | {desc} |")
        else:
            for r in posts_sorted[:6]:
                title = r.get('title', '无标题')[:30]
                desc = r.get('description', '')[:40]
                lines.append(f"| {r['note_id']} | {title} | {safe_int(r.get('liked_count',0))} | {safe_int(r.get('comment_count',0))} | {safe_int(r.get('collected_count',0))} | {safe_int(r.get('share_count',0))} | {desc} |")
            rest = posts_sorted[6:]
            rest_likes = sum(safe_int(r.get('liked_count',0)) for r in rest)
            rest_comments = sum(safe_int(r.get('comment_count',0)) for r in rest)
            rest_collected = sum(safe_int(r.get('collected_count',0)) for r in rest)
            rest_shares = sum(safe_int(r.get('share_count',0)) for r in rest)
            lines.append(f"| {rest[0]['note_id']} 等{len(rest)}条 | 其他（{len(rest)}条） | {rest_likes} | {rest_comments} | {rest_collected} | {rest_shares} | 其他相关问题 |")

        neg_posts = [r for r in posts if r.get('sentiment_label') == 'Negative']
        pos_posts = [r for r in posts if r.get('sentiment_label') == 'Positive']
        lines.append(f"\n**要点总结**：{CAT_LABELS[cat]}类帖子共{len(posts)}条，其中负面{len(neg_posts)}条、正面{len(pos_posts)}条。" +
                     (f"最高影响力帖子「{posts_sorted[0].get('title','无标题')[:20]}」（{influence(posts_sorted[0])}分）。" if posts_sorted else ""))
        if neg_posts:
            top_neg = max(neg_posts, key=lambda r: influence(r))
            lines.append(f"核心负面议题：{top_neg.get('description','')[:80]}。")
        lines.append("\n---\n")

    # === 三、负面信息分类及综合影响力 ===
    lines.append("## 三、负面信息分类及综合影响力分析\n")
    lines.append("> 综合影响力 = 点赞数 × 2 + 评论数 × 3\n")

    neg_by_cat = defaultdict(list)
    for r in biz_neg:
        neg_by_cat[r['category']].append(r)

    neg_total_inf = sum(influence(r) for r in biz_neg) if biz_neg else 1

    neg_ranked = sorted(neg_by_cat.items(), key=lambda x: sum(influence(r) for r in x[1]), reverse=True)

    lines.append("| 排名 | 负面信息类别 | 帖子数 | 帖子占比 | 综合影响力 | 影响力占比 |")
    lines.append("|------|-----------|--------|--------|----------|----------|")
    for rank, (cat, posts) in enumerate(neg_ranked, 1):
        cat_inf = sum(influence(r) for r in posts)
        lines.append(f"| {rank} | {CAT_LABELS.get(cat, cat)} | {len(posts)} | {len(posts)*100/len(biz_neg):.1f}% | {cat_inf} | {cat_inf*100/neg_total_inf:.1f}% |")
    lines.append(f"| | **合计** | **{len(biz_neg)}** | **100%** | **{neg_total_inf}** | **100%** |")
    lines.append("")

    # ASCII bar chart
    lines.append("```")
    max_inf = max((sum(influence(r) for r in posts) for _, posts in neg_ranked), default=1)
    for cat, posts in neg_ranked:
        cat_inf = sum(influence(r) for r in posts)
        bar_len = int(cat_inf / max_inf * 30) if max_inf else 0
        label = CAT_LABELS.get(cat, cat)
        lines.append(f"  {label:10s} {'█' * bar_len} {cat_inf}")
    lines.append("```\n")

    # === 四、小红书内容类帖子结构 ===
    lines.append("## 四、小红书内容类帖子结构\n")
    xhs_cats = Counter(r['category'] for r in xhs)
    lines.append("| 类型 | 数量 | 占内容类比例 | 占总体比例 |")
    lines.append("|------|------|------------|----------|")
    xhs_total = len(xhs) or 1
    for cat_key, cat_name in [("xhs_guide", "攻略"), ("xhs_share", "分享"), ("xhs_recommendation", "推荐"), ("xhs_other", "其他")]:
        n = xhs_cats.get(cat_key, 0)
        lines.append(f"| {cat_name} | {n} | {n*100/xhs_total:.1f}% | {n*100/total:.1f}% |")
    lines.append("")

    # === 五、建议关注事项 ===
    lines.append("## 五、建议关注事项\n")
    suggestions = []
    if cat_counter.get('transport_connection', 0) > 0:
        suggestions.append(f"**交通接驳优化**：交通接驳类帖子{cat_counter['transport_connection']}条，建议优化网约车/出租车接驳体验，提升地铁线路性价比宣传")
    if cat_counter.get('passenger_flow_efficiency', 0) > 0:
        suggestions.append(f"**客流效率提升**：客流效率类帖子{cat_counter['passenger_flow_efficiency']}条，建议优化值机/安检/行李提取流程，减少旅客等待时间")
    if cat_counter.get('staff_management', 0) > 0:
        suggestions.append(f"**服务态度改进**：人员管理类帖子{cat_counter['staff_management']}条，建议加强一线员工服务培训")
    if cat_counter.get('commercial_waiting', 0) > 0:
        suggestions.append(f"**商业配套完善**：候车商业类帖子{cat_counter['commercial_waiting']}条，建议丰富餐饮和商业选择，优化价格合理性")
    if cat_counter.get('digital_information', 0) > 0:
        suggestions.append(f"**信息化升级**：智慧信息化类帖子{cat_counter['digital_information']}条，建议完善航班延误信息推送机制")
    if guide_count > 5:
        suggestions.append(f'**信息化补强**：攻略帖高达{guide_count}条（{guide_count*100/total:.1f}%），旅客大量依赖网络"自救"，建议加强现场信息引导')
    for i, s in enumerate(suggestions[:5], 1):
        lines.append(f"{i}. {s}")
    lines.append("")

    return "\n".join(lines)


def main():
    configs = [
        {
            "key": "daxingjichang",
            "name": "北京大兴国际机场",
            "analysis": "/home/lu.junting/station/outputs/daxingjichang/daxingjichang_analysis.json",
            "output": "/home/lu.junting/station/outputs/daxingjichang/大兴机场舆情分析报告.md",
            "comments": 14779,
        },
        {
            "key": "shoudujichang",
            "name": "北京首都国际机场",
            "analysis": "/home/lu.junting/station/outputs/shoudujichang/shoudujichang_analysis.json",
            "output": "/home/lu.junting/station/outputs/shoudujichang/首都机场舆情分析报告.md",
            "comments": 7586,
        },
    ]

    for cfg in configs:
        with open(cfg["analysis"], "r", encoding="utf-8") as f:
            data = json.load(f)
        report = gen_report(data, cfg["name"], cfg["comments"])
        with open(cfg["output"], "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Generated: {cfg['output']} ({len(data)} records)")


if __name__ == "__main__":
    main()
