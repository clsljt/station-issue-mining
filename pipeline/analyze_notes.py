#!/usr/bin/env python3
"""Analyze station notes with Qwen3.6-27B (multimodal): generate description + category for each note.

Usage:
    python analyze_notes.py --station <station_key>

Supported stations: xizhan, nanzhan, beizhan, fengtai, chaoyang, qinghe, tongzhou, beijing, daxingjichang, shoudujichang
"""

import argparse
import json
import os
import glob
import re
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "/home/lu.junting/station/qwen-3.6"

STATION_CONFIG = {
    "xizhan":   {"data_dir": "/home/lu.junting/station/data/xizhan/XiZhan_all",           "output": "/home/lu.junting/station/outputs/xizhan/xizhan_analysis.json",     "name": "北京西站"},
    "nanzhan":  {"data_dir": "/home/lu.junting/station/data/nanzhan/NanZhan_media",        "output": "/home/lu.junting/station/outputs/nanzhan/nanzhan_analysis.json",   "name": "北京南站"},
    "beizhan":  {"data_dir": "/home/lu.junting/station/data/beizhan/BeiZhan_media",        "output": "/home/lu.junting/station/outputs/beizhan/beizhan_analysis.json",   "name": "北京北站"},
    "fengtai":  {"data_dir": "/home/lu.junting/station/data/fengtai/FengTai_media",        "output": "/home/lu.junting/station/outputs/fengtai/fengtai_analysis.json",   "name": "北京丰台站"},
    "chaoyang": {"data_dir": "/home/lu.junting/station/data/chaoyang/ChaoYangZhan_media",  "output": "/home/lu.junting/station/outputs/chaoyang/chaoyang_analysis.json", "name": "北京朝阳站"},
    "qinghe":   {"data_dir": "/home/lu.junting/station/data/qinghe/QingHeZhan_20260625",   "output": "/home/lu.junting/station/outputs/qinghe/qinghe_analysis.json",     "name": "北京清河站"},
    "tongzhou": {"data_dir": "/home/lu.junting/station/data/tongzhou/TongZhouZhan_media",   "output": "/home/lu.junting/station/outputs/tongzhou/tongzhou_analysis.json", "name": "北京通州站"},
    "beijing":  {"data_dir": "/home/lu.junting/station/data/beijing/BeiJingZhan_media",     "output": "/home/lu.junting/station/outputs/beijing/beijing_analysis.json",   "name": "北京站"},
    "daxingjichang":  {"data_dir": "/home/lu.junting/station/data/daxingjichang/DaXingJiChang_media",  "output": "/home/lu.junting/station/outputs/daxingjichang/daxingjichang_analysis.json",  "name": "北京大兴国际机场"},
    "shoudujichang":  {"data_dir": "/home/lu.junting/station/data/shoudujichang/ShouDuJiChang_media",  "output": "/home/lu.junting/station/outputs/shoudujichang/shoudujichang_analysis.json",  "name": "北京首都国际机场"},
}

CATEGORY_DEFINITIONS = {
    "navigation_guidance": ("business", "导航导视：站内外指示牌、导向标识、地贴、电子导视、寻路、进出口、检票口、站台、车厢位置、换乘路径、问询台和人工导乘。"),
    "transport_connection": ("business", "交通接驳：网约车、出租车、自驾接送、停车、公交、地铁、步行到站及不同交通方式之间的上落客、等候和换乘衔接。"),
    "environment_hygiene": ("business", "环境卫生：垃圾、脏乱、保洁、异味、厕所、污水、漏水、积水、吸烟、虫害等卫生环境问题。"),
    "digital_information": ("business", "智慧信息化：电子屏、广播、车次/票务/检票/晚点信息、App、小程序、线上服务、Wi-Fi、信号、自助机、智能客服和数字化体验。"),
    "station_comparison": ("business", "站区比较：本站与其他车站，或不同广场/站区之间的整体或具体服务比较。"),
    "facility_accessibility": ("business", "设施无障碍：电梯、扶梯、轮椅通行、老人/母婴设施、座椅、候车休息、充电、行李寄存和大件行李通行。"),
    "passenger_flow_efficiency": ("business", "客流效率：拥挤、排队、安检、检票、闸机、进出站、步行或换乘距离、耗时、客流秩序和通行效率。"),
    "commercial_waiting": ("business", "候车商业：餐饮、便利店、零售、价格、热水、夜间开放、休息、过夜和其他候车配套服务。"),
    "staff_management": ("business", "人员管理：工作人员态度、人员不足、服务窗口、投诉反馈、问题整改、管理规定及其执行方式。"),
    "safety_emergency": ("business", "安全应急：治安、人身或财物安全、危险设施、跌倒、诈骗、事故、火情、医疗救助、疏散和应急处置。"),
    "xhs_guide": ("xhs", "攻略：路线、流程、时间、换乘、避坑等实用攻略；没有明确的站区问题或体验投诉。"),
    "xhs_share": ("xhs", "分享：普通经历、照片、见闻、打卡、怀旧等内容分享；没有明确的站区问题或体验投诉。"),
    "xhs_recommendation": ("xhs", "推荐：推荐、安利、建议他人使用或前往某服务/站点；没有明确的站区问题或体验投诉。"),
    "xhs_other": ("xhs", "其他：与小红书内容形态相关，但不属于攻略、分享或推荐的有效内容。"),
    "blank_no_information": ("blank", "无信息：文本虽完整但没有任何可归类的事实、体验、观点或诉求。"),
    "blank_chitchat": ("blank", "纯闲聊：问候、寒暄、玩笑、与车站服务无关的对话或回复。"),
    "blank_emoji": ("blank", "表情：仅有表情符号、颜文字、无实义单字、哈哈/顶/蹲/已阅等极短互动。"),
}

VALID_KEYS = list(CATEGORY_DEFINITIONS.keys())

CATEGORY_PROMPT_BLOCK = "\n".join(
    f"  - {key}: [{grp}] {desc}" for key, (grp, desc) in CATEGORY_DEFINITIONS.items()
)


def build_system_prompt(station_name: str) -> str:
    return f"""你是一位专业的交通枢纽舆情分析师。你需要分析小红书上关于{station_name}的帖子，并输出以下两项内容：

1. **帖子描述**：用一段简洁的文字（50-150字）概括帖子的核心内容，包括用户反映的主要问题、体验或观点。结合帖子图片内容进行分析（如果提供了图片）。
2. **帖子分类**：从以下分类中选择最匹配的一个分类key：

{CATEGORY_PROMPT_BLOCK}

请严格按照以下JSON格式输出，不要输出任何其他内容：
```json
{{
  "description": "帖子描述文字",
  "category": "分类key"
}}
```"""


def build_message_content(info: dict, comments: list, image_paths: list) -> list:
    content = []
    for img_path in image_paths[:4]:
        content.append({"type": "image", "image": f"file://{img_path}"})
    text_parts = []
    text_parts.append(f"【标题】{info.get('title', '无标题')}")
    text_parts.append(f"【正文】{info.get('desc', '无正文')}")
    text_parts.append(f"【点赞数】{info.get('liked_count', 0)}  【评论数】{info.get('comment_count', 0)}  【收藏数】{info.get('collected_count', 0)}")
    tags = info.get("tags", [])
    if tags:
        text_parts.append(f"【标签】{'、'.join(tags)}")
    text_parts.append(f"【发布时间】{info.get('upload_time', '未知')}")
    if comments:
        top_comments = sorted(comments, key=lambda c: c.get("like_count", 0), reverse=True)[:15]
        comment_lines = []
        for c in top_comments:
            nick = c.get("nickname", "匿名")
            c_content = c.get("content", "")
            likes = c.get("like_count", 0)
            comment_lines.append(f"  - {nick}({likes}赞): {c_content}")
        text_parts.append("【热门评论（按点赞排序）】")
        text_parts.extend(comment_lines)
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return content


def parse_response(text: str) -> dict:
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{[^{}]*"description"\s*:\s*"[^"]*"[^{}]*"category"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {"description": text.strip()[:200], "category": "xhs_other", "parse_error": True}
    try:
        result = json.loads(json_str)
        if "description" in result and "category" in result:
            if result["category"] not in VALID_KEYS:
                result["category_raw"] = result["category"]
                result["category"] = "xhs_other"
            return result
    except json.JSONDecodeError:
        pass
    return {"description": text.strip()[:200], "category": "xhs_other", "parse_error": True}


def main():
    parser = argparse.ArgumentParser(description="Analyze station notes with Qwen3.6-27B")
    parser.add_argument("--station", required=True, choices=STATION_CONFIG.keys())
    args = parser.parse_args()

    cfg = STATION_CONFIG[args.station]
    data_dir = cfg["data_dir"]
    output_file = cfg["output"]
    station_name = cfg["name"]
    system_prompt = build_system_prompt(station_name)

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    note_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    print(f"Found {len(note_dirs)} notes to analyze.")

    results = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results.")
    done_ids = {r["note_id"] for r in results}

    for idx, note_dir in enumerate(note_dirs):
        note_id = os.path.basename(note_dir)
        if note_id in done_ids:
            continue

        info_path = os.path.join(note_dir, "info.json")
        comment_path = os.path.join(note_dir, "comment.json")

        if not os.path.exists(info_path):
            print(f"[{idx+1}/{len(note_dirs)}] {note_id}: no info.json, skipping")
            continue

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        comments = []
        if os.path.exists(comment_path):
            with open(comment_path, "r", encoding="utf-8") as f:
                comments = json.load(f)

        image_paths = sorted(glob.glob(os.path.join(note_dir, "image_*.jpg")))
        user_content = build_message_content(info, comments, image_paths)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        images = []
        for img_path in image_paths[:4]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  Warning: failed to load {img_path}: {e}")

        if images:
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - t0

        if response_text.startswith("\n\n</think>\n\n"):
            response_text = response_text[len("\n\n</think>\n\n"):]

        parsed = parse_response(response_text)

        record = {
            "note_id": note_id,
            "title": info.get("title", ""),
            "upload_time": info.get("upload_time", ""),
            "liked_count": info.get("liked_count", 0),
            "comment_count": info.get("comment_count", 0),
            "description": parsed.get("description", ""),
            "category": parsed.get("category", "xhs_other"),
            "category_group": CATEGORY_DEFINITIONS.get(parsed.get("category", "xhs_other"), ("unknown", ""))[0],
            "raw_response": response_text,
            "num_images": len(images),
        }
        if parsed.get("parse_error"):
            record["parse_error"] = True

        results.append(record)
        done_ids.add(note_id)

        cat_label = CATEGORY_DEFINITIONS.get(record["category"], ("?", "?"))[1][:20]
        print(f"[{idx+1}/{len(note_dirs)}] {note_id} | {record['category']} ({cat_label}...) | imgs={len(images)} | {elapsed:.1f}s")

        for img in images:
            img.close()

        if (idx + 1) % 5 == 0 or idx == len(note_dirs) - 1:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(results)} notes analyzed. Results saved to {output_file}")

    from collections import Counter
    cat_counts = Counter(r["category"] for r in results)
    group_counts = Counter(r["category_group"] for r in results)
    print("\n=== Category distribution ===")
    for cat, count in cat_counts.most_common():
        grp, desc = CATEGORY_DEFINITIONS.get(cat, ("?", "?"))
        print(f"  {cat} [{grp}]: {count} ({desc[:30]}...)")
    print(f"\n=== Group distribution ===")
    for grp, count in group_counts.most_common():
        print(f"  {grp}: {count}")


if __name__ == "__main__":
    main()
