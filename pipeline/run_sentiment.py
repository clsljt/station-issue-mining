#!/usr/bin/env python3
"""Run sentiment analysis on all station analysis JSONs using the BERT model.

Adds sentiment_label (Positive/Negative) and sentiment_score to each record,
plus the new 5-category mapping (business_category_v2).
"""

import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

SENTIMENT_MODEL_PATH = "/home/lu.junting/station/senModel"

ANALYSIS_FILES = [
    "/home/lu.junting/station/outputs/xizhan/xizhan_analysis.json",
    "/home/lu.junting/station/outputs/nanzhan/nanzhan_analysis.json",
    "/home/lu.junting/station/outputs/beizhan/beizhan_analysis.json",
    "/home/lu.junting/station/outputs/fengtai/fengtai_analysis.json",
    "/home/lu.junting/station/outputs/chaoyang/chaoyang_analysis.json",
    "/home/lu.junting/station/outputs/qinghe/qinghe_analysis.json",
    "/home/lu.junting/station/outputs/tongzhou/tongzhou_analysis.json",
    "/home/lu.junting/station/outputs/beijing/beijing_analysis.json",
    "/home/lu.junting/station/outputs/daxingjichang/daxingjichang_analysis.json",
    "/home/lu.junting/station/outputs/shoudujichang/shoudujichang_analysis.json",
]

CATEGORY_V2_MAPPING = {
    "transport_connection":       "智慧接驳衔接不畅",
    "navigation_guidance":        "导航导视指引不足",
    "passenger_flow_efficiency":  "导航导视指引不足",
    "environment_hygiene":        "室内外环境问题",
    "facility_accessibility":     "室内外环境问题",
    "safety_emergency":           "室内外环境问题",
    "commercial_waiting":         "站区服务和商业配套",
    "staff_management":           "站区服务和商业配套",
    "digital_information":        "站区服务和商业配套",
    "station_comparison":         "城管管理，周边治理",
}


def main():
    print("Loading sentiment model...")
    tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Sentiment model loaded on {device}.")

    id2label = {0: "Negative", 1: "Positive"}

    total_processed = 0

    for filepath in ANALYSIS_FILES:
        station_name = os.path.basename(os.path.dirname(filepath))
        print(f"\nProcessing {station_name} ({filepath})...")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, record in enumerate(data):
            desc = record.get("description", "")
            if not desc:
                record["sentiment_label"] = "Neutral"
                record["sentiment_score"] = 0.0
            else:
                text = desc[:510]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).item()
                    score = probs[0][pred_id].item()

                record["sentiment_label"] = id2label[pred_id]
                record["sentiment_score"] = round(score, 4)

            old_cat = record.get("category", "")
            if old_cat in CATEGORY_V2_MAPPING:
                record["business_category_v2"] = CATEGORY_V2_MAPPING[old_cat]
            else:
                record["business_category_v2"] = None

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        pos = sum(1 for r in data if r.get("sentiment_label") == "Positive")
        neg = sum(1 for r in data if r.get("sentiment_label") == "Negative")
        print(f"  {station_name}: {len(data)} records | Positive: {pos} | Negative: {neg}")
        total_processed += len(data)

    print(f"\nDone! Processed {total_processed} records across {len(ANALYSIS_FILES)} stations.")


if __name__ == "__main__":
    main()
