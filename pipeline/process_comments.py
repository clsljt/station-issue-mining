#!/usr/bin/env python3
"""Put comments into corresponding note folders as comment.json.

Usage:
    python process_comments.py --station <station_key>

Supported stations: xizhan, nanzhan, beizhan, fengtai, chaoyang, qinghe, tongzhou, beijing, daxingjichang, shoudujichang
"""

import argparse
import json
import os
import pandas as pd

STATION_CONFIG = {
    "xizhan": {
        "data_dir": "/home/lu.junting/station/data/xizhan",
        "media_dir": "XiZhan_all",
        "comments_file": "XiZhan_mments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "nanzhan": {
        "data_dir": "/home/lu.junting/station/data/nanzhan",
        "media_dir": "NanZhan_media",
        "comments_file": "NanZhan_comments.xlsx",
        "col_content": "comment_content",
        "col_likes": "likes_count",
        "col_time": "created_at",
    },
    "beizhan": {
        "data_dir": "/home/lu.junting/station/data/beizhan",
        "media_dir": "BeiZhan_media",
        "comments_file": "BeiZhan_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "fengtai": {
        "data_dir": "/home/lu.junting/station/data/fengtai",
        "media_dir": "FengTai_media",
        "comments_file": "FengTai_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "chaoyang": {
        "data_dir": "/home/lu.junting/station/data/chaoyang",
        "media_dir": "ChaoYangZhan_media",
        "comments_file": "ChaoYang_comments.xlsx",
        "col_content": "comment_content",
        "col_likes": "likes_count",
        "col_time": "created_at",
    },
    "qinghe": {
        "data_dir": "/home/lu.junting/station/data/qinghe",
        "media_dir": "QingHeZhan_20260625",
        "comments_file": "QingHeZhan_20260625_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "tongzhou": {
        "data_dir": "/home/lu.junting/station/data/tongzhou",
        "media_dir": "TongZhouZhan_media",
        "comments_file": "TongZhouZhan_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "beijing": {
        "data_dir": "/home/lu.junting/station/data/beijing",
        "media_dir": "BeiJingZhan_media",
        "comments_file": "BeiJingZhan_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "daxingjichang": {
        "data_dir": "/home/lu.junting/station/data/daxingjichang",
        "media_dir": "DaXingJiChang_media",
        "comments_file": "DaXingJiChang_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
    "shoudujichang": {
        "data_dir": "/home/lu.junting/station/data/shoudujichang",
        "media_dir": "ShouDuJiChang_media",
        "comments_file": "ShouDuJiChang_comments.xlsx",
        "col_content": "content",
        "col_likes": "like_count",
        "col_time": "upload_time",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Process comments for a station")
    parser.add_argument("--station", required=True, choices=STATION_CONFIG.keys())
    args = parser.parse_args()

    cfg = STATION_CONFIG[args.station]
    data_dir = cfg["data_dir"]
    all_dir = os.path.join(data_dir, cfg["media_dir"])
    comments_file = os.path.join(data_dir, cfg["comments_file"])

    col_content = cfg["col_content"]
    col_likes = cfg["col_likes"]
    col_time = cfg["col_time"]

    df = pd.read_excel(comments_file)
    grouped = df.groupby("note_id")

    saved = 0
    skipped = 0

    for note_id, group in grouped:
        folder = os.path.join(all_dir, str(note_id))
        if not os.path.isdir(folder):
            skipped += 1
            continue

        comments = []
        for _, row in group.iterrows():
            comment = {
                "comment_id": str(row.get("comment_id", "")),
                "comment_level": int(row["comment_level"]) if pd.notna(row.get("comment_level")) else 1,
                "root_comment_id": str(row["root_comment_id"]) if pd.notna(row.get("root_comment_id")) else None,
                "parent_comment_id": str(row["parent_comment_id"]) if pd.notna(row.get("parent_comment_id")) else None,
                "user_id": str(row.get("user_id", "")),
                "nickname": str(row.get("nickname", "")),
                "content": str(row.get(col_content, "")),
                "like_count": int(row[col_likes]) if pd.notna(row.get(col_likes)) else 0,
                "upload_time": str(row.get(col_time, "")),
                "ip_location": str(row.get("ip_location", "")),
            }
            comments.append(comment)

        out_path = os.path.join(folder, "comment.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        saved += 1

    print(f"Done: saved {saved} comment.json files, skipped {skipped} notes (no folder)")


if __name__ == "__main__":
    main()
