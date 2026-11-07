#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Spider_XHS 的输出目录拍平成 ``<note_id>/``，供 analyze_notes.py 使用。

Spider_XHS 的落盘结构是::

    <输出根>/<昵称>_<user_id>/<标题>_<note_id>/
        ├── info.json
        ├── detail.txt
        └── image_*.jpg

而本项目的 ``code/pipeline/analyze_notes.py`` 期望的是::

    <站>_media/<note_id>/
        ├── info.json
        ├── detail.txt
        └── image_*.jpg

用法::

    # 复制（默认），原采集目录保留
    python flatten_notes.py --src ~/xhs_dump --dst ../../data/raw/station_data/BeiJingZhan/BeiJingZhan_media

    # 移动（省空间），原采集目录会被清空
    python flatten_notes.py --src ~/xhs_dump --dst ... --move
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def find_note_dirs(root: Path) -> list[Path]:
    """找出所有含 info.json 的目录。"""
    return sorted(p.parent for p in root.rglob("info.json") if p.is_file())


def read_note_id(note_dir: Path) -> str | None:
    try:
        info = json.loads((note_dir / "info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  !! 跳过 {note_dir}：info.json 读取失败（{exc}）", file=sys.stderr)
        return None
    note_id = str(info.get("note_id") or "").strip()
    if not note_id:
        print(f"  !! 跳过 {note_dir}：info.json 里没有 note_id", file=sys.stderr)
        return None
    return note_id


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".txt", ".json"}:
            shutil.copy2(item, target)


def main() -> int:
    ap = argparse.ArgumentParser(description="拍平 Spider_XHS 输出为 <note_id>/ 结构")
    ap.add_argument("--src", required=True, type=Path, help="Spider_XHS 的输出根目录")
    ap.add_argument("--dst", required=True, type=Path, help="目标 <站>_media 目录")
    ap.add_argument("--move", action="store_true", help="移动而非复制")
    ap.add_argument("--overwrite", action="store_true", help="目标已存在时覆盖")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"源目录不存在：{args.src}", file=sys.stderr)
        return 1

    note_dirs = find_note_dirs(args.src)
    print(f"找到 {len(note_dirs)} 个帖子目录 → {args.dst}")

    args.dst.mkdir(parents=True, exist_ok=True)
    ok = skipped = dup = 0
    for note_dir in note_dirs:
        note_id = read_note_id(note_dir)
        if note_id is None:
            skipped += 1
            continue
        target = args.dst / note_id
        if target.exists() and not args.overwrite:
            dup += 1
            continue
        if args.move:
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(note_dir), str(target))
        else:
            copy_tree(note_dir, target)
        ok += 1

    print(f"完成：写入 {ok}，已存在跳过 {dup}，异常跳过 {skipped}")
    if args.move:
        # 清掉搬空后的两级空目录
        for p in sorted(args.src.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
