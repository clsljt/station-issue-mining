#!/usr/bin/env python3
"""Create a resumable {relative_image_filename: Chinese_caption} JSON file.

The script targets an OpenAI-compatible local vLLM endpoint serving
Qwen/Qwen3-VL-8B-Instruct.  It intentionally sends one image per request:
this preserves a one-to-one, inspectable caption for every image in a post.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


PROMPT = """请用简体中文为这张图片写一条客观、可检索的视觉描述。
要求：
1. 只描述画面中实际可见的内容，不根据文件名、帖子文字或常识补全。
2. 说明主要场景、主体、行为/状态，以及与车站出行有关的可见设施（如站台、候车厅、闸机、标识、列车）；若与车站无关也如实描述。
3. 对清晰可读且有信息价值的文字（站名、方向、车次、公告等）原样摘录；模糊文字不要猜。
4. 不评价、不臆测人物身份或地点；看不清时明确写“无法辨认”。
5. 输出一到两句，总长度不超过 100 个汉字；不要标题、列表、JSON 或思考过程。"""

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/XiZhan/XiZhan_media"))
    parser.add_argument("--output", type=Path, default=Path("outputs/xizhan_media_image_captions.json"))
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate keys already in the JSON output.")
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_existing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise ValueError(f"{path} must contain a JSON object of string keys and string captions")
    return data


def atomic_write(path: Path, captions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(dict(sorted(captions.items())), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        temporary_path = Path(fh.name)
    temporary_path.replace(path)


def image_data_url(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}[path.suffix.lower()]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def caption_one(client: AsyncOpenAI, model: str, path: Path, max_tokens: int) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        # Qwen3-VL Instruct captions do not benefit from a reasoning trace.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content: Any = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The server returned an empty caption")
    return content.strip()


async def run(args: argparse.Namespace) -> None:
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    captions = {} if args.overwrite else read_existing(args.output)
    pending = [(path, path.relative_to(root).as_posix()) for path in image_paths(root)]
    pending = [(path, key) for path, key in pending if key not in captions]
    print(f"images total={len(image_paths(root))}, existing={len(captions)}, pending={len(pending)}")
    if not pending:
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600.0, max_retries=3)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0

    async def worker(path: Path, key: str) -> tuple[str, str]:
        async with semaphore:
            return key, await caption_one(client, args.model, path, args.max_tokens)

    tasks = [asyncio.create_task(worker(path, key)) for path, key in pending]
    try:
        for task in asyncio.as_completed(tasks):
            key, caption = await task
            captions[key] = caption
            completed += 1
            # Persist after every success: Ctrl-C or a node restart loses at most one result.
            atomic_write(args.output, captions)
            print(f"[{completed}/{len(pending)}] {key}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
