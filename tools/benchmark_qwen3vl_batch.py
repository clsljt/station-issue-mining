#!/usr/bin/env python3
"""Benchmark Qwen3-VL image-caption throughput for several batch sizes on one GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


PROMPT = """请用简体中文为这张图片写一条客观、可检索的视觉描述。
只描述画面中实际可见的内容，不根据文件名、帖子文字或常识补全。
说明主要场景、主体、行为/状态，以及可见的车站出行设施。
清晰可读且有信息价值的文字原样摘录；模糊文字不要猜。
输出一到两句、总长度不超过100个汉字；不要标题、列表、JSON或思考过程。"""
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def sample_paths(root: Path, size: int) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found below {root}")
    if size >= len(paths):
        return paths
    # Spread the sample across the sorted corpus, not merely the first few posts.
    return [paths[round(index * (len(paths) - 1) / (size - 1))] for index in range(size)] if size > 1 else [paths[0]]


def groups(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    paths = sample_paths(args.input_dir, args.sample_size)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda:0").eval()

    def infer(batch: list[Path]) -> None:
        images = []
        try:
            for path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": PROMPT},
            ]}] for path in batch]
            texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        finally:
            for image in images:
                image.close()

    results: list[dict[str, float | int | str]] = []
    for batch_size in args.batch_sizes:
        if batch_size < 1:
            continue
        test_groups = groups(paths, batch_size)
        try:
            infer(test_groups[0])  # warm-up vision kernels and allocator
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for batch in test_groups:
                infer(batch)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            result: dict[str, float | int | str] = {
                "batch_size": batch_size,
                "images": len(paths),
                "seconds": round(seconds, 3),
                "images_per_second": round(len(paths) / seconds, 3),
                "seconds_per_image": round(seconds / len(paths), 3),
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "status": "ok",
            }
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "status": "oom"}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
