#!/usr/bin/env python3
"""Two-GPU, resumable image captioning for a local Qwen3-VL Instruct model.

Use it from Jupyter:
    from tools.caption_xizhan_qwen3vl import caption_dataset
    captions = caption_dataset("data/XiZhan/XiZhan_media", "/models/Qwen3-VL-3B-Instruct")
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


BASE_PROMPT = """请用简体中文为这张图片生成尽可能详细、准确的描述，服务于后续的站点空间与出行体验研究。

请系统说明：整体场景与空间类型；前景、中景、背景中的物体和人物；人物数量、行为、互动和移动状态；车站相关设施、列车、闸机、站台、候车区、标识与无障碍设施；可见的秩序、拥挤程度、光照、天气、时间线索和氛围；以及清晰可读的文字、站名、方向、车次、公告或广告。

下方的帖子标题与正文仅用于帮助理解图片的拍摄语境，不是图像证据。不要把其中的信息直接当成“画面可见内容”；若上下文与图像不一致、或图像无法证实该信息，以图像为准，并明确说明不确定之处。不要臆测人物身份或不可见事实。可以自然分段详细描述；不要输出 JSON、标题或思考过程。"""

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _load_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"Invalid caption JSON: {path}")
    return data


def _atomic_write(path: Path, captions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(dict(sorted(captions.items())), file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def _chunks(items: list[str], count: int) -> list[list[str]]:
    """Round-robin sharding keeps the two workers similarly loaded."""
    return [items[index::count] for index in range(count)]


def _prompt_for_image(image_path: Path) -> str:
    """Build the per-image prompt with its post's title and body as soft context."""
    info_path = image_path.parent / "info.json"
    title = "（未提供）"
    description = "（未提供）"
    try:
        with info_path.open(encoding="utf-8") as file:
            info = json.load(file)
        if isinstance(info, dict):
            title = str(info.get("title") or title)
            description = str(info.get("desc") or description)
    except (OSError, json.JSONDecodeError):
        pass
    return f"""{BASE_PROMPT}

帖子辅助上下文（请按上述规则谨慎使用）：
<post_title>{title}</post_title>
<post_description>{description}</post_description>"""


def caption_dataset(
    input_dir: str | Path,
    model_dir: str | Path,
    output_path: str | Path = "outputs/xizhan_media_image_captions.json",
    *,
    gpu_ids: tuple[int, ...] = (0, 1),
    batch_size: int = 4,
    max_new_tokens: int = 512,
    overwrite: bool = False,
) -> dict[str, str]:
    """Caption every image below *input_dir* with one Qwen3-VL model replica per GPU.

    This public function is safe to call from Jupyter: it starts ordinary Python
    subprocesses instead of CUDA multiprocessing workers, avoiding notebook
    fork/spawn problems.  Existing output keys are skipped, so re-running it
    resumes an interrupted job.
    """
    root = Path(input_dir).expanduser().resolve()
    model = Path(model_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not model.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must contain one or more distinct GPU ids")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    captions = {} if overwrite else _load_json(output)
    paths = _images(root)
    pending = [path.relative_to(root).as_posix() for path in paths if overwrite or path.relative_to(root).as_posix() not in captions]
    print(f"total={len(paths)}, existing={len(captions)}, pending={len(pending)}, gpus={gpu_ids}")
    if not pending:
        return captions

    with tempfile.TemporaryDirectory(prefix="qwen3vl_caption_") as temporary_dir:
        temp = Path(temporary_dir)
        shards = _chunks(pending, len(gpu_ids))
        processes: list[tuple[subprocess.Popen[str], Path]] = []
        for worker_index, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            if not shard:
                continue
            manifest = temp / f"worker_{worker_index}_input.json"
            result = temp / f"worker_{worker_index}_output.json"
            manifest.write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            # Set before torch is imported in the worker: each model then sees one GPU as cuda:0.
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--input-dir", str(root), "--model-dir", str(model),
                "--manifest", str(manifest), "--result", str(result),
                "--batch-size", str(batch_size), "--max-new-tokens", str(max_new_tokens),
            ]
            print(f"starting worker {worker_index} on physical GPU {gpu_id}: {len(shard)} images")
            processes.append((subprocess.Popen(command, env=env, text=True), result))

        # Workers atomically refresh their partial JSON after every batch.  Polling
        # those files gives the notebook a single reliable progress bar even when
        # two independent GPU processes are doing the generation.
        from tqdm.auto import tqdm

        completed = 0
        with tqdm(total=len(pending), desc="Captioning images", unit="image", dynamic_ncols=True) as progress:
            while any(process.poll() is None for process, _ in processes):
                observed = sum(len(_load_json(result)) for _, result in processes)
                if observed > completed:
                    progress.update(observed - completed)
                    completed = observed
                time.sleep(0.5)
            observed = sum(len(_load_json(result)) for _, result in processes)
            if observed > completed:
                progress.update(observed - completed)

        failed = [process for process, _ in processes if process.wait() != 0]
        if failed:
            raise RuntimeError("At least one GPU worker failed; previous output is unchanged. Check the worker traceback above.")
        for _, result in processes:
            captions.update(_load_json(result))
            _atomic_write(output, captions)

    print(f"completed={len(captions)}; wrote {output}")
    return captions


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _worker(input_dir: Path, model_dir: Path, manifest: Path, result: Path, batch_size: int, max_new_tokens: int) -> None:
    # Imports deliberately live in the child, after CUDA_VISIBLE_DEVICES has been set.
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker")
    # RTX 4090 Tensor Cores and PyTorch SDPA provide the fastest available
    # attention path without requiring a separately compiled flash-attn wheel.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    paths: list[str] = json.loads(manifest.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    # Qwen is decoder-only: left padding keeps batched generation aligned.
    processor.tokenizer.padding_side = "left"
    # AutoModel also accepts a Qwen2.5-VL-3B local checkpoint if “Qwen3VL-3B”
    # was used informally to refer to it; both model families share this flow.
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
    ).to(device).eval()
    captions: dict[str, str] = {}

    def prepare(group: list[str]) -> tuple[list[str], list[str], list[object]]:
        """CPU-only work, overlapped with GPU generation of the previous batch."""
        image_paths = [input_dir / relative_path for relative_path in group]
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": _prompt_for_image(path)},
            ]}]
            for path in image_paths
        ]
        texts = []
        for message in messages:
            # This local Instruct template has no `enable_thinking` parameter.
            # Passing it through **kwargs triggers a Transformers 5 warning once
            # per image and does not alter the rendered prompt.
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        images = []
        try:
            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            return group, texts, images
        except BaseException:
            for image in images:
                image.close()
            raise

    batches = list(_batched(paths, batch_size))
    # A single prefetch thread avoids oversubscribing CPU while hiding disk I/O,
    # PIL decoding, JSON context loading, and template construction behind GPU work.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-prefetch") as executor:
        next_batch = executor.submit(prepare, batches[0])
        for index in range(len(batches)):
            group, texts, images = next_batch.result()
            if index + 1 < len(batches):
                next_batch = executor.submit(prepare, batches[index + 1])
            try:
                inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    )
                trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                captions.update({key: text.strip() for key, text in zip(group, decoded)})
                print(f"gpu={torch.cuda.get_device_name(0)} completed={len(captions)}/{len(paths)}", flush=True)
            finally:
                for image in images:
                    image.close()
            _atomic_write(result, captions)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.worker:
        raise SystemExit("Import caption_dataset() from Jupyter, rather than executing this module directly.")
    _worker(args.input_dir.resolve(), args.model_dir.resolve(), args.manifest.resolve(), args.result.resolve(), args.batch_size, args.max_new_tokens)
