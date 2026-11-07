#!/usr/bin/env python3
"""Convert PDF, SVG, and Plotly HTML figures to EPS, EMF, or SVG.

Examples:
    python3 convert_figures_to_eps.py --format eps
    conda run -n py312 python convert_figures_to_eps.py --format svg --paper-html
    conda run -n py312 python convert_figures_to_eps.py --format emf --paper-html
    python3 convert_figures_to_eps.py path/to/figure.svg

The script preserves vectors whenever possible. Plotly HTML is parsed directly
into SVG with Kaleido; this path does not launch Chromium or any browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".svg", ".html", ".htm"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eps"
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "v2.ipynb"

# These are the four Plotly HTML figures written by v2.ipynb and referenced by
# the manuscript as bar.png, vs.png, hot.png, and yxl.png, respectively.
PAPER_HTML_FIGURES = {
    "bertopic_chinese_keywords_barchart.html": "bar",
    "kmeans_topic_distribution_vs_impact.html": "vs",
    "kmeans_topic_station_heatmap.html": "hot",
    "kmeans_topic_scale_impact_quadrant.html": "yxl",
}

# Dimensions already used by the manuscript's four static Plotly figures.
# Keeping them here makes the HTML-to-EMF result match the paper layout.
PAPER_HTML_RENDER_SIZES = {
    "bertopic_chinese_keywords_barchart.html": (2252, 990),
    "kmeans_topic_distribution_vs_impact.html": (2252, 620),
    "kmeans_topic_station_heatmap.html": (1380, 720),
    "kmeans_topic_scale_impact_quadrant.html": (1120, 720),
}


class ConversionError(RuntimeError):
    """Raised when no suitable converter is available or a conversion fails."""


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    destination: Path
    converter: str


def command_path(*names: str) -> str | None:
    """Return the first available executable from a list of candidate names."""
    for name in names:
        found = shutil.which(name)
        if found and Path(found).exists():
            return found
        candidate = Path(name)
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate)
    return None


def run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ConversionError(f"未找到转换程序：{command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "转换程序未返回可读错误信息").strip()
        raise ConversionError(f"命令执行失败：{' '.join(command)}\n{detail}") from exc


def assert_eps(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ConversionError(f"未生成 EPS 文件：{path}")
    with path.open("rb") as handle:
        header = handle.read(32)
    if not header.startswith(b"%!PS"):
        raise ConversionError(f"输出不是有效的 PostScript/EPS 文件：{path}")


def assert_emf(path: Path) -> None:
    """Verify the output has the EMF signature at the standard header offset."""
    if not path.exists() or path.stat().st_size < 44:
        raise ConversionError(f"未生成 EMF 文件：{path}")
    with path.open("rb") as handle:
        handle.seek(40)
        signature = handle.read(4)
    if signature != b" EMF":
        raise ConversionError(f"输出不是有效的 Windows EMF 文件：{path}")


def convert_pdf_to_eps(source: Path, destination: Path) -> str:
    """Convert a PDF to EPS using a vector-preserving command-line tool."""
    inkscape = command_path("inkscape", "/Applications/Inkscape.app/Contents/MacOS/inkscape")
    if inkscape:
        run([inkscape, str(source), "--export-type=eps", f"--export-filename={destination}"])
        assert_eps(destination)
        return "Inkscape"

    pdftops = command_path("pdftops")
    if pdftops:
        run([pdftops, "-eps", str(source), str(destination)])
        assert_eps(destination)
        return "Poppler pdftops"

    ghostscript = command_path("gs", "ghostscript")
    if ghostscript:
        run(
            [
                ghostscript,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=eps2write",
                f"-sOutputFile={destination}",
                str(source),
            ]
        )
        assert_eps(destination)
        return "Ghostscript"

    raise ConversionError(
        "无法转换 PDF：请安装 Inkscape、Poppler（pdftops）或 Ghostscript（gs）中的任一工具。"
    )


def convert_svg_to_eps(source: Path, destination: Path) -> str:
    """Convert an SVG directly to EPS where possible."""
    inkscape = command_path("inkscape", "/Applications/Inkscape.app/Contents/MacOS/inkscape")
    if inkscape:
        run([inkscape, str(source), "--export-type=eps", f"--export-filename={destination}"])
        assert_eps(destination)
        return "Inkscape"

    rsvg_convert = command_path("rsvg-convert")
    if rsvg_convert:
        run([rsvg_convert, "--format=eps", "--output", str(destination), str(source)])
        assert_eps(destination)
        return "librsvg"

    raise ConversionError("无法转换 SVG：请安装 Inkscape 或 librsvg（rsvg-convert）。")


def inkscape_path() -> str | None:
    return command_path("inkscape", "/Applications/Inkscape.app/Contents/MacOS/inkscape")


def convert_to_emf(source: Path, destination: Path) -> str:
    """Export a PDF, SVG, or PNG to a correctly bounded EMF."""
    inkscape = inkscape_path()
    if not inkscape:
        raise ConversionError("无法转换为 EMF：请安装 Inkscape。")
    # draw.io SVG exports commonly store text in foreignObject elements. EMF
    # cannot represent those elements faithfully through Inkscape, so render
    # only this unsupported SVG construct at 3× resolution before packaging it
    # in EMF. Ordinary SVG and PDF inputs remain fully vector.
    if source.suffix.lower() == ".svg" and "<foreignObject" in source.read_text(encoding="utf-8"):
        rsvg_convert = command_path("rsvg-convert")
        if not rsvg_convert:
            raise ConversionError("含 foreignObject 的 SVG 转 EMF 需要 librsvg（rsvg-convert）。")
        with tempfile.TemporaryDirectory(prefix="foreignobject_to_emf_") as temp_dir:
            png_path = Path(temp_dir) / f"{source.stem}.png"
            run([rsvg_convert, "--format=png", "--zoom=3", "--output", str(png_path), str(source)])
            run([inkscape, str(png_path), "--export-type=emf", f"--export-filename={destination}"])
        assert_emf(destination)
        return "librsvg（3×）→ Inkscape"
    run([inkscape, str(source), "--export-type=emf", f"--export-filename={destination}"])
    assert_emf(destination)
    return "Inkscape"


def split_plotly_arguments(html: str) -> list[str]:
    """Extract the last Plotly.newPlot call without using a fragile regex."""
    marker = "Plotly.newPlot("
    start = html.rfind(marker)
    if start < 0:
        raise ConversionError("HTML 中未找到 Plotly.newPlot，当前仅支持 Plotly 导出的 HTML 图。")
    arguments: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    for char in html[start + len(marker) :]:
        if quote:
            buffer.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\\\"`":
            quote = char
            buffer.append(char)
        elif char in "([{":
            depth += 1
            buffer.append(char)
        elif char in ")]}":
            if char == ")" and depth == 0:
                arguments.append("".join(buffer).strip())
                break
            depth -= 1
            buffer.append(char)
        elif char == "," and depth == 0:
            arguments.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    if len(arguments) < 3:
        raise ConversionError("无法解析 Plotly HTML 中的图形数据。")
    return arguments


def render_plotly_html_to_svg(source: Path, svg_path: Path) -> str:
    """Render a local Plotly HTML figure to SVG without launching a browser."""
    try:
        from plotly import io as pio
    except ModuleNotFoundError as exc:
        raise ConversionError(
            "缺少 Plotly/Kaleido。请使用：conda run -n py312 python convert_figures_to_eps.py ..."
        ) from exc
    try:
        arguments = split_plotly_arguments(source.read_text(encoding="utf-8"))
        layout = json.loads(arguments[2])
        figure_json = json.dumps({"data": json.loads(arguments[1]), "layout": layout}, ensure_ascii=False)
        figure = pio.from_json(figure_json)
        width, height = PAPER_HTML_RENDER_SIZES.get(
            source.name, (layout.get("width") or 1400, layout.get("height") or 900)
        )
        figure.write_image(str(svg_path), format="svg", width=width, height=height, scale=1, engine="kaleido")
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ConversionError(f"无法将 Plotly HTML 导出为 SVG：{exc}") from exc
    if not svg_path.exists() or svg_path.stat().st_size == 0:
        raise ConversionError("Kaleido 未生成 SVG 文件。")
    return "Plotly/Kaleido"


def render_plotly_html_to_png(source: Path, png_path: Path) -> str:
    """Render Plotly HTML at the manuscript's publication dimensions."""
    try:
        from plotly import io as pio
    except ModuleNotFoundError as exc:
        raise ConversionError(
            "缺少 Plotly/Kaleido。请使用：conda run -n py312 python convert_figures_to_eps.py ..."
        ) from exc
    try:
        arguments = split_plotly_arguments(source.read_text(encoding="utf-8"))
        layout = json.loads(arguments[2])
        figure = pio.from_json(
            json.dumps({"data": json.loads(arguments[1]), "layout": layout}, ensure_ascii=False)
        )
        width, height = PAPER_HTML_RENDER_SIZES.get(
            source.name, (layout.get("width") or 1400, layout.get("height") or 900)
        )
        figure.write_image(str(png_path), format="png", width=width, height=height, scale=1, engine="kaleido")
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ConversionError(f"无法将 Plotly HTML 导出为 PNG：{exc}") from exc
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise ConversionError("Kaleido 未生成 PNG 文件。")
    return "Plotly/Kaleido"


def convert_html_to_eps(source: Path, destination: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="html_to_eps_") as temp_dir:
        svg_path = Path(temp_dir) / f"{source.stem}.svg"
        renderer = render_plotly_html_to_svg(source, svg_path)
        converter = convert_svg_to_eps(svg_path, destination)
    return f"{renderer} → {converter}"


def convert_html_to_emf(source: Path, destination: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="html_to_emf_") as temp_dir:
        # Inkscape's EMF writer mis-renders several Plotly text backgrounds.
        # A high-resolution Plotly raster inside EMF keeps Word-compatible
        # appearance and preserves the intended publication dimensions.
        png_path = Path(temp_dir) / f"{source.stem}.png"
        renderer = render_plotly_html_to_png(source, png_path)
        converter = convert_to_emf(png_path, destination)
    return f"{renderer} → {converter}"


def convert_one(source: Path, destination: Path, output_format: str) -> ConversionResult:
    extension = source.suffix.lower()
    if output_format == "svg":
        if extension in {".html", ".htm"}:
            converter = render_plotly_html_to_svg(source, destination)
        elif extension == ".svg":
            shutil.copy2(source, destination)
            converter = "直接复制 SVG"
        else:
            raise ConversionError(f"SVG 输出仅支持 Plotly HTML 或 SVG 输入：{source}")
    elif output_format == "emf":
        if extension in {".pdf", ".svg"}:
            converter = convert_to_emf(source, destination)
        elif extension in {".html", ".htm"}:
            converter = convert_html_to_emf(source, destination)
        else:
            raise ConversionError(f"不支持的文件类型：{source}")
    elif extension == ".pdf":
        converter = convert_pdf_to_eps(source, destination)
    elif extension == ".svg":
        converter = convert_svg_to_eps(source, destination)
    elif extension in {".html", ".htm"}:
        converter = convert_html_to_eps(source, destination)
    else:
        raise ConversionError(f"不支持的文件类型：{source}")
    return ConversionResult(source, destination, converter)


def collect_inputs(paths: list[Path], recursive: bool, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                print(f"跳过不支持的文件：{path}", file=sys.stderr)
            continue
        if not path.is_dir():
            print(f"跳过不存在的路径：{path}", file=sys.stderr)
            continue
        iterator: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
        for candidate in iterator:
            candidate = candidate.resolve()
            if not candidate.is_file() or output_dir in candidate.parents:
                continue
            if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(candidate)
    return sorted(set(files))


def output_paths(sources: list[Path], output_dir: Path, output_format: str) -> dict[Path, Path]:
    """Avoid overwriting outputs when two source files share the same stem."""
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.stem] = counts.get(source.stem, 0) + 1
    return {
        source: output_dir / f"{source.stem if counts[source.stem] == 1 else source.name}.{output_format}"
        for source in sources
    }


def manuscript_html_figures(notebook: Path, output_dir: Path, output_format: str) -> dict[Path, Path]:
    """Locate the four HTML figures produced by v2.ipynb for the manuscript."""
    notebook = notebook.resolve()
    if not notebook.is_file():
        raise ConversionError(f"找不到 notebook：{notebook}")
    notebook_text = notebook.read_text(encoding="utf-8")
    html_dir = notebook.parent / "output" / "0623"
    selected: dict[Path, Path] = {}
    for html_name, output_stem in PAPER_HTML_FIGURES.items():
        if html_name not in notebook_text:
            raise ConversionError(f"{notebook.name} 中未找到论文图定义：{html_name}")
        html_path = html_dir / html_name
        if not html_path.is_file():
            raise ConversionError(f"notebook 对应的 HTML 图不存在：{html_path}")
        selected[html_path.resolve()] = output_dir / f"{output_stem}.{output_format}"
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量将 PDF、SVG、Plotly HTML 图转换为 EPS、EMF 或 SVG。")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[],
        help="待转换的文件或目录（默认：脚本所在 figures 目录；使用 --paper-html 时默认只处理四张论文 HTML 图）。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认：Papers/figures/eps、emf 或 svg）。",
    )
    parser.add_argument("--format", choices=("eps", "emf", "svg"), default="eps", help="输出格式（默认：eps）。")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归扫描子目录。")
    parser.add_argument(
        "--paper-html",
        action="store_true",
        help="从 v2.ipynb 对应的四张论文 HTML 图生成 bar/vs/hot/yxl 四个输出文件。",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="--paper-html 使用的 notebook 路径（默认：项目根目录/v2.ipynb）。",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件。")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待执行转换，不写入文件。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (args.output_dir or SCRIPT_DIR / args.format).resolve()
    input_paths = args.inputs or ([] if args.paper_html else [SCRIPT_DIR])
    sources = collect_inputs(input_paths, args.recursive, output_dir)
    paper_targets: dict[Path, Path] = {}
    if args.paper_html:
        try:
            paper_targets = manuscript_html_figures(args.notebook, output_dir, args.format)
        except ConversionError as exc:
            print(f"[失败] {exc}", file=sys.stderr)
            return 1
        sources = sorted(set(sources) | set(paper_targets))

    if not sources:
        print("未找到 PDF、SVG 或 HTML 图。")
        return 0

    destinations = output_paths(sources, output_dir, args.format)
    destinations.update(paper_targets)
    if args.dry_run:
        for source in sources:
            print(f"[DRY RUN] {source} -> {destinations[source]}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source in sources:
        destination = destinations[source]
        if destination.exists() and not args.overwrite:
            print(f"[跳过] {destination.name} 已存在；使用 --overwrite 可覆盖。")
            continue
        try:
            result = convert_one(source, destination, args.format)
            print(f"[完成] {result.source.name} -> {result.destination.name}（{result.converter}）")
        except ConversionError as exc:
            failures += 1
            print(f"[失败] {source.name}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
