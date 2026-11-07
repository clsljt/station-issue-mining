from copy import deepcopy
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Inches


ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "Papers"
TEMPLATE = PAPERS / "多模态大模型驱动的重点场站社交媒体舆情分析.docx"
OUTPUT = PAPERS / "多模态大模型驱动的交通枢纽社交媒体舆情分析_完整稿（相关工作待补）.docx"
OUTPUT_MD = PAPERS / "多模态大模型驱动的交通枢纽社交媒体舆情分析_完整稿（相关工作待补）.md"


FONT_FALLBACKS = {
    # The supplied manuscript template names Windows-only Chinese fonts.
    # These macOS fonts are present in the rendering environment and keep
    # Chinese glyphs visible in both Word and LibreOffice exports.
    "宋体": "Songti SC",
    "新宋体": "Songti SC",
    "黑体": "Heiti SC",
    "楷体": "STSong",
}


def set_font(run, name="宋体", size=10.5, bold=None, italic=None, color=None):
    name = FONT_FALLBACKS.get(name, name)
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_width(table, widths_cm):
    table.autofit = False
    for row in table.rows:
        for cell, width in zip(row.cells, widths_cm):
            cell.width = Cm(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width / 2.54 * 1440)))
            tc_w.set(qn("w:type"), "dxa")


class Builder:
    def __init__(self, document):
        self.doc = document
        self.last_was_caption = False

    def paragraph(self, text="", kind="body", align=None, first_indent=True):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        if kind == "body":
            pf.first_line_indent = Pt(21)
            pf.line_spacing = 1.45
            pf.space_after = Pt(0)
            pf.space_before = Pt(0)
            set_font(p.add_run(text), "宋体", 10.5)
        elif kind == "major":
            pf.space_before = Pt(12)
            pf.space_after = Pt(6)
            pf.keep_with_next = True
            set_font(p.add_run(text), "黑体", 13.5, bold=True)
        elif kind == "minor":
            pf.space_before = Pt(8)
            pf.space_after = Pt(4)
            pf.keep_with_next = True
            set_font(p.add_run(text), "黑体", 12.5, bold=True)
        elif kind == "third":
            pf.space_before = Pt(6)
            pf.space_after = Pt(2)
            pf.keep_with_next = True
            set_font(p.add_run(text), "楷体", 11.5, bold=True)
        elif kind == "formula":
            pf.space_before = Pt(2)
            pf.space_after = Pt(2)
            pf.line_spacing = 1.1
            set_font(p.add_run(text), "Cambria Math", 10.5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "caption":
            pf.space_before = Pt(3)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.0
            set_font(p.add_run(text), "宋体", 9)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "note":
            pf.space_before = Pt(3)
            pf.space_after = Pt(6)
            pf.line_spacing = 1.2
            set_font(p.add_run(text), "宋体", 9, italic=True, color=(89, 89, 89))
        elif kind == "front_label":
            pf.first_line_indent = Pt(0)
            pf.line_spacing = 1.25
            pf.space_after = Pt(1)
            label, sep, body = text.partition("】")
            if sep:
                set_font(p.add_run(label + sep), "宋体", 10.5, bold=True)
                set_font(p.add_run(body), "宋体", 10.5)
            else:
                set_font(p.add_run(text), "宋体", 10.5)
        elif kind == "keyword":
            pf.space_before = Pt(2)
            pf.space_after = Pt(4)
            set_font(p.add_run(text), "宋体", 10.5)
        if align is not None:
            p.alignment = align
        return p

    def front_title(self, text, english=False):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_before = Pt(0 if not english else 12)
        pf.space_after = Pt(5)
        pf.line_spacing = 1.1
        set_font(p.add_run(text), "黑体" if not english else "Times New Roman", 16 if not english else 15, bold=True)
        return p

    def author(self, text, english=False):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        set_font(p.add_run(text), "宋体" if not english else "Times New Roman", 10)
        return p

    def figure(self, image_path, cn_caption, en_caption):
        path = Path(image_path)
        if not path.is_absolute():
            path = PAPERS / path
        if not path.exists():
            path = PAPERS / "figures" / path.name
        if not path.exists():
            self.paragraph(f"[图片缺失：{cn_caption}]", "note")
            return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.add_picture(str(path), width=Cm(15.1))
        self.paragraph(cn_caption, "caption")
        self.paragraph(en_caption, "caption")

    def table(self, rows):
        if not rows:
            return
        col_count = len(rows[0])
        table = self.doc.add_table(rows=len(rows), cols=col_count)
        # The journal template intentionally carries no built-in table style.
        # Apply borders directly instead of falling back to a foreign Word style.
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element = OxmlElement(f"w:{edge}")
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "808080")
            borders.append(element)
        tbl_pr.append(borders)
        maxlens = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(col_count)]
        total = sum(max(5, n) for n in maxlens)
        widths = [15.1 * max(5, n) / total for n in maxlens]
        widths = [max(1.3, w) for w in widths]
        scale = 15.1 / sum(widths)
        widths = [w * scale for w in widths]
        set_table_width(table, widths)
        for r_i, row in enumerate(rows):
            for c_i, value in enumerate(row):
                cell = table.cell(r_i, c_i)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                set_cell_margins(cell)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (r_i == 0 or len(value) < 15) else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.1
                run = p.add_run(value)
                set_font(run, "宋体", 8.5, bold=(r_i == 0))
                if r_i == 0:
                    set_cell_shading(cell, "E6E6E6")
        set_repeat_table_header(table.rows[0])
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)


def clean_latex(text):
    compact = re.sub(r"\s+", " ", text.strip())
    # Use legible Unicode equations rather than exposing LaTeX commands in a
    # Word document without a MathType/OMML dependency.
    equation_map = [
        ("X_i=", "Xᵢ = {Tᵢ, Iᵢ, Cᵢ, Mᵢ}"),
        ("f_{\\mathrm{MLLM}}", "(dᵢ, yᵢ) = f_MLLM(Xᵢ; P)"),
        ("h_i=", "hᵢ = BERT(xᵢ)[CLS]"),
        ("softmax", "[pᵢ⁻, pᵢ⁺] = softmax(W hᵢ + b)"),
        ("\\hat{s}_i", "ŝᵢ = negative if pᵢ⁻ ≥ θ₋; positive if pᵢ⁺ ≥ θ₊; otherwise neutral"),
        ("\\mathcal{D}", "D⁻ = { i | ŝᵢ ∈ {negative, neutral} }"),
        ("f_{\\mathrm{Qwen}}", "eᵢ = f_Qwen(xᵢ),  eᵢ ∈ Rᵈ"),
        ("\\min_", "min Σₖ Σᵢ:zᵢ₌ₖ ‖eᵢ − μₖ‖²"),
        ("w_{t,k}", "wₜ,ₖ = tfₜ,ₖ · log(1 + K*/dfₜ)"),
        ("NPMI", "NPMI(wᵢ, wⱼ) = log[P(wᵢ,wⱼ)/(P(wᵢ)P(wⱼ))] / −log P(wᵢ,wⱼ)"),
        ("operatorname{TD}", "TD = |⋃ₖ Wₖ| / Σₖ |Wₖ|"),
        ("s_{ij}", "sᵢⱼ = eᵢᵀ eⱼ"),
        ("m_i=", "mᵢ = Lᵢ + Cᵢ + Rᵢ + Sᵢ"),
        ("p_i=", "pᵢ = log(1 + mᵢ) / Σⱼ log(1 + mⱼ)"),
        ("\\mathbf{r}", "r = α WᵀD⁻¹r + (1 − α)p"),
        ("I_i=", "Iᵢ = 10 000 × rᵢ"),
        ("I_k", "Iₖᴷᴹ = Σᵢ:zᵢ₌ₖ Iᵢᴷᴹ"),
    ]
    for marker, replacement in equation_map:
        if marker in compact:
            return replacement
    text = text.strip()
    text = re.sub(r"\\operatorname\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\1", text)
    text = text.replace("\\cdot", "·").replace("\\top", "ᵀ")
    text = text.replace("\\sum", "Σ").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\frac", "frac")
    text = text.replace("\\log", "log").replace("\\alpha", "α").replace("\\mu", "μ")
    text = text.replace("\\|", "|").replace("\\_", "_")
    text = re.sub(r"\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\\([A-Za-z]+)", r"\1", text)


def clean_inline_latex(text):
    """Make inline LaTeX readable in a DOCX without a MathType dependency."""
    replacements = {
        r"\(": "", r"\)": "", r"\[": "", r"\]": "",
        r"\mathbf": "", r"\boldsymbol": "", r"\mathcal": "",
        r"\mathrm": "", r"\mathbb": "", r"\operatorname": "",
        r"\text": "", r"\quad": " ", r"\cdot": "·",
        r"\top": "ᵀ", r"\alpha": "α", r"\beta": "β",
        r"\theta": "θ", r"\mu": "μ", r"\geq": "≥",
        r"\leq": "≤", r"\in": "∈", r"\mid": "|",
        r"\sum": "Σ", r"\log": "log", r"\left": "", r"\right": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\\([A-Za-z]+)", r"\1", text)


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


FIG_CAPTIONS = {
    "图1　基于多模态公众感知的站区服务问题识别与影响分析框架": "Fig. 1 Framework for identifying and analyzing station-area service issues from multimodal public perception data",
    "图 4-1　负向舆情主题关键词权重": "Fig. 4-1 Keyword weights of negative-sentiment topics",
    "图 4-2　负向舆情主题的文本占比与影响占比": "Fig. 4-2 Text and influence shares of negative-sentiment topics",
    "图 4-3　负向舆情主题在各站场的标准化分布": "Fig. 4-3 Normalized distribution of negative-sentiment topics across hubs",
    "图 4-4　主题规模与综合影响力四象限": "Fig. 4-4 Topic size and composite influence quadrants",
}


def add_markdown(builder, text, skip_top_title=False):
    lines = text.splitlines()
    i = 0
    pending_caption = None
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if not (skip_top_title and level == 1):
                builder.paragraph(title, {1: "major", 2: "major", 3: "minor", 4: "third"}.get(level, "body"))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            builder.table(parse_table(block))
            continue
        image = re.match(r"!\[[^]]*\]\(([^)]+)\)", line.strip())
        if image:
            image_path = image.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption = lines[j].strip() if j < len(lines) and lines[j].strip().startswith("图") else "图　分析结果示意"
            en = FIG_CAPTIONS.get(caption, "Fig. Analytical result")
            builder.figure(image_path, caption, en)
            i = j + 1 if j < len(lines) and caption == lines[j].strip() else i + 1
            continue
        if line.strip() in ("$$", "\\["):
            end = "$$" if line.strip() == "$$" else "\\]"
            formula = []
            i += 1
            while i < len(lines) and lines[i].strip() != end:
                formula.append(lines[i].strip())
                i += 1
            builder.paragraph(clean_latex(" ".join(formula)), "formula")
            i += 1
            continue
        if line.strip().startswith("表 ") or line.strip().startswith("表　"):
            builder.paragraph(line.strip(), "caption")
            i += 1
            continue
        paragraph_lines = [line.strip()]
        i += 1
        while i < len(lines):
            candidate = lines[i].rstrip()
            if not candidate.strip() or re.match(r"^(#{1,4})\s+", candidate) or candidate.lstrip().startswith("|") or candidate.strip() in ("$$", "\\[") or re.match(r"!\[[^]]*\]\(([^)]+)\)", candidate.strip()):
                break
            paragraph_lines.append(candidate.strip())
            i += 1
        value = clean_inline_latex(" ".join(paragraph_lines))
        if value.startswith("图 ") or value.startswith("图　"):
            builder.paragraph(value, "caption")
        elif value.startswith("（") and "）" in value[:10] and len(value) < 55:
            builder.paragraph(value, "third")
        else:
            builder.paragraph(value, "body")


def extract_section(path, start, end):
    text = path.read_text(encoding="utf-8")
    begin = text.find(start)
    if begin < 0:
        raise ValueError(f"Cannot locate {start} in {path}")
    fragment = text[begin:]
    if end:
        stop = fragment.find(end)
        if stop >= 0:
            fragment = fragment[:stop]
    return fragment


def write_markdown():
    """Produce the editable master manuscript requested by the author."""
    intro = extract_section(PAPERS / "paper0.1.md", "## 1 引言", "## 2 相关工作")
    abstract = """## 摘要

【目的】针对重点交通枢纽旅客服务治理中社交媒体反馈识别滞后、问题归类粗放和跨枢纽比较不足等问题，构建基于多模态大模型与短文本主题建模的社交媒体舆情分析流程。

【应用背景】研究面向北京市火车站站区与机场场区的旅客服务评估、舆情监测和治理整改场景，以小红书平台中与北京“八站两场”相关的图文帖子、评论、图片及互动指标为对象。

【方法】以 note_id 为追溯主键，整合帖子标题、正文、图片、标签、评论和互动数据；利用 Qwen3.6-27B 解析图文 UGC 并完成业务相关性分类，采用 BERT 情感识别筛选负向体验，再以 Qwen3-Embedding-0.6B、UMAP、KMeans 和 c-TF-IDF 实现 BERTopic 主题发现；结合语义网络 PageRank 与互动信息计算综合影响力，形成站场问题画像和关注优先级。

【结果】在 1 208 条帖子和 54 357 条评论构成的样本中，466 条帖子被识别为业务类，293 条负向帖子用于主题实验。主方法获得 0.136 的轮廓系数、11.532 的 CHI 和 0.089 的 NPMI，并识别出候车与进站、出站换乘、运行延误、航站楼服务和网约车接驳等 10 个主题。

【局限】样本主要来自单一社交平台，用户结构和表达习惯可能影响结论代表性；模型分类与情感识别仍需人工抽样复核；社交媒体中的表达分布不能直接等同于问题发生频率或服务质量。

【结论】该流程可将非结构化、多模态、短文本 UGC 转化为可回溯、可比较、可解释的站场服务问题证据，并为旅客服务效能提升提供排序依据。

**关键词：** 交通枢纽；旅客服务；社交媒体舆情；多模态大模型；短文本主题建模；BERTopic

## Abstract

**Objective:** To address delayed identification, coarse categorization, and limited cross-hub comparison of social-media feedback in passenger-service governance, this study develops an analysis workflow based on multimodal large models and short-text topic modeling.

**Application Context:** The workflow is applied to passenger-service assessment, public-opinion monitoring, and corrective governance for Beijing's major railway stations and airports, using Xiaohongshu posts, comments, images, and engagement indicators.

**Methods:** Posts are organized with note_id as the traceability key. Qwen3.6-27B parses multimodal UGC and filters business-relevant content; BERT identifies negative experiences; Qwen3-Embedding-0.6B, UMAP, KMeans, and c-TF-IDF support BERTopic topic discovery. A semantic-network PageRank score and engagement data are combined to rank topic influence.

**Results:** Among 1,208 posts and 54,357 comments, 466 posts are business-relevant and 293 negative posts are used for topic experiments. The proposed configuration achieves a silhouette score of 0.136, a CHI of 11.532, and an NPMI of 0.089, producing ten interpretable service topics.

**Limitations:** The sample is drawn mainly from one platform, model outputs require manual spot checks, and online expression patterns do not directly measure incident frequency or service quality.

**Conclusions:** The workflow transforms multimodal, unstructured, short UGC into traceable and comparable evidence for station-area service improvement.

**Keywords:** transportation hub; passenger service; social media sentiment; multimodal large model; short-text topic modeling; BERTopic
"""
    conclusion = """## 5 结语

### 5.1 主要结论

本文面向重点交通枢纽旅客服务治理，构建了由多模态语义清洗、情感识别、短文本主题发现和综合影响力计算组成的社交媒体舆情分析流程。以 note_id 贯通原始帖文、图片、评论、模型输出和互动指标，使主题解释能够回溯至具体表达。应用结果表明，图文综合描述能够补充短文本中的空间场景和服务环节信息；Qwen3-Embedding-0.6B 与 KMeans 的组合在本研究样本上兼顾了聚类结构、主题词一致性和主题多样性。

### 5.2 治理启示

主题的文本规模、站场内集中程度与传播影响并不完全一致。候车与进站体验、丰台站出站换乘动线、机场运行延误与旅客滞留、航站楼服务及网约车接驳等主题在样本中呈现较高的关注度，可作为现场核查和联动整改的优先线索；站台吸烟等低频但高互动主题，则适合纳入事件驱动的重点监测。分析结果应与客流、投诉、巡查和运营数据共同使用，而不宜单独作为服务质量评价或责任认定依据。

### 5.3 局限与未来工作

本研究的样本主要来自小红书，平台用户结构、内容推荐机制和站场曝光度会影响表达分布；不同枢纽的服务流程和空间组织差异也限制了直接横向比较。后续可扩展微博、短视频和热线等多源数据，建立分层抽样复核与模型漂移监测机制，并结合客流、时段、天气和突发事件信息开展时序分析，以提升主题解释的稳定性和治理建议的针对性。

> 基金项目、作者简介、通讯作者信息待补充。
"""
    master = "# 多模态大模型驱动的交通枢纽社交媒体舆情分析\n\n"
    master += "**作者：** 待补充  \n**单位：** 待补充\n\n"
    master += abstract + "\n\n"
    master += intro.strip() + "\n\n"
    master += "## 2 相关工作\n\n> 本节按作者要求暂不纳入本稿，待后续补充。\n\n"
    master += (PAPERS / "paper_chapter3.1.md").read_text(encoding="utf-8").strip() + "\n\n"
    chapter4 = (PAPERS / "chapter4.1.md").read_text(encoding="utf-8").strip()
    chapter4 = chapter4.replace("# 4 应用效果评估与实验分析", "## 4 应用效果评估与实验分析", 1)
    chapter4 = chapter4.replace("\n## 4.", "\n### 4.")
    master += chapter4 + "\n\n"
    chapter45 = (PAPERS / "paper_chapter4.5.md").read_text(encoding="utf-8").strip()
    chapter45 = chapter45.replace("## 4.5", "### 4.5", 1).replace("\n### 4.5.", "\n#### 4.5.")
    master += chapter45 + "\n\n"
    master += conclusion
    OUTPUT_MD.write_text(master, encoding="utf-8")
    print(OUTPUT_MD)


def main():
    doc = Document(TEMPLATE)
    # Clear template example content but retain document-level page and style settings.
    doc._body.clear_content()
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.175)
    section.right_margin = Cm(3.175)

    builder = Builder(doc)

    # Front matter
    builder.front_title("多模态大模型驱动的交通枢纽社交媒体舆情分析")
    builder.author("作者姓名待补充")
    builder.author("（单位、城市、邮编待补充）")
    builder.paragraph("摘要：", "front_label")
    abstract = [
        ("【目的】", "针对重点交通枢纽旅客服务治理中社交媒体反馈识别滞后、问题归类粗放和跨枢纽比较不足等问题，构建基于多模态大模型与短文本主题建模的社交媒体舆情分析流程。"),
        ("【应用背景】", "研究面向北京市火车站站区与机场场区的旅客服务评估、舆情监测和治理整改场景，以小红书平台中与北京“八站两场”相关的图文帖子、评论、图片及互动指标为对象。"),
        ("【方法】", "以 note_id 为追溯主键，整合帖子标题、正文、图片、标签、评论和互动数据；利用 Qwen3.6-27B 解析图文 UGC 并完成业务相关性分类，采用 BERT 情感识别筛选负向体验，再以 Qwen3-Embedding-0.6B、UMAP、KMeans 和 c-TF-IDF 实现 BERTopic 主题发现；结合语义网络 PageRank 与互动信息计算综合影响力，形成站场问题画像和关注优先级。"),
        ("【结果】", "在 1 208 条帖子和 54 357 条评论构成的样本中，466 条帖子被识别为业务类，293 条负向帖子用于主题实验。主方法获得 0.136 的轮廓系数、11.532 的 CHI 和 0.089 的 NPMI，并识别出候车与进站、出站换乘、运行延误、航站楼服务和网约车接驳等 10 个主题。"),
        ("【局限】", "样本主要来自单一社交平台，用户结构和表达习惯可能影响结论代表性；模型分类与情感识别仍需人工抽样复核；社交媒体中的表达分布不能直接等同于问题发生频率或服务质量。"),
        ("【结论】", "该流程可将非结构化、多模态、短文本 UGC 转化为可回溯、可比较、可解释的站场服务问题证据，并为旅客服务效能提升提供排序依据。"),
    ]
    for label, body in abstract:
        builder.paragraph(label + body, "front_label")
    builder.paragraph("关键词：交通枢纽；旅客服务；社交媒体舆情；多模态大模型；短文本主题建模；BERTopic", "keyword")

    builder.front_title("Social Media Sentiment Analysis for Transportation Hubs Driven by Multimodal Large Models", english=True)
    builder.author("Author Name to Be Added")
    builder.author("(Affiliation, City, Postal Code to Be Added)")
    builder.paragraph("Abstract：", "front_label")
    english_abstract = [
        ("【Objective】", "To address delayed identification, coarse categorization, and limited cross-hub comparison of social-media feedback in passenger-service governance, this study develops an analysis workflow based on multimodal large models and short-text topic modeling."),
        ("【Application Context】", "The workflow is applied to passenger-service assessment, public-opinion monitoring, and corrective governance for Beijing's major railway stations and airports, using Xiaohongshu posts, comments, images, and engagement indicators."),
        ("【Methods】", "Posts are organized with note_id as the traceability key. Qwen3.6-27B parses multimodal UGC and filters business-relevant content; BERT identifies negative experiences; Qwen3-Embedding-0.6B, UMAP, KMeans, and c-TF-IDF support BERTopic topic discovery. A semantic-network PageRank score and engagement data are combined to rank topic influence."),
        ("【Results】", "Among 1,208 posts and 54,357 comments, 466 posts are business-relevant and 293 negative posts are used for topic experiments. The proposed configuration achieves a silhouette score of 0.136, a CHI of 11.532, and an NPMI of 0.089, producing ten interpretable service topics."),
        ("【Limitations】", "The sample is drawn mainly from one platform, model outputs require manual spot checks, and online expression patterns do not directly measure incident frequency or service quality."),
        ("【Conclusions】", "The workflow transforms multimodal, unstructured, short UGC into traceable and comparable evidence for station-area service improvement."),
    ]
    for label, body in english_abstract:
        builder.paragraph(label + body, "front_label")
    builder.paragraph("Keywords：transportation hub; passenger service; social media sentiment; multimodal large model; short-text topic modeling; BERTopic", "keyword")

    # Introduction from the user-provided draft, retaining the paper_Lili research theme.
    intro = extract_section(PAPERS / "paper0.1.md", "## 1 引言", "## 2 相关工作")
    add_markdown(builder, intro)

    # Related work intentionally omitted at the author's request.
    builder.paragraph("2 相关工作", "major")
    builder.paragraph("本节按作者要求暂不纳入本稿，待后续补充。", "note")

    add_markdown(builder, (PAPERS / "paper_chapter3.1.md").read_text(encoding="utf-8"))
    add_markdown(builder, (PAPERS / "chapter4.1.md").read_text(encoding="utf-8"))
    add_markdown(builder, (PAPERS / "paper_chapter4.5.md").read_text(encoding="utf-8"))

    # Conclusion and practical implication are synthesized from the provided framework and results.
    builder.paragraph("5 结语", "major")
    builder.paragraph("5.1 主要结论", "minor")
    builder.paragraph("本文面向重点交通枢纽旅客服务治理，构建了由多模态语义清洗、情感识别、短文本主题发现和综合影响力计算组成的社交媒体舆情分析流程。以 note_id 贯通原始帖文、图片、评论、模型输出和互动指标，使主题解释能够回溯至具体表达。应用结果表明，图文综合描述能够补充短文本中的空间场景和服务环节信息；Qwen3-Embedding-0.6B 与 KMeans 的组合在本研究样本上兼顾了聚类结构、主题词一致性和主题多样性。", "body")
    builder.paragraph("5.2 治理启示", "minor")
    builder.paragraph("主题的文本规模、站场内集中程度与传播影响并不完全一致。候车与进站体验、丰台站出站换乘动线、机场运行延误与旅客滞留、航站楼服务及网约车接驳等主题在样本中呈现较高的关注度，可作为现场核查和联动整改的优先线索；站台吸烟等低频但高互动主题，则适合纳入事件驱动的重点监测。分析结果应与客流、投诉、巡查和运营数据共同使用，而不宜单独作为服务质量评价或责任认定依据。", "body")
    builder.paragraph("5.3 局限与未来工作", "minor")
    builder.paragraph("本研究的样本主要来自小红书，平台用户结构、内容推荐机制和站场曝光度会影响表达分布；不同枢纽的服务流程和空间组织差异也限制了直接横向比较。后续可扩展微博、短视频和热线等多源数据，建立分层抽样复核与模型漂移监测机制，并结合客流、时段、天气和突发事件信息开展时序分析，以提升主题解释的稳定性和治理建议的针对性。", "body")

    builder.paragraph("基金项目、作者简介、通讯作者信息：待补充。", "note")
    doc.core_properties.title = "多模态大模型驱动的交通枢纽社交媒体舆情分析"
    doc.core_properties.subject = "基于多模态大模型与短文本主题建模的社交媒体舆情分析"
    doc.core_properties.author = "待补充"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    if "--markdown-only" in sys.argv:
        write_markdown()
    else:
        main()
