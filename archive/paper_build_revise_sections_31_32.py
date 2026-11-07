from __future__ import annotations

import copy
import datetime as dt
import sys
import zipfile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def w(name: str) -> str:
    return f"{{{W_NS}}}{name}"


REVISIONS = {
    "3.1 总体框架": "3.1 舆情分析框架",
    "针对重点交通枢纽社交媒体数据中图文线索分散": "针对重点交通枢纽社交媒体数据中图文线索分散、表达口语化、攻略内容与服务问题交织，以及互动表现难以直接转化为治理判断等问题，本文构建从多模态用户生成内容到站区服务问题治理线索的分析框架，如图1所示。该框架以小红书公开图文帖子为经验材料，依次包括多模态数据汇集与追溯、受约束的场景重构与问题样本构建、主题发现与相对关注度分析、跨枢纽问题画像与治理任务映射四个阶段。各阶段均保留帖子原始标识及其处理结果的对应关系，以支持主题解释和后续现场核查。",
    "本文以帖子标题与正文、图片或视频、评论和互动数据构建多模态场景表征": "首先，汇集帖子标题、正文、标签、图片或视频、评论、发布时间和互动数据等信息，并以帖子为基本分析单元。其次，利用多模态大模型在固定提示词和JSON输出约束下生成帖子综合描述，补充原始短文本中缺失的站区、场景、服务环节和旅客体验信息；再由独立的BERT模型筛选问题导向候选文本。负向筛选仅用于构建问题线索语料，不直接判定问题严重程度、责任归属或治理优先级。",
    "在问题导向主题发现部分": "在主题发现阶段，利用Qwen3-Embedding-0.6B获得候选文本的语义表征，并结合K-means与c-TF-IDF识别问题主题、提取主题词和核验代表表达。同时，基于文本语义相似性构建帖子网络，并将互动信息用于个性化PageRank计算，形成主题的相对关注度。",
    "最终，框架从主题构成与代表表达": "最终，框架从主题语义与代表表达、站场内分布、相对关注度以及治理任务线索四个维度形成问题画像。对于不同主题，进一步区分站区直接服务问题、需多主体协同的问题和不宜直接归责于站区的外部运行问题，并据此提供现场核查线索，而非将社交媒体表达直接等同于问题发生率或服务质量。",
    "3.2 数据预处理": "3.2 多模态样本构建",
    "（1）多模态数据清洗": "（1）受约束的多模态场景重构",
    "小红书平台中的数据天然具有多模态属性": "小红书帖子通常同时包含标题、正文、标签、图片或视频、评论和互动信息。原始文本具有篇幅短、口语化和语境省略等特点，直接用于主题建模时，容易出现图文割裂、场景信息缺失，以及攻略、分享与服务问题混杂等情况。为此，本文不将多模态大模型作为开放式解释工具，而是将其用于受约束的场景重构：在保留原始帖及其标识的前提下，将分散线索整理为统一、可回查的帖子综合描述。",
    "已有研究表明，大语言模型能够在结构化提示约束下生成任务适配的文本": "已有研究表明，大语言模型能够在结构化提示约束下生成任务适配的文本、标签或解释字段，多模态大语言模型还可以将图像内容转化为信息较完整的文本描述，以提高图文数据的可计算性和可复核性。借鉴LLaVA、ShareGPT4V等研究中利用模型生成图文语义描述的思路，本文将多模态大语言模型定位为受业务规则约束的场景重构工具，而非开放式解释工具；模型输出不得超出原始帖子、图片和评论所能支持的内容[22]。",
    "本文将标题、正文、标签、互动指标、高互动评论和帖子图片共同输入": "本文将标题、正文、标签、帖子图片或视频帧，以及预先选取的热门评论共同输入Qwen3.6-27B多模态大语言模型，生成帖子层面的综合描述。点赞、收藏和评论等互动数值不输入模型，而是在后续相对关注度分析中单独使用。该描述用于补充原始短文本中缺失的空间、设施和服务环节信息，并将分散的图文与评论线索组织为统一文本表示。模型输出约束为固定JSON中的description字段；对解析结果进行格式校验，解析失败时回退保留原始文本，以保证后续分析与原始证据之间的关联。",
    "其中，表示标题、正文和标签等文本信息": "其中，表示标题、正文和标签等文本信息，表示图片或视频帧等视觉信息，表示预先选取的热门评论集合；发布时间等元数据仍与原始帖子关联保存，但不作为场景重构的事实依据。多模态场景重构过程可表示为：",
    "图2展示多模态帖文总结提示词的结构": "图2展示多模态帖文总结提示词的结构。该提示词由角色与任务目标、多模态上下文插槽、证据融合规则、摘要规范和输出格式五部分构成。上下文插槽接收最多4张帖子图片、标题、正文、标签，以及按点赞量排序预先选取的前15条热门评论。点赞量仅用于评论排序，数值本身不传入模型；发布时间和互动数据保留在原始记录中，供后续统计与相对关注度计算使用。",
    "证据融合按照": "证据融合按照“主体语义提取—视觉信息补充—评论观点聚合—证据约束与去噪”的顺序进行：模型以标题和正文确定叙述对象、事件与核心诉求，利用图片补充场景、设施、环境等可观察事实，并从热门评论中提取高频问题、共同体验和代表性分歧；同时删除表情、重复和无关信息，不生成输入中未出现的事实。互动数据不参与模型生成或事实判断。最终以description字段输出150字的单段落综合描述，内容依次涵盖对象或场景、主要问题或体验、核心观点以及必要的视觉或评论补充；原始帖、图片和评论仍保留用于后续主题解释与回查。",
    "（2）负向文本筛选": "（2）问题导向候选样本筛选",
    "情感倾向是识别服务问题强度和治理优先级的重要依据": "多模态综合描述生成后，样本中仍同时包含经验性攻略、中性说明与负向体验。例如，关于“如何换乘”“哪里停车”“怎么进站”的帖子可能只是出行攻略，而“导视不清”“接驳不便”“排队过长”等表达更可能包含服务问题线索。为避免将正向体验、普通攻略与问题表达混合建模，本文采用独立的文本情感分析模型筛选问题导向候选文本。需要说明的是，负向情感仅用于构建问题线索语料，并不直接判定问题是否实际发生、严重程度、责任归属或治理优先级。",
    "互联网舆情研究通常将情感视为主题演化和公众态度分析的重要维度": "情感分类以帖子综合描述作为主要输入。该描述已经关联标题、正文、图片和热门评论信息，但图片和评论均不单独构成样本，而是作为所属帖子的补充语义。本文采用独立的BERT情感分析模型进行判别，以降低攻略、分享和一般性说明进入问题主题建模的比例。情感分类结果仅用于候选样本筛选；主题命名、治理任务判断和责任边界仍须结合代表帖、站场分布与实际业务边界进行核验。",
    "本文的主题建模语料仅保留负向文本": "本文的主题建模语料仅保留负向文本，正向和中性文本不进入本轮问题主题建模。由此得到情感分析后的问题导向候选语料集合：",
    "在此基础上，本文形成面向主题发现的负向问题导向样本": "在此基础上，本文形成面向主题发现的负向问题导向候选样本。正向和中性样本可用于服务亮点、一般性说明或对照分析，但不作为本轮问题主题建模数据；中性文本中可能存在的具体服务障碍不据此自动排除其业务意义。",
    "通过上述处理，本文将社交媒体UGC从一般性体验表达进一步筛选为负向问题导向样本库": "通过上述处理，本文将社交媒体UGC从一般性体验表达进一步筛选为保留原始证据关联的负向问题导向候选样本库。该样本库汇集旅客在站区导航导视、交通接驳、候车商业、环境卫生和现场管理等方面的负向体验，为后续Qwen3-Embedding、K-means聚类和c-TF-IDF主题词提取提供输入依据。",
}


def paragraph_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def set_space(t: etree._Element, value: str) -> None:
    if value[:1].isspace() or value[-1:].isspace():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def revision_run(tag: str, text: str, ident: int, author: str, when: str, rpr: etree._Element | None) -> etree._Element:
    outer = etree.Element(w(tag))
    outer.set(w("id"), str(ident))
    outer.set(w("author"), author)
    outer.set(w("date"), when)
    run = etree.SubElement(outer, w("r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    text_el = etree.SubElement(run, w("delText") if tag == "del" else w("t"))
    text_el.text = text
    set_space(text_el, text)
    return outer


def next_id(root: etree._Element) -> int:
    ids = []
    for el in root.xpath(".//*[@w:id]", namespaces=NS):
        try:
            ids.append(int(el.get(w("id"))))
        except (TypeError, ValueError):
            pass
    return max(ids, default=0) + 1


def enable_tracking(settings: etree._Element) -> None:
    if settings.find("w:trackRevisions", namespaces=NS) is None:
        settings.insert(0, etree.Element(w("trackRevisions")))


def main(source: str, output: str) -> None:
    when = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    author = "Codex 审阅"
    with zipfile.ZipFile(source) as zin:
        document = etree.fromstring(zin.read("word/document.xml"))
        settings_name = "word/settings.xml"
        settings = etree.fromstring(zin.read(settings_name))
        enable_tracking(settings)
        cid = next_id(document)
        remaining = dict(REVISIONS)
        changed = 0
        for p in document.xpath(".//w:body/w:p", namespaces=NS):
            current = paragraph_text(p)
            matched = next((old for old in remaining if current.startswith(old)), None)
            if matched is None:
                continue
            new = remaining.pop(matched)
            first_run = p.find("w:r", namespaces=NS)
            rpr = first_run.find("w:rPr", namespaces=NS) if first_run is not None else None
            ppr = p.find("w:pPr", namespaces=NS)
            for child in list(p):
                if child is not ppr:
                    p.remove(child)
            p.append(revision_run("del", current, cid, author, when, rpr))
            cid += 1
            p.append(revision_run("ins", new, cid, author, when, rpr))
            cid += 1
            changed += 1
        if remaining:
            raise RuntimeError(f"Unmatched revision anchors: {list(remaining)}")
        overrides = {
            "word/document.xml": etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True),
            settings_name: etree.tostring(settings, xml_declaration=True, encoding="UTF-8", standalone=True),
        }
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                zout.writestr(info, overrides.get(info.filename, zin.read(info.filename)))
    print(f"[OK] wrote {output} (tracked paragraph revisions={changed})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: revise_sections_31_32.py INPUT.docx OUTPUT.docx")
    main(sys.argv[1], sys.argv[2])
