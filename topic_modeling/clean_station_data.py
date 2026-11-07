"""
Clean station opinion data from Xiaohongshu Excel exports.

Jupyter example
---------------
import sys
sys.path.append("/Users/clsljt/Documents/hhgh/station/scripts")

from clean_station_data import clean_station_data, sample_valid_texts

df = clean_station_data(
    files=["/Users/clsljt/Documents/hhgh/station/data/北京西站_comments_合并.xlsx"]
)
df.head()

CLI example
-----------
python clean_station_data.py --files /Users/clsljt/Documents/hhgh/station/data/北京西站_comments_合并.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_STATION_DIR = Path("/Users/clsljt/Documents/hhgh/station")
DEFAULT_DATA_DIR = DEFAULT_STATION_DIR / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_STATION_DIR / "outputs"


STANDARD_COLUMNS = [
    "station",
    "source_type",
    "note_id",
    "comment_id",
    "root_comment_id",
    "parent_comment_id",
    "comment_level",
    "raw_text",
    "note_title",
    "like_count",
    "favorite_count",
    "comment_count",
    "share_count",
    "created_at",
    "ip_location",
    "url",
    "nickname",
    "source_file",
]


ISSUE_KEYWORDS: dict[str, list[str]] = {
    "异味/臭味": ["臭", "异味", "味道", "味儿", "污水", "厕所味"],
    "漏水/积水": ["漏水", "积水", "渗水", "水管", "水渍"],
    "卫生/环境": ["脏", "脏乱", "垃圾", "卫生", "保洁", "厕所"],
    "电梯/无障碍": ["电梯", "扶梯", "无障碍", "老人", "行李", "楼梯"],
    "停车/接送": ["停车", "车库", "接人", "网约车", "出租车", "自驾", "停车场", "打车"],
    "拥挤/排队": ["人多", "拥挤", "排队", "堵", "堵车", "人山人海", "滞留"],
    "指引/动线": ["指示", "指引", "标识", "出口", "入口", "迷路", "绕", "找不到"],
    "候车/商业": ["候车", "座位", "餐饮", "便利店", "商务候车室", "存包", "寄存"],
    "运行/晚点": ["晚点", "限速", "延误", "停运", "通报", "高铁晚点"],
}


RESPONSIBILITY_RULES: dict[str, dict[str, object]] = {
    "环境卫生": {
        "keywords": ["臭", "异味", "味道", "味儿", "污水", "脏", "垃圾", "卫生", "尿骚味", "厕所味"],
        "owner": "环境卫生/物业运维",
        "action": "排查异味、保洁、通风和垃圾清运薄弱点，形成点位巡检清单。",
    },
    "漏水积水": {
        "keywords": ["漏水", "积水", "渗水", "水管", "下雨", "水渍"],
        "owner": "设施设备/物业运维",
        "action": "核查漏水积水点位，区分临修、排水、结构维修并跟踪闭环。",
    },
    "电梯无障碍": {
        "keywords": ["电梯", "扶梯", "直梯", "无障碍", "楼梯", "老人", "轮椅", "行李", "拎箱子"],
        "owner": "设施设备/客运服务",
        "action": "核查扶梯直梯开放、故障、导引和大件行李通行体验。",
    },
    "进出站动线": {
        "keywords": ["进站", "出站", "检票", "安检", "入口", "出口", "候车室", "南广场", "北广场", "地铁", "换乘", "闸机"],
        "owner": "客运组织/动线管理",
        "action": "复核高频路径的导流、检票安检排队和换乘衔接。",
    },
    "标识指引": {
        "keywords": ["指示", "指引", "标识", "标志", "牌子", "找不到", "迷路", "绕", "方向", "怎么走", "不清楚"],
        "owner": "导视系统/客运服务",
        "action": "补强易迷路点位的标识、地贴、电子屏和人工引导。",
    },
    "接送停车": {
        "keywords": ["停车", "车库", "停车场", "接人", "接站", "网约车", "出租车", "打车", "自驾", "落客"],
        "owner": "交通接驳/停车管理",
        "action": "优化网约车、出租车、自驾接送点位和停车场秩序提示。",
    },
    "拥挤排队": {
        "keywords": ["人多", "拥挤", "排队", "堵", "堵车", "人山人海", "滞留", "乱哄哄", "挤"],
        "owner": "客流组织/现场秩序",
        "action": "识别高峰拥堵环节，调整开口、排队蛇形线和现场疏导。",
    },
    "候车配套": {
        "keywords": ["候车", "座位", "充电", "餐饮", "便利店", "存包", "寄存", "休息", "过夜", "空调", "冷"],
        "owner": "候车服务/商业配套",
        "action": "评估候车座椅、充电、夜间服务、餐饮和寄存供给。",
    },
    "商业价格": {
        "keywords": ["贵", "价格", "收费", "翻倍", "离谱", "宰", "租金", "消费", "外卖"],
        "owner": "商业管理/价格监管协同",
        "action": "梳理价格投诉点，核查明码标价和高频商品服务价格。",
    },
    "人员服务": {
        "keywords": ["工作人员", "服务态度", "态度不好", "没人管", "不让", "解释", "举报", "投诉", "12345"],
        "owner": "客运服务/投诉处置",
        "action": "复盘服务态度、解释口径和投诉闭环，补齐现场响应机制。",
    },
}


NON_RESPONSIBILITY_RULES: dict[str, list[str]] = {
    "城市摄影/建筑审美": ["摄影", "拍照", "机位", "长焦", "背景", "山脉", "风景", "建筑群", "地标", "层次感"],
    "旅游攻略/泛出行内容": ["北京旅游", "旅行", "一日游", "美食", "攻略", "指南", "打卡", "路线分享"],
    "站点比较/泛问答": ["北京站", "南站", "朝阳站", "丰台站", "离市区", "哪个近", "求解答", "请问"],
    "个人社交/闲聊": ["姐妹", "朋友", "偶遇", "收藏", "码住", "哈哈", "谢谢", "感谢", "蹲"],
    "铁路运行非站区管理": ["晚点", "延误", "停运", "限速", "列车晚点"],
}


STRONG_ACTION_SIGNAL_WORDS = [
    "臭",
    "脏",
    "漏水",
    "积水",
    "不开",
    "坏了",
    "损坏",
    "找不到",
    "迷路",
    "投诉",
    "举报",
    "12345",
    "不能",
    "不让",
    "没人",
    "态度不好",
    "可怕",
    "太难",
]


WEAK_ACTION_SIGNAL_WORDS = ["排队", "堵", "拥挤", "绕路", "等待", "等了", "很慢", "太慢", "慢死", "难打", "难找"]


POSITIVE_GUIDE_PATTERNS = [
    "不绕路",
    "不排队",
    "非常棒",
    "最丝滑",
    "很方便",
    "好方便",
    "人性化",
    "推荐",
    "顺利",
    "赶上了",
    "畅通无阻",
    "祝愿",
    "攻略",
    "指南",
    "路线",
]


CHATTER_PATTERNS = [
    r"^哈+$",
    r"^哈哈哈*$",
    r"^hhh+$",
    r"^第一个$",
    r"^我是第一个$",
    r"^蹲$",
    r"^码住$",
    r"^收藏$",
    r"^谢谢$",
    r"^感谢$",
    r"^聪明$",
    r"^好$",
    r"^嗯+$",
    r"^啊+$",
    r"^求解答$",
    r"^请问$",
]


STOPWORDS = {
    "北京西站",
    "北京朝阳站",
    "北京",
    "西站",
    "朝阳站",
    "车站",
    "高铁",
    "火车站",
    "小红书",
    "话题",
    "一个",
    "这个",
    "那个",
    "就是",
    "真的",
    "感觉",
    "可以",
    "没有",
    "不是",
    "还是",
    "然后",
    "大家",
    "内容",
    "帖子",
    "标题",
    "包含",
    "标签",
    "帖子内容",
    "包含标签",
    "无标题",
    "姐妹",
    "朋友",
    "攻略",
    "指南",
    "旅游",
    "旅行",
    "城市",
    "摄影",
    "背景",
    "建筑",
    "风景",
    "打卡",
    "请问",
    "求助",
    "求解答",
}


def infer_station(path: Path) -> str:
    file_name = path.name.lower()
    if "北京西站" in path.name or "xizhan" in file_name:
        return "北京西站"
    if "北京朝阳站" in path.name or "chaoyang" in file_name:
        return "北京朝阳站"
    return "未知车站"


def infer_source_type(path: Path) -> str:
    file_name = path.name.lower()
    if "comments" in file_name or "mments" in file_name:
        return "comment"
    if "notes" in file_name:
        return "note"
    return "unknown"


def get_first_available_column(raw: pd.DataFrame, *names: str, default: object = "") -> pd.Series:
    """Return the first matching column, supporting Chinese and English exports."""
    for name in names:
        if name in raw.columns:
            return raw[name]
    return pd.Series([default] * len(raw), index=raw.index)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_xhs_noise(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#([^#\[]+)(?:\[话题\])?#", r" \1 ", text)
    text = re.sub(r"\[[^\[\]]{1,12}\]", " ", text)
    text = re.sub(r"[#*_`~|<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_punctuation_only(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    useful = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text if useful else ""


def hit_issue_labels(text: str, issue_keywords: dict[str, list[str]] | None = None) -> list[str]:
    keywords = issue_keywords or ISSUE_KEYWORDS
    labels: list[str] = []
    for label, words in keywords.items():
        if any(word in text for word in words):
            labels.append(label)
    return labels


def hit_rule_labels(text: str, rules: dict[str, object]) -> list[str]:
    labels: list[str] = []
    for label, value in rules.items():
        words = value.get("keywords", []) if isinstance(value, dict) else value
        if any(str(word) in text for word in words):
            labels.append(label)
    return labels


def hit_responsibility_labels(text: str) -> list[str]:
    return hit_rule_labels(text, RESPONSIBILITY_RULES)


def hit_non_responsibility_labels(text: str) -> list[str]:
    return hit_rule_labels(text, NON_RESPONSIBILITY_RULES)


def has_action_signal(text: str) -> bool:
    positive_context = any(pattern in text for pattern in POSITIVE_GUIDE_PATTERNS)
    strong_hit = any(word in text for word in STRONG_ACTION_SIGNAL_WORDS)
    price_complaint = bool(re.search(r"(贵|价格|收费).{0,8}离谱|离谱.{0,8}(贵|价格|收费)", text))
    service_complaint = bool(re.search(r"(服务|态度|体验|卫生).{0,4}差|太差|脏乱差|乱哄哄|秩序乱|很乱", text))
    price_hit = bool(re.search(r"(太贵|很贵|价格高|价格贵|贵一倍|贵太多|收费高|价格翻倍)", text))
    if positive_context and not strong_hit and not price_complaint and not service_complaint and not price_hit:
        return False
    if strong_hit or price_complaint or service_complaint or price_hit:
        return True
    return any(word in text for word in WEAK_ACTION_SIGNAL_WORDS)


def choose_primary_issue(labels: list[str]) -> str:
    return labels[0] if labels else "未归类"


def classify_relevance(text: str, responsibility_labels: list[str], non_responsibility_labels: list[str]) -> tuple[bool, str]:
    if responsibility_labels:
        if has_action_signal(text):
            return True, "职责相关，含明确问题信号"
        if non_responsibility_labels:
            return True, "职责相关，混合攻略/泛内容，保留供主题分析"
        return True, "职责相关，宽松保留供主题分析"
    if non_responsibility_labels and not responsibility_labels:
        return False, "非管委会职责或无整改指向"
    return False, "未命中职责问题"


def estimate_issue_severity(text: str, labels: list[str], like_count: object = 0) -> str:
    score = 0
    high_labels = {"环境卫生", "漏水积水", "电梯无障碍", "拥挤排队", "人员服务"}
    if any(label in high_labels for label in labels):
        score += 2
    if has_action_signal(text):
        score += 1
    if any(word in text for word in ["投诉", "举报", "12345", "可怕", "直冲天灵盖", "影响形象"]):
        score += 2
    try:
        if int(like_count) >= 20:
            score += 1
    except (TypeError, ValueError):
        pass
    if score >= 4:
        return "高"
    if score >= 2:
        return "中"
    return "低"


def suggest_rectification_action(labels: list[str]) -> str:
    actions = []
    for label in labels:
        rule = RESPONSIBILITY_RULES.get(label)
        if rule:
            actions.append(str(rule["action"]))
    return "；".join(dict.fromkeys(actions)) if actions else "建议人工复核后归入具体整改事项。"


def suggest_responsible_owner(labels: list[str]) -> str:
    owners = []
    for label in labels:
        rule = RESPONSIBILITY_RULES.get(label)
        if rule:
            owners.append(str(rule["owner"]))
    return "；".join(dict.fromkeys(owners)) if owners else "待人工判定"


def is_chatter(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return True
    for pattern in CHATTER_PATTERNS:
        if re.fullmatch(pattern, compact):
            return True
    return False


def is_guide_content(text: str) -> bool:
    guide_words = [
        "攻略",
        "指南",
        "路线",
        "怎么走",
        "快速进站",
        "免安检",
        "接人",
        "接站",
        "打车点",
        "停车指南",
        "寄存",
        "存包",
    ]
    return any(word in text for word in guide_words)


def build_model_text(clean_text: str) -> str:
    text = clean_text
    for word in STOPWORDS:
        text = text.replace(word, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_responsibility_fields(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Add management-relevance and rectification helper fields."""
    annotated = df.copy()
    texts = annotated[text_col].fillna("").astype(str)
    annotated["responsibility_labels"] = texts.map(hit_responsibility_labels)
    annotated["responsibility_label_text"] = annotated["responsibility_labels"].map(lambda labels: "|".join(labels))
    annotated["non_responsibility_labels"] = texts.map(hit_non_responsibility_labels)
    annotated["non_responsibility_label_text"] = annotated["non_responsibility_labels"].map(lambda labels: "|".join(labels))

    relevance = [
        classify_relevance(text, resp, non_resp)
        for text, resp, non_resp in zip(
            texts,
            annotated["responsibility_labels"],
            annotated["non_responsibility_labels"],
        )
    ]
    annotated["is_responsibility_relevant"] = [item[0] for item in relevance]
    annotated["responsibility_relevance_reason"] = [item[1] for item in relevance]
    annotated["primary_issue_category"] = annotated["responsibility_labels"].map(choose_primary_issue)
    annotated["responsible_owner"] = annotated["responsibility_labels"].map(suggest_responsible_owner)
    annotated["suggested_action"] = annotated["responsibility_labels"].map(suggest_rectification_action)
    annotated["has_problem_signal"] = texts.map(has_action_signal)

    if "like_count" in annotated.columns:
        likes = annotated["like_count"]
    else:
        likes = pd.Series([0] * len(annotated), index=annotated.index)
    annotated["issue_severity"] = [
        estimate_issue_severity(text, labels, like)
        for text, labels, like in zip(texts, annotated["responsibility_labels"], likes)
    ]
    annotated["rectification_ready"] = (
        annotated["is_responsibility_relevant"]
        & annotated["has_problem_signal"]
        & annotated["primary_issue_category"].ne("未归类")
        & texts.str.len().ge(6)
    )
    return annotated


def resolve_excel_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    files: str | Path | Iterable[str | Path] | None = None,
) -> list[Path]:
    """Resolve explicit Excel files, or all .xlsx files under data_dir."""
    data_dir = Path(data_dir)
    if files is None:
        return sorted(data_dir.glob("*.xlsx"))

    if isinstance(files, (str, Path)):
        files = [files]

    resolved: list[Path] = []
    for file_path in files:
        path = Path(file_path)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        if path.suffix.lower() != ".xlsx":
            raise ValueError(f"Only .xlsx files are supported: {path}")
        resolved.append(path)
    return resolved


def make_output_paths(
    output_dir: str | Path,
    files: list[Path],
    output_prefix: str | None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if output_prefix is None:
        if len(files) == 1:
            output_prefix = files[0].stem
        else:
            output_prefix = "selected_station_texts"

    return {
        "cleaned": output_dir / f"{output_prefix}_cleaned.csv",
        "profile": output_dir / f"{output_prefix}_profile.csv",
        "issues": output_dir / f"{output_prefix}_issue_keyword_counts.csv",
        "rectification": output_dir / f"{output_prefix}_rectification_ledger.csv",
    }


def load_station_excels(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    files: str | Path | Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Load selected station Excel files and return one standardized DataFrame."""
    frames: list[pd.DataFrame] = []
    excel_files = resolve_excel_files(data_dir=data_dir, files=files)

    for path in excel_files:
        source_type = infer_source_type(path)
        station = infer_station(path)
        raw = pd.read_excel(path)

        if source_type == "comment":
            text = get_first_available_column(raw, "评论内容", "content").map(normalize_text)
            out = pd.DataFrame(
                {
                    "station": station,
                    "source_type": source_type,
                    "note_id": get_first_available_column(raw, "笔记id", "note_id", default=None),
                    "comment_id": get_first_available_column(raw, "评论id", "comment_id", default=None),
                    "root_comment_id": get_first_available_column(raw, "根评论id", "root_comment_id", default=None),
                    "parent_comment_id": get_first_available_column(raw, "父评论id", "parent_comment_id", default=None),
                    "comment_level": get_first_available_column(raw, "评论层级", "comment_level", default=None),
                    "raw_text": text,
                    "like_count": get_first_available_column(raw, "点赞数量", "like_count", default=0),
                    "created_at": get_first_available_column(raw, "上传时间", "upload_time", default=None),
                    "ip_location": get_first_available_column(raw, "ip归属地", "ip_location", default=None),
                    "url": get_first_available_column(raw, "笔记url", "note_url", default=None),
                    "nickname": get_first_available_column(raw, "昵称", "nickname", default=None),
                    "source_file": path.name,
                }
            )
        elif source_type == "note":
            title = get_first_available_column(raw, "标题", "title").map(normalize_text)
            desc = get_first_available_column(raw, "描述", "desc").map(normalize_text)
            text = (title + " " + desc).str.strip()
            out = pd.DataFrame(
                {
                    "station": station,
                    "source_type": source_type,
                    "note_id": get_first_available_column(raw, "笔记id", "note_id", default=None),
                    "comment_id": None,
                    "root_comment_id": None,
                    "parent_comment_id": None,
                    "comment_level": None,
                    "raw_text": text,
                    "note_title": title,
                    "like_count": get_first_available_column(raw, "点赞数量", "liked_count", default=0),
                    "favorite_count": get_first_available_column(raw, "收藏数量", "collected_count", default=None),
                    "comment_count": get_first_available_column(raw, "评论数量", "comment_count", default=None),
                    "share_count": get_first_available_column(raw, "分享数量", "share_count", default=None),
                    "created_at": get_first_available_column(raw, "上传时间", "upload_time", default=None),
                    "ip_location": get_first_available_column(raw, "ip归属地", "ip_location", default=None),
                    "url": get_first_available_column(raw, "笔记url", "note_url", default=None),
                    "nickname": get_first_available_column(raw, "昵称", "nickname", default=None),
                    "source_file": path.name,
                }
            )
        else:
            continue

        frames.append(out)

    if not frames:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[STANDARD_COLUMNS + [column for column in df.columns if column not in STANDARD_COLUMNS]]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    return df


def clean_dataframe(
    df: pd.DataFrame,
    min_len_without_issue: int = 6,
    keep_duplicates: bool = False,
) -> pd.DataFrame:
    """Clean standardized station data and add modeling helper columns."""
    cleaned = df.copy()
    cleaned["raw_text"] = cleaned["raw_text"].map(normalize_text)
    cleaned["clean_text"] = cleaned["raw_text"].map(strip_xhs_noise).map(remove_punctuation_only)
    cleaned["text_length"] = cleaned["clean_text"].str.len()
    cleaned["issue_labels"] = cleaned["clean_text"].map(hit_issue_labels)
    cleaned["issue_label_text"] = cleaned["issue_labels"].map(lambda labels: "|".join(labels))
    cleaned["has_issue_keyword"] = cleaned["issue_labels"].map(bool)
    cleaned = add_responsibility_fields(cleaned, text_col="clean_text")
    cleaned["is_chatter"] = cleaned["clean_text"].map(is_chatter)
    cleaned["is_guide_content"] = cleaned["clean_text"].map(is_guide_content)
    cleaned["model_text"] = cleaned["clean_text"].map(build_model_text)
    cleaned["model_text_length"] = cleaned["model_text"].str.len()

    cleaned["is_valid_text"] = True
    cleaned.loc[cleaned["clean_text"].eq(""), "is_valid_text"] = False
    cleaned.loc[cleaned["is_chatter"], "is_valid_text"] = False
    cleaned.loc[
        (cleaned["text_length"] < min_len_without_issue) & (~cleaned["has_issue_keyword"]),
        "is_valid_text",
    ] = False

    cleaned["duplicate_key"] = (
        cleaned["station"].fillna("")
        + "|"
        + cleaned["source_type"].fillna("")
        + "|"
        + cleaned["clean_text"].fillna("")
    )
    cleaned["is_duplicate_text"] = cleaned.duplicated("duplicate_key", keep="first")
    if not keep_duplicates:
        cleaned.loc[cleaned["is_duplicate_text"], "is_valid_text"] = False

    cleaned["source_weight"] = 1.0
    cleaned.loc[cleaned["source_type"].eq("note"), "source_weight"] = 0.8
    cleaned.loc[cleaned["comment_level"].astype(str).eq("2"), "source_weight"] = 0.75
    cleaned.loc[cleaned["is_guide_content"], "source_weight"] *= 0.85
    cleaned["like_weight"] = cleaned["like_count"].map(lambda value: math.log1p(max(value, 0)))
    cleaned["analysis_weight"] = cleaned["source_weight"] * (1.0 + cleaned["like_weight"] / 5.0)

    return cleaned


def annotate_text_id_dataframe(
    df: pd.DataFrame,
    text_col: str = "content",
    id_col: str = "id",
    like_col: str | None = None,
) -> pd.DataFrame:
    """Add responsibility and rectification fields to a simple id/text DataFrame."""
    missing = [column for column in [id_col, text_col] if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    annotated = df.copy()
    annotated["clean_text_for_relevance"] = (
        annotated[text_col].map(normalize_text).map(strip_xhs_noise).map(remove_punctuation_only)
    )
    if like_col and like_col in annotated.columns:
        annotated["like_count"] = pd.to_numeric(annotated[like_col], errors="coerce").fillna(0).astype(int)
    elif "like_count" not in annotated.columns:
        annotated["like_count"] = 0
    annotated = add_responsibility_fields(annotated, text_col="clean_text_for_relevance")
    return annotated


def clean_text_id_dataframe(
    df: pd.DataFrame,
    text_col: str = "content",
    id_col: str = "id",
    drop_empty: bool = True,
    drop_chatter: bool = True,
) -> pd.DataFrame:
    """
    Clean a simple DataFrame with id/text columns and return only id/text.

    This is the minimal Jupyter-friendly entry point for step-by-step work:

    cleaned_df = clean_text_id_dataframe(df)

    Parameters
    ----------
    df:
        Input DataFrame containing an id column and a text column.
    text_col:
        Name of the text column in the input DataFrame.
    id_col:
        Name of the id column in the input DataFrame.
    drop_empty:
        If True, remove rows whose cleaned text is empty. Defaults to True.
    drop_chatter:
        If True, remove obvious low-information chatter such as "哈哈哈",
        "谢谢", "蹲", and "码住". Defaults to True.
    """
    missing = [column for column in [id_col, text_col] if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    cleaned = pd.DataFrame(
        {
            "id": df[id_col],
            "content": df[text_col].map(normalize_text).map(strip_xhs_noise).map(remove_punctuation_only),
        }
    )
    if drop_chatter:
        cleaned = cleaned[~cleaned["content"].map(is_chatter)].copy()
    if drop_empty:
        cleaned = cleaned[cleaned["content"].ne("")].copy()
    return cleaned.reset_index(drop=True)


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data quality profile by station and source type."""
    rows = []
    for (station, source_type), group in df.groupby(["station", "source_type"], dropna=False):
        rows.append(
            {
                "station": station,
                "source_type": source_type,
                "rows": len(group),
                "valid_rows": int(group["is_valid_text"].sum()),
                "empty_clean_text": int(group["clean_text"].eq("").sum()),
                "duplicates": int(group["is_duplicate_text"].sum()),
                "median_text_length": float(group["text_length"].median()),
                "mean_text_length": float(group["text_length"].mean()),
                "max_text_length": int(group["text_length"].max()),
                "like_sum": int(group["like_count"].sum()),
                "start_time": group["created_at"].min(),
                "end_time": group["created_at"].max(),
            }
        )
    return pd.DataFrame(rows)


def explode_issue_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count issue keyword hits by station and source type."""
    valid = df[df["is_valid_text"]].copy()
    exploded = valid.explode("issue_labels")
    exploded = exploded[exploded["issue_labels"].notna() & exploded["issue_labels"].ne("")]
    if exploded.empty:
        return pd.DataFrame(columns=["station", "source_type", "issue_label", "rows", "likes"])
    return (
        exploded.groupby(["station", "source_type", "issue_labels"], as_index=False)
        .agg(rows=("clean_text", "size"), likes=("like_count", "sum"))
        .rename(columns={"issue_labels": "issue_label"})
        .sort_values(["station", "rows"], ascending=[True, False])
    )


def build_rectification_ledger(df: pd.DataFrame, top_examples: int = 3) -> pd.DataFrame:
    """Aggregate responsibility-relevant texts into an action-oriented issue ledger."""
    if "rectification_ready" not in df.columns:
        fallback_text_col = "clean_text" if "clean_text" in df.columns else "content"
        df = add_responsibility_fields(df, text_col=fallback_text_col)

    data = df[df["rectification_ready"]].copy()
    evidence_col = "clean_text" if "clean_text" in data.columns else "clean_text_for_relevance"
    if evidence_col not in data.columns:
        evidence_col = "content"
    if "url" not in data.columns:
        data["url"] = ""
    if "text_length" not in data.columns:
        data["text_length"] = data[evidence_col].fillna("").astype(str).str.len()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "primary_issue_category",
                "evidence_count",
                "like_sum",
                "severity_mix",
                "responsible_owner",
                "suggested_action",
                "representative_evidence",
                "representative_urls",
            ]
        )

    def top_texts(group: pd.DataFrame) -> str:
        sorted_group = group.sort_values(["like_count", "text_length"], ascending=[False, False]).head(top_examples)
        return " || ".join(sorted_group[evidence_col].astype(str).tolist())

    def top_urls(group: pd.DataFrame) -> str:
        sorted_group = group.sort_values(["like_count", "text_length"], ascending=[False, False]).head(top_examples)
        urls = [str(url) for url in sorted_group["url"].fillna("").tolist() if str(url).strip()]
        return " || ".join(urls)

    ledger = (
        data.groupby("primary_issue_category", as_index=False)
        .agg(
            evidence_count=(evidence_col, "size"),
            like_sum=("like_count", "sum"),
            severity_mix=("issue_severity", lambda values: "|".join(values.value_counts().index.astype(str))),
            responsible_owner=("responsible_owner", lambda values: "|".join(values.value_counts().index.astype(str)[:2])),
            suggested_action=("suggested_action", lambda values: values.value_counts().index.astype(str)[0]),
            representative_evidence=(evidence_col, lambda _: ""),
            representative_urls=("url", lambda _: ""),
        )
        .sort_values(["evidence_count", "like_sum"], ascending=[False, False])
    )

    examples = {
        category: top_texts(group)
        for category, group in data.groupby("primary_issue_category", sort=False)
    }
    urls = {
        category: top_urls(group)
        for category, group in data.groupby("primary_issue_category", sort=False)
    }
    ledger["representative_evidence"] = ledger["primary_issue_category"].map(examples).fillna("")
    ledger["representative_urls"] = ledger["primary_issue_category"].map(urls).fillna("")
    return ledger


def clean_station_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    files: str | Path | Iterable[str | Path] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_prefix: str | None = None,
    min_len_without_issue: int = 6,
    keep_duplicates: bool = False,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Load, clean, optionally write outputs, and return the cleaned DataFrame."""
    excel_files = resolve_excel_files(data_dir=data_dir, files=files)
    raw = load_station_excels(data_dir=data_dir, files=excel_files)
    cleaned = clean_dataframe(
        raw,
        min_len_without_issue=min_len_without_issue,
        keep_duplicates=keep_duplicates,
    )

    if write_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if files is None and output_prefix is None:
            paths = {
                "cleaned": output_dir / "cleaned_station_texts.csv",
                "profile": output_dir / "data_profile.csv",
                "issues": output_dir / "issue_keyword_counts.csv",
                "rectification": output_dir / "rectification_ledger.csv",
            }
        else:
            paths = make_output_paths(output_dir, excel_files, output_prefix)

        cleaned.to_csv(paths["cleaned"], index=False, encoding="utf-8-sig")
        profile_dataframe(cleaned).to_csv(paths["profile"], index=False, encoding="utf-8-sig")
        explode_issue_counts(cleaned).to_csv(paths["issues"], index=False, encoding="utf-8-sig")
        build_rectification_ledger(cleaned).to_csv(paths["rectification"], index=False, encoding="utf-8-sig")

    return cleaned


def clean_station_file(
    file_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_prefix: str | None = None,
    min_len_without_issue: int = 6,
    keep_duplicates: bool = False,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for cleaning one Excel file."""
    return clean_station_data(
        files=[file_path],
        output_dir=output_dir,
        output_prefix=output_prefix,
        min_len_without_issue=min_len_without_issue,
        keep_duplicates=keep_duplicates,
        write_outputs=write_outputs,
    )


def sample_valid_texts(df: pd.DataFrame, n: int = 20, station: str | None = None) -> pd.DataFrame:
    """Return a readable sample for manual cleaning QA."""
    sample = df[df["is_valid_text"]].copy()
    if station:
        sample = sample[sample["station"].eq(station)]
    columns = [
        "station",
        "source_type",
        "clean_text",
        "issue_label_text",
        "like_count",
        "created_at",
        "url",
    ]
    return sample.sort_values("like_count", ascending=False)[columns].head(n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean station Xiaohongshu note/comment data.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific .xlsx files to clean. Relative paths are resolved under --data-dir.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--min-len-without-issue", type=int, default=6)
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = clean_station_data(
        data_dir=args.data_dir,
        files=args.files,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        min_len_without_issue=args.min_len_without_issue,
        keep_duplicates=args.keep_duplicates,
        write_outputs=True,
    )
    profile = profile_dataframe(cleaned)
    print("Cleaned rows:", len(cleaned))
    print("Valid rows:", int(cleaned["is_valid_text"].sum()))
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
