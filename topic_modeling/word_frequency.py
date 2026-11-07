"""
Word frequency analysis for cleaned and sentiment-filtered station comments.

Input: a DataFrame with at least id/text columns.
Output: word frequency DataFrame.

Jupyter example
---------------
import sys
sys.path.insert(0, "/Users/clsljt/Documents/hhgh/station/scripts")

from word_frequency import compute_word_frequency

freq_df = compute_word_frequency(filtered_df)
freq_df.head(30)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_STATION_DIR = Path("/Users/clsljt/Documents/hhgh/station")
DEFAULT_STOPWORDS_PATH = DEFAULT_STATION_DIR / "stopword.txt"
DEFAULT_STOPINTHIS_PATH = DEFAULT_STATION_DIR / "stopinthis.txt"
DEFAULT_FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")


EXTRA_STOPWORDS = {
    "北京",
    "北京西站",
    "北京朝阳站",
    "西站",
    "朝阳站",
    "车站",
    "火车站",
    "高铁",
    "火车",
    "小红书",
    "话题",
    "内容",
    "帖子",
    "标题",
    "包含",
    "标签",
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
    "一下",
    "现在",
    "今天",
    "知道",
    "发现",
    "地方",
    "应该",
    "已经",
    "帖子内容",
    "包含标签",
    "无标题",
    "谢谢",
    "感谢",
    "哈哈",
    "哈哈哈",
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
    "偶遇",
    "请问",
    "求助",
    "求解答",
    "北京站",
    "南站",
    "朝阳站",
    "丰台站",
    "大兴",
}


def load_stopwords(
    stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
    use_stopinthis: bool = True,
    stopinthis_path: str | Path = DEFAULT_STOPINTHIS_PATH,
    extra_stopwords: Iterable[str] | None = EXTRA_STOPWORDS,
) -> set[str]:
    """Load stopwords from base file plus optional project-specific file."""
    path = Path(stopwords_path)
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")

    stopwords = {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }
    if use_stopinthis:
        project_path = Path(stopinthis_path)
        if project_path.exists():
            stopwords.update(
                line.strip()
                for line in project_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            )
    if extra_stopwords:
        stopwords.update(word.strip() for word in extra_stopwords if str(word).strip())
    return stopwords


def tokenize_text(
    text: object,
    stopwords: set[str] | None = None,
    min_word_len: int = 2,
    keep_english: bool = True,
) -> list[str]:
    """Tokenize Chinese text with jieba and filter noisy tokens."""
    import jieba

    stopwords = stopwords or set()
    text = "" if pd.isna(text) else str(text)
    tokens = jieba.lcut(text)

    kept: list[str] = []
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        if token in stopwords:
            continue
        if len(token) < min_word_len:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
            continue
        if not keep_english and re.fullmatch(r"[a-zA-Z]+", token):
            continue
        kept.append(token)
    return kept


def compute_word_frequency(
    df: pd.DataFrame,
    text_col: str = "content",
    stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
    use_stopinthis: bool = True,
    stopinthis_path: str | Path = DEFAULT_STOPINTHIS_PATH,
    extra_stopwords: Iterable[str] | None = EXTRA_STOPWORDS,
    min_word_len: int = 2,
    top_n: int | None = None,
    keep_english: bool = True,
) -> pd.DataFrame:
    """Compute overall word frequency for a DataFrame."""
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    stopwords = load_stopwords(
        stopwords_path,
        use_stopinthis=use_stopinthis,
        stopinthis_path=stopinthis_path,
        extra_stopwords=extra_stopwords,
    )
    counter: Counter[str] = Counter()
    doc_counter: Counter[str] = Counter()

    for text in df[text_col].fillna("").astype(str):
        tokens = tokenize_text(
            text,
            stopwords=stopwords,
            min_word_len=min_word_len,
            keep_english=keep_english,
        )
        counter.update(tokens)
        doc_counter.update(set(tokens))

    total = sum(counter.values())
    rows = []
    for word, count in counter.most_common(top_n):
        rows.append(
            {
                "word": word,
                "count": count,
                "doc_count": doc_counter[word],
                "frequency": count / total if total else 0,
            }
        )
    return pd.DataFrame(rows)


def compute_grouped_word_frequency(
    df: pd.DataFrame,
    group_col: str,
    text_col: str = "content",
    stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
    use_stopinthis: bool = True,
    stopinthis_path: str | Path = DEFAULT_STOPINTHIS_PATH,
    extra_stopwords: Iterable[str] | None = EXTRA_STOPWORDS,
    min_word_len: int = 2,
    top_n_per_group: int = 30,
    keep_english: bool = True,
) -> pd.DataFrame:
    """Compute word frequency inside each group, for example by sentiment."""
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    frames: list[pd.DataFrame] = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        freq = compute_word_frequency(
            group_df,
            text_col=text_col,
            stopwords_path=stopwords_path,
            use_stopinthis=use_stopinthis,
            stopinthis_path=stopinthis_path,
            extra_stopwords=extra_stopwords,
            min_word_len=min_word_len,
            top_n=top_n_per_group,
            keep_english=keep_english,
        )
        freq.insert(0, group_col, group_value)
        frames.append(freq)

    if not frames:
        return pd.DataFrame(columns=[group_col, "word", "count", "doc_count", "frequency"])
    return pd.concat(frames, ignore_index=True)


def generate_wordcloud_from_frequency(
    freq_df: pd.DataFrame,
    output_path: str | Path,
    word_col: str = "word",
    count_col: str = "count",
    font_path: str | Path = DEFAULT_FONT_PATH,
    width: int = 1600,
    height: int = 1000,
    background_color: str = "white",
    max_words: int = 200,
    colormap: str = "viridis",
    random_state: int = 42,
) -> Path:
    """Generate a word cloud PNG from a word-frequency DataFrame."""
    from wordcloud import WordCloud

    missing = [column for column in [word_col, count_col] if column not in freq_df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    frequencies = {
        str(row[word_col]): float(row[count_col])
        for _, row in freq_df.iterrows()
        if str(row[word_col]).strip() and float(row[count_col]) > 0
    }
    if not frequencies:
        raise ValueError("No valid words found for word cloud.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wc = WordCloud(
        font_path=str(font_path),
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        random_state=random_state,
    )
    wc.generate_from_frequencies(frequencies)
    wc.to_file(str(output_path))
    return output_path


def generate_wordcloud_from_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    text_col: str = "content",
    stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
    use_stopinthis: bool = True,
    stopinthis_path: str | Path = DEFAULT_STOPINTHIS_PATH,
    extra_stopwords: Iterable[str] | None = EXTRA_STOPWORDS,
    min_word_len: int = 2,
    top_n: int | None = 200,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> Path:
    """Compute word frequency from text and save a word cloud PNG."""
    freq_df = compute_word_frequency(
        df,
        text_col=text_col,
        stopwords_path=stopwords_path,
        use_stopinthis=use_stopinthis,
        stopinthis_path=stopinthis_path,
        extra_stopwords=extra_stopwords,
        min_word_len=min_word_len,
        top_n=top_n,
    )
    return generate_wordcloud_from_frequency(
        freq_df,
        output_path=output_path,
        font_path=font_path,
    )
