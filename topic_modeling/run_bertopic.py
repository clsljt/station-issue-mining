"""
Run BERTopic on cleaned station opinion data.

Jupyter example
---------------
import sys
sys.path.append("/Users/clsljt/Documents/hhgh/station/scripts")

from run_bertopic import load_cleaned_data, run_bertopic

df = load_cleaned_data()
result = run_bertopic(df, station="北京西站", min_topic_size=20)
result.topic_info.head()
result.assignments.head()

CLI example
-----------
python run_bertopic.py --station 北京西站 --min-topic-size 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


DEFAULT_STATION_DIR = Path("/Users/clsljt/Documents/hhgh/station")
DEFAULT_OUTPUT_DIR = DEFAULT_STATION_DIR / "outputs"
DEFAULT_CLEANED_CSV = DEFAULT_OUTPUT_DIR / "cleaned_station_texts.csv"
DEFAULT_EMBEDDING_MODEL = DEFAULT_STATION_DIR / "embModel" / "BAAI" / "bge-small-zh-v1.5"

# Avoid numba/umap cache locator errors in some conda + Jupyter setups.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/station_numba_cache")


DEFAULT_STOPWORDS = {
    "北京西站",
    "北京朝阳站",
    "北京",
    "西站",
    "朝阳站",
    "车站",
    "高铁",
    "火车",
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
    "一下",
    "直接",
    "现在",
    "今天",
    "内容",
    "帖子",
    "标题",
    "包含",
    "标签",
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
    "丰台站",
    "大兴",
    "正常",
    "成功",
    "希望",
    "主要",
    "情况",
    "方式",
}


@dataclass
class TopicRunResult:
    topic_model: object
    input_data: pd.DataFrame
    assignments: pd.DataFrame
    topic_info: pd.DataFrame
    representative_docs: pd.DataFrame
    output_dir: Path


def require_package(import_name: str, install_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError as exc:
        package = install_name or import_name
        raise ImportError(
            f"Missing package '{package}'. Install it in the Jupyter environment first. "
            f"Suggested: pip install {package}"
        ) from exc


def load_cleaned_data(cleaned_csv: str | Path = DEFAULT_CLEANED_CSV) -> pd.DataFrame:
    cleaned_csv = Path(cleaned_csv)
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {cleaned_csv}. Run 01_clean_station_data.py first."
        )
    df = pd.read_csv(cleaned_csv)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def simple_chinese_tokenizer(text: str) -> list[str]:
    """Tokenizer for CountVectorizer. Uses jieba when available, otherwise regex chunks."""
    text = "" if pd.isna(text) else str(text)
    try:
        import jieba

        tokens = jieba.lcut(text)
    except ImportError:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", text)

    clean_tokens = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        if token in DEFAULT_STOPWORDS:
            continue
        if re.fullmatch(r"\d+", token):
            continue
        clean_tokens.append(token)
    return clean_tokens


def prepare_topic_data(
    df: pd.DataFrame,
    station: str | None = None,
    source_types: list[str] | None = None,
    text_col: str = "model_text",
    valid_only: bool = True,
    responsibility_relevant_only: bool = False,
    min_chars: int = 4,
) -> pd.DataFrame:
    data = df.copy()
    if valid_only and "is_valid_text" in data.columns:
        data = data[data["is_valid_text"].astype(str).str.lower().isin(["true", "1"])]
    if responsibility_relevant_only and "is_responsibility_relevant" in data.columns:
        data = data[data["is_responsibility_relevant"].astype(str).str.lower().isin(["true", "1"])]
    if station:
        data = data[data["station"].eq(station)]
    if source_types:
        data = data[data["source_type"].isin(source_types)]
    if text_col not in data.columns:
        raise KeyError(f"Text column '{text_col}' is missing.")

    data[text_col] = data[text_col].fillna("").astype(str).str.strip()
    data = data[data[text_col].str.len() >= min_chars].copy()
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("No texts left after filtering. Relax station/source/min_chars filters.")
    return data


def build_embedding_model(model_name: str | Path = DEFAULT_EMBEDDING_MODEL) -> object:
    require_package("sentence_transformers", "sentence-transformers")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_name))


def build_topic_model(
    embedding_model: object | str | Path = DEFAULT_EMBEDDING_MODEL,
    min_topic_size: int = 15,
    nr_topics: str | int | None = "auto",
    ngram_range: tuple[int, int] = (1, 2),
    vectorizer_min_df: int = 1,
    calculate_probabilities: bool = True,
    tokenizer: Callable[[str], list[str]] = simple_chinese_tokenizer,
    verbose: bool = True,
) -> object:
    require_package("bertopic")
    require_package("sklearn", "scikit-learn")
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    if isinstance(embedding_model, (str, Path)):
        embedding_model = build_embedding_model(embedding_model)

    vectorizer_model = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        stop_words=None,
        ngram_range=ngram_range,
        min_df=vectorizer_min_df,
    )

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=calculate_probabilities,
        verbose=verbose,
    )


def collect_representative_docs(
    topic_model: object,
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    text_col: str,
    top_n: int = 8,
) -> pd.DataFrame:
    rows = []
    merged = data.copy()
    merged["topic"] = assignments["topic"].to_numpy()
    merged["topic_probability"] = assignments["topic_probability"].to_numpy()

    for topic_id, group in merged[merged["topic"].ne(-1)].groupby("topic"):
        group = group.sort_values(
            ["topic_probability", "like_count"],
            ascending=[False, False],
        ).head(top_n)
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            rows.append(
                {
                    "topic": topic_id,
                    "rank": rank,
                    "station": row.get("station"),
                    "source_type": row.get("source_type"),
                    "content": row.get("clean_text", row.get(text_col)),
                    "model_text": row.get(text_col),
                    "topic_probability": row.get("topic_probability"),
                    "like_count": row.get("like_count"),
                    "issue_label_text": row.get("issue_label_text"),
                    "responsibility_label_text": row.get("responsibility_label_text"),
                    "primary_issue_category": row.get("primary_issue_category"),
                    "responsible_owner": row.get("responsible_owner"),
                    "suggested_action": row.get("suggested_action"),
                    "issue_severity": row.get("issue_severity"),
                    "created_at": row.get("created_at"),
                    "url": row.get("url"),
                }
            )

    return pd.DataFrame(rows)


def run_bertopic(
    df: pd.DataFrame | None = None,
    cleaned_csv: str | Path = DEFAULT_CLEANED_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    station: str | None = None,
    source_types: list[str] | None = None,
    text_col: str = "model_text",
    embedding_model: object | str | Path = DEFAULT_EMBEDDING_MODEL,
    min_topic_size: int = 15,
    nr_topics: str | int | None = "auto",
    vectorizer_min_df: int = 1,
    calculate_probabilities: bool = True,
    reduce_outliers: bool = True,
    responsibility_relevant_only: bool = False,
    write_outputs: bool = True,
) -> TopicRunResult:
    """Fit BERTopic and return model plus export-ready DataFrames."""
    if df is None:
        df = load_cleaned_data(cleaned_csv)

    data = prepare_topic_data(
        df,
        station=station,
        source_types=source_types,
        text_col=text_col,
        responsibility_relevant_only=responsibility_relevant_only,
    )
    docs = data[text_col].tolist()

    topic_model = build_topic_model(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_min_df=vectorizer_min_df,
        calculate_probabilities=calculate_probabilities,
    )

    topics, probabilities = topic_model.fit_transform(docs)

    if reduce_outliers and hasattr(topic_model, "reduce_outliers") and any(topic == -1 for topic in topics):
        try:
            reduced_topics = topic_model.reduce_outliers(docs, topics, probabilities=probabilities)
            topic_model.update_topics(docs, topics=reduced_topics)
            topics = reduced_topics
        except Exception as exc:
            print(f"Warning: reduce_outliers failed, keeping original topics. Reason: {exc}")

    assignment_columns = [
        "station",
        "source_type",
        "note_id",
        "comment_id",
        "clean_text",
        text_col,
        "like_count",
        "issue_label_text",
        "responsibility_label_text",
        "non_responsibility_label_text",
        "responsibility_relevance_reason",
        "primary_issue_category",
        "responsible_owner",
        "suggested_action",
        "issue_severity",
        "rectification_ready",
        "created_at",
        "url",
    ]
    assignments = data[[column for column in assignment_columns if column in data.columns]].copy()
    assignments["topic"] = topics
    assignments["topic_probability"] = extract_topic_probabilities(topics, probabilities)

    topic_info = topic_model.get_topic_info()
    topic_info = add_topic_summary_metrics(topic_info, assignments)
    representative_docs = collect_representative_docs(topic_model, data, assignments, text_col=text_col)

    output_dir = Path(output_dir)
    run_name = make_run_name(station=station, source_types=source_types, min_topic_size=min_topic_size)
    run_output_dir = output_dir / run_name

    if write_outputs:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(run_output_dir / "topic_assignments.csv", index=False, encoding="utf-8-sig")
        topic_info.to_csv(run_output_dir / "topic_info.csv", index=False, encoding="utf-8-sig")
        representative_docs.to_csv(
            run_output_dir / "representative_docs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with open(run_output_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "station": station,
                    "source_types": source_types,
                    "text_col": text_col,
                    "embedding_model": str(embedding_model),
                    "min_topic_size": min_topic_size,
                    "nr_topics": nr_topics,
                    "vectorizer_min_df": vectorizer_min_df,
                    "calculate_probabilities": calculate_probabilities,
                    "reduce_outliers": reduce_outliers,
                    "responsibility_relevant_only": responsibility_relevant_only,
                    "rows": len(data),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        try:
            topic_model.save(run_output_dir / "bertopic_model", serialization="pickle")
        except Exception as exc:
            print(f"Warning: could not save BERTopic model. Reason: {exc}")

    return TopicRunResult(
        topic_model=topic_model,
        input_data=data,
        assignments=assignments,
        topic_info=topic_info,
        representative_docs=representative_docs,
        output_dir=run_output_dir,
    )


def prepare_filtered_topic_data(
    df: pd.DataFrame,
    id_col: str = "id",
    text_col: str = "content",
    min_chars: int = 4,
) -> pd.DataFrame:
    """Prepare a simple id/text DataFrame for BERTopic."""
    missing = [column for column in [id_col, text_col] if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    data = df.copy()
    data[id_col] = data[id_col]
    data[text_col] = data[text_col].fillna("").astype(str).str.strip()
    data = data[data[text_col].str.len() >= min_chars].copy()
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("No texts left after filtering. Relax min_chars or check your text column.")
    return data


def run_bertopic_on_filtered_df(
    filtered_df: pd.DataFrame,
    id_col: str = "id",
    text_col: str = "content",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_name: str = "bertopic_filtered_df",
    embedding_model: object | str | Path = DEFAULT_EMBEDDING_MODEL,
    min_topic_size: int = 15,
    nr_topics: str | int | None = "auto",
    vectorizer_min_df: int = 1,
    calculate_probabilities: bool = True,
    reduce_outliers: bool = True,
    responsibility_relevant_only: bool = False,
    write_outputs: bool = True,
) -> TopicRunResult:
    """
    Run BERTopic directly on a filtered id/text DataFrame.

    This is the recommended entry point for the step-by-step notebook pipeline:
    clean -> sentiment filter -> BERTopic.
    """
    data = filtered_df.copy()
    if responsibility_relevant_only:
        if "is_responsibility_relevant" not in data.columns:
            try:
                from clean_station_data import annotate_text_id_dataframe

                data = annotate_text_id_dataframe(data, id_col=id_col, text_col=text_col)
            except Exception as exc:
                print(f"Warning: responsibility annotation failed, keeping all texts. Reason: {exc}")
        if "is_responsibility_relevant" in data.columns:
            data = data[data["is_responsibility_relevant"].astype(str).str.lower().isin(["true", "1"])].copy()

    data = prepare_filtered_topic_data(data, id_col=id_col, text_col=text_col)
    docs = data[text_col].tolist()

    topic_model = build_topic_model(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_min_df=vectorizer_min_df,
        calculate_probabilities=calculate_probabilities,
    )
    topics, probabilities = topic_model.fit_transform(docs)

    if reduce_outliers and hasattr(topic_model, "reduce_outliers") and any(topic == -1 for topic in topics):
        try:
            reduced_topics = topic_model.reduce_outliers(docs, topics, probabilities=probabilities)
            topic_model.update_topics(docs, topics=reduced_topics)
            topics = reduced_topics
        except Exception as exc:
            print(f"Warning: reduce_outliers failed, keeping original topics. Reason: {exc}")

    assignments = pd.DataFrame(
        {
            "id": data[id_col],
            "content": data[text_col],
            "topic": topics,
            "topic_probability": extract_topic_probabilities(topics, probabilities),
        }
    )
    for optional_col in [
        "sentiment",
        "p_negative",
        "p_positive",
        "is_responsibility_relevant",
        "has_problem_signal",
        "responsibility_label_text",
        "non_responsibility_label_text",
        "responsibility_relevance_reason",
        "primary_issue_category",
        "responsible_owner",
        "suggested_action",
        "issue_severity",
        "rectification_ready",
    ]:
        if optional_col in data.columns:
            assignments[optional_col] = data[optional_col]

    topic_info = topic_model.get_topic_info()
    topic_info = add_simple_topic_summary_metrics(topic_info, assignments)
    representative_docs = collect_simple_representative_docs(assignments)

    output_dir = Path(output_dir)
    run_output_dir = output_dir / run_name
    if write_outputs:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(run_output_dir / "topic_assignments.csv", index=False, encoding="utf-8-sig")
        topic_info.to_csv(run_output_dir / "topic_info.csv", index=False, encoding="utf-8-sig")
        representative_docs.to_csv(
            run_output_dir / "representative_docs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with open(run_output_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "input": "filtered_df",
                    "id_col": id_col,
                    "text_col": text_col,
                    "embedding_model": str(embedding_model),
                    "min_topic_size": min_topic_size,
                    "nr_topics": nr_topics,
                    "vectorizer_min_df": vectorizer_min_df,
                    "calculate_probabilities": calculate_probabilities,
                    "reduce_outliers": reduce_outliers,
                    "responsibility_relevant_only": responsibility_relevant_only,
                    "rows": len(data),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        try:
            topic_model.save(run_output_dir / "bertopic_model", serialization="pickle")
        except Exception as exc:
            print(f"Warning: could not save BERTopic model. Reason: {exc}")

    return TopicRunResult(
        topic_model=topic_model,
        input_data=data,
        assignments=assignments,
        topic_info=topic_info,
        representative_docs=representative_docs,
        output_dir=run_output_dir,
    )


def get_bertopic_visualizations(
    result: TopicRunResult,
    docs: list[str] | None = None,
    include_documents: bool = False,
    top_n_topics: int | None = None,
) -> dict[str, object]:
    """
    Build BERTopic Plotly visualizations for Jupyter.

    Example:
        figs = get_bertopic_visualizations(result)
        figs["barchart"].show()
    """
    topic_model = result.topic_model
    figures: dict[str, object] = {}

    try:
        figures["topics"] = topic_model.visualize_topics()
    except Exception as exc:
        print(f"Warning: visualize_topics failed: {exc}")

    try:
        figures["barchart"] = topic_model.visualize_barchart(top_n_topics=top_n_topics)
    except Exception as exc:
        print(f"Warning: visualize_barchart failed: {exc}")

    try:
        figures["heatmap"] = topic_model.visualize_heatmap()
    except Exception as exc:
        print(f"Warning: visualize_heatmap failed: {exc}")

    try:
        figures["hierarchy"] = topic_model.visualize_hierarchy()
    except Exception as exc:
        print(f"Warning: visualize_hierarchy failed: {exc}")

    try:
        figures["term_rank"] = topic_model.visualize_term_rank()
    except Exception as exc:
        print(f"Warning: visualize_term_rank failed: {exc}")

    if include_documents:
        if docs is None:
            if "content" in result.input_data.columns:
                docs = result.input_data["content"].fillna("").astype(str).tolist()
            elif "model_text" in result.input_data.columns:
                docs = result.input_data["model_text"].fillna("").astype(str).tolist()
        if docs:
            try:
                figures["documents"] = topic_model.visualize_documents(docs)
            except Exception as exc:
                print(f"Warning: visualize_documents failed: {exc}")

    return figures


def save_bertopic_visualizations(
    result: TopicRunResult,
    output_dir: str | Path | None = None,
    docs: list[str] | None = None,
    include_documents: bool = False,
    top_n_topics: int | None = None,
) -> dict[str, Path]:
    """
    Save BERTopic visualizations as standalone HTML files.

    Returns a mapping from visualization name to saved HTML path.
    """
    output_path = Path(output_dir) if output_dir else result.output_dir / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)

    figures = get_bertopic_visualizations(
        result,
        docs=docs,
        include_documents=include_documents,
        top_n_topics=top_n_topics,
    )
    saved: dict[str, Path] = {}
    for name, fig in figures.items():
        path = output_path / f"{name}.html"
        fig.write_html(str(path))
        saved[name] = path
    return saved


def extract_topic_probabilities(topics: list[int], probabilities: object) -> list[float | None]:
    if probabilities is None:
        return [None] * len(topics)
    try:
        import numpy as np

        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            return [float(value) for value in probs]
        values: list[float | None] = []
        for i, topic in enumerate(topics):
            if topic == -1:
                values.append(None)
            else:
                values.append(float(probs[i].max()))
        return values
    except Exception:
        return [None] * len(topics)


def add_topic_summary_metrics(topic_info: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "assigned_rows": ("clean_text", "size"),
        "like_sum": ("like_count", "sum"),
        "like_median": ("like_count", "median"),
        "station_mix": ("station", lambda values: "|".join(values.value_counts().index.astype(str)[:3])),
        "issue_mix": ("issue_label_text", summarize_issue_labels),
    }
    if "primary_issue_category" in assignments.columns:
        agg["primary_issue_mix"] = (
            "primary_issue_category",
            lambda values: "|".join(values.value_counts().index.astype(str)[:5]),
        )
    if "responsible_owner" in assignments.columns:
        agg["owner_mix"] = ("responsible_owner", lambda values: "|".join(values.value_counts().index.astype(str)[:3]))
    if "issue_severity" in assignments.columns:
        agg["severity_mix"] = ("issue_severity", lambda values: "|".join(values.value_counts().index.astype(str)[:3]))
    metrics = assignments.groupby("topic", as_index=False).agg(**agg).rename(columns={"topic": "Topic"})
    return topic_info.merge(metrics, on="Topic", how="left")


def add_simple_topic_summary_metrics(topic_info: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    agg = {"assigned_rows": ("content", "size")}
    if "sentiment" in assignments.columns:
        agg["sentiment_mix"] = ("sentiment", lambda values: "|".join(values.value_counts().index.astype(str)[:3]))
    metrics = assignments.groupby("topic", as_index=False).agg(**agg).rename(columns={"topic": "Topic"})
    return topic_info.merge(metrics, on="Topic", how="left")


def collect_simple_representative_docs(assignments: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    for topic_id, group in assignments[assignments["topic"].ne(-1)].groupby("topic"):
        group = group.sort_values("topic_probability", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            item = {
                "topic": topic_id,
                "rank": rank,
                "id": row.get("id"),
                "content": row.get("content"),
                "topic_probability": row.get("topic_probability"),
            }
            for optional_col in ["sentiment", "p_negative", "p_positive"]:
                if optional_col in row.index:
                    item[optional_col] = row.get(optional_col)
            rows.append(item)
    return pd.DataFrame(rows)


def summarize_issue_labels(values: Iterable[object]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if pd.isna(value) or not str(value).strip():
            continue
        for label in str(value).split("|"):
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
    return "|".join(label for label, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5])


def make_run_name(
    station: str | None,
    source_types: list[str] | None,
    min_topic_size: int,
) -> str:
    station_part = station or "all_stations"
    source_part = "-".join(source_types) if source_types else "all_sources"
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", f"bertopic_{station_part}_{source_part}_mts{min_topic_size}")
    return safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic on cleaned station data.")
    parser.add_argument("--cleaned-csv", default=str(DEFAULT_CLEANED_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--station", default=None, help="Example: 北京西站")
    parser.add_argument(
        "--source-types",
        nargs="*",
        default=None,
        choices=["note", "comment"],
        help="Optionally choose note/comment.",
    )
    parser.add_argument("--text-col", default="model_text")
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--min-topic-size", type=int, default=15)
    parser.add_argument("--nr-topics", default="auto")
    parser.add_argument("--vectorizer-min-df", type=int, default=1)
    parser.add_argument("--no-reduce-outliers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nr_topics: str | int | None
    if args.nr_topics.lower() == "none":
        nr_topics = None
    elif args.nr_topics.lower() == "auto":
        nr_topics = "auto"
    else:
        nr_topics = int(args.nr_topics)

    result = run_bertopic(
        cleaned_csv=args.cleaned_csv,
        output_dir=args.output_dir,
        station=args.station,
        source_types=args.source_types,
        text_col=args.text_col,
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=nr_topics,
        vectorizer_min_df=args.vectorizer_min_df,
        reduce_outliers=not args.no_reduce_outliers,
        write_outputs=True,
    )
    print("Rows used:", len(result.input_data))
    print("Output dir:", result.output_dir)
    print(result.topic_info.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
