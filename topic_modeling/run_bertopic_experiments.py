from __future__ import annotations

from pathlib import Path

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# BERTopic imports UMAP/PyNNDescent, whose installed numba decorators use
# cache=True. In this local conda env that can fail when imported through
# conda-run, so disable numba caching for this process only.
import numba

for _decorator_name in ["jit", "njit", "vectorize", "guvectorize"]:
    _decorator = getattr(numba, _decorator_name)

    def _without_cache(original):
        def patched(*args, **kwargs):
            kwargs.pop("cache", None)
            return original(*args, **kwargs)

        return patched

    setattr(numba, _decorator_name, _without_cache(_decorator))

import hdbscan
import jieba
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm.auto import tqdm
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
LLM_OUTPUT_DIR = ROOT / "analysis_json"
OUTPUT_DIR = ROOT / "output" / "bertopic_experiments"
MODEL_PATHS = {
    "qwen": ROOT / "embModel" / "Qwen3-Embedding-0.6B",
    "bge-small": ROOT / "embModel" / "BAAI" / "bge-small-zh-v1.5",
}
KEYWORD_FILTER_CONFIG_PATH = ROOT / "config" / "station_keyword_filter.json"
STOPWORD_PATHS = [ROOT / "stopword.txt", ROOT / "stopinthis.txt"]

MIN_TEXT_LENGTH = 4
N_TOPICS = 10
N_WORDS = 10
BATCH_SIZE = 16
RANDOM_STATE = 42
K_SWEEP_VALUES = list(range(5, 26))
SELECTED_KMEANS_K = N_TOPICS


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    data_variant: str
    clusterer: str
    embedding_name: str
    filter_business: bool
    text_mode: str
    cleaned_vectorizer: bool = True


EXPERIMENTS = [
    Experiment(
        "01_clean_desc_kmeans_qwen",
        "clean_all_description",
        "kmeans",
        "qwen",
        filter_business=False,
        text_mode="description",
    ),
    Experiment(
        "02_clean_desc_hdbscan_qwen",
        "clean_all_description",
        "hdbscan",
        "qwen",
        filter_business=False,
        text_mode="description",
    ),
    Experiment(
        "03_clean_desc_hdbscan_bge",
        "clean_all_description",
        "hdbscan",
        "bge-small",
        filter_business=False,
        text_mode="description",
    ),
    Experiment(
        "04_raw_title_desc_hdbscan_bge",
        "raw_notes_xlsx_title_desc",
        "hdbscan",
        "bge-small",
        filter_business=False,
        text_mode="title_description",
    ),
]


def load_stop_words() -> tuple[set[str], set[str]]:
    stop_words = {"无标题"}
    for path in STOPWORD_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"找不到停用词文件：{path}")
        stop_words.update(word.strip() for word in path.read_text(encoding="utf-8").splitlines() if word.strip())

    prefer_words: set[str] = set()
    if KEYWORD_FILTER_CONFIG_PATH.is_file():
        config = json.loads(KEYWORD_FILTER_CONFIG_PATH.read_text(encoding="utf-8"))
        exclude_words = {str(word).strip() for word in config.get("exclude_words", []) if str(word).strip()}
        prefer_words = {str(word).strip() for word in config.get("prefer_words", []) if str(word).strip()}
        stop_words.update(exclude_words)
    return stop_words, prefer_words


STOP_WORDS, KEYWORD_PREFER_WORDS = load_stop_words()
CHINESE_WORD = re.compile(r"^[\u4e00-\u9fff]{2,}$")


def tokenize_zh(text: str) -> list[str]:
    return [
        word.strip()
        for word in jieba.lcut(str(text))
        if len(word.strip()) > 1
        and not word.strip().isdigit()
        and word.strip() not in STOP_WORDS
    ]


def clean_text(text: object) -> str:
    text = "" if text is None or pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_records() -> pd.DataFrame:
    paths = sorted(LLM_OUTPUT_DIR.glob("*/*_analysis.json"))
    if not paths:
        raise FileNotFoundError(f"在 {LLM_OUTPUT_DIR} 下没有找到 *_analysis.json")

    records = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} 的顶层结构应为 list")
        station = path.parent.name
        for row in data:
            if isinstance(row, dict):
                records.append({**row, "station": station})
    return pd.DataFrame(records)


def load_notes_records() -> pd.DataFrame:
    paths = sorted((ROOT / "xlsx").glob("**/*notes*.xlsx"))
    if not paths:
        raise FileNotFoundError("在 xlsx 目录下没有找到 *notes*.xlsx 文件")

    frames = []
    skipped = []
    rename_map = {
        "笔记id": "note_id",
        "标题": "title",
        "描述": "desc",
        "笔记类型": "note_type",
        "点赞数量": "liked_count",
        "收藏数量": "collected_count",
        "评论数量": "comment_count",
        "分享数量": "share_count",
        "上传时间": "upload_time",
        "ip归属地": "ip_location",
    }
    for path in paths:
        try:
            frame = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            skipped.append(f"{path.relative_to(ROOT)}: {type(exc).__name__} {exc}")
            continue
        frame = frame.rename(columns=rename_map)
        frame["source_file"] = str(path.relative_to(ROOT))
        frame["station"] = path.stem.replace("_notes_合并", "").replace("_notes", "").replace("notes", "")
        frames.append(frame)

    if not frames:
        raise ValueError("没有可读取的 notes xlsx 文件")
    raw_notes = pd.concat(frames, ignore_index=True)
    for column in ["note_id", "title", "desc", "note_type"]:
        if column not in raw_notes.columns:
            raw_notes[column] = ""
    raw_notes = raw_notes.drop_duplicates(subset=["note_id"], keep="first").reset_index(drop=True)
    raw_notes["category"] = raw_notes["note_type"].fillna("").astype(str)
    raw_notes["category_group"] = "raw_notes_xlsx"
    raw_notes["description"] = raw_notes["desc"]

    if skipped:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "skipped_notes_files.txt").write_text("\n".join(skipped), encoding="utf-8")
        print("Skipped unreadable notes files:")
        for item in skipped:
            print(f"  - {item}")
    else:
        (OUTPUT_DIR / "skipped_notes_files.txt").unlink(missing_ok=True)
    return raw_notes


def build_dataset(raw: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    df = raw.copy()
    if experiment.filter_business:
        df = df.loc[df["category_group"].eq("business")].copy()

    if experiment.text_mode == "description":
        text = df["description"].map(clean_text)
    elif experiment.text_mode == "title_description":
        title = df.get("title", pd.Series("", index=df.index)).map(clean_text)
        desc_source = "desc" if "desc" in df.columns else "description"
        desc = df.get(desc_source, pd.Series("", index=df.index)).map(clean_text)
        text = (title + " " + desc).map(clean_text)
    else:
        raise ValueError(f"未知 text_mode: {experiment.text_mode}")

    keep = text.ne("") & text.ne("无标题") & text.str.len().ge(MIN_TEXT_LENGTH)
    dataset = df.loc[keep].copy().reset_index(drop=True)
    dataset["document"] = text.loc[keep].tolist()
    for column in ["note_id", "title", "description", "category", "category_group", "station"]:
        if column not in dataset.columns:
            dataset[column] = ""
    return dataset


def load_embedding_model(embedding_name: str) -> SentenceTransformer:
    model_path = MODEL_PATHS[embedding_name]
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"找不到 embedding 模型：{model_path}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(str(model_path), device=device, trust_remote_code=True)


def encode_documents(model: SentenceTransformer, docs: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.is_file():
        cached = np.load(cache_path)
        if len(cached) == len(docs):
            print(f"Using cached embeddings: {cache_path.name} ({len(cached):,} docs)", flush=True)
            return cached
        print(f"Embedding cache length changed for {cache_path.name}; recomputing.", flush=True)

    batches = [docs[index : index + BATCH_SIZE] for index in range(0, len(docs), BATCH_SIZE)]
    encoded_batches = []
    for batch in tqdm(batches, desc=f"Embedding {cache_path.stem}", unit="batch"):
        batch_embeddings = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        encoded_batches.append(batch_embeddings)

    embeddings = np.vstack(encoded_batches) if encoded_batches else np.empty((0, 0), dtype=np.float32)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings: {cache_path.name} ({len(embeddings):,} docs)", flush=True)
    return embeddings


def make_topic_model(experiment: Experiment) -> BERTopic:
    vectorizer_model = CountVectorizer(
        tokenizer=tokenize_zh if experiment.cleaned_vectorizer else None,
        token_pattern=None if experiment.cleaned_vectorizer else r"(?u)\b\w\w+\b",
        lowercase=False,
        ngram_range=(1, 1),
        min_df=1,
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    if experiment.clusterer == "kmeans":
        cluster_model = KMeans(n_clusters=SELECTED_KMEANS_K, random_state=RANDOM_STATE, n_init=10)
    elif experiment.clusterer == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    else:
        raise ValueError(f"未知 clusterer: {experiment.clusterer}")

    return BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        top_n_words=50,
        calculate_probabilities=False,
        low_memory=True,
        verbose=False,
    )


def safe_cluster_metric(metric_func, embeddings: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    labels = np.asarray(labels)
    valid_mask = labels != -1
    x = embeddings[valid_mask]
    y = labels[valid_mask]
    unique_labels = np.unique(y)
    if len(unique_labels) < 2 or len(y) <= len(unique_labels):
        return np.nan
    try:
        return float(metric_func(x, y, **kwargs))
    except ValueError:
        return np.nan


def topic_word_lists(topic_model: BERTopic, labels: np.ndarray, top_n_words: int = N_WORDS) -> list[list[str]]:
    word_lists = []
    for topic_id in sorted(int(topic) for topic in np.unique(labels) if int(topic) != -1):
        raw_words = topic_model.get_topic(topic_id) or []
        words = []
        for word, _ in raw_words:
            word = str(word).strip()
            if word and word not in STOP_WORDS:
                words.append(word)
        if words:
            word_lists.append(words[:top_n_words])
    return word_lists


def topic_word_lists_from_labels(docs: list[str], labels: np.ndarray, top_n_words: int = N_WORDS) -> list[list[str]]:
    labels = np.asarray(labels)
    vectorizer = CountVectorizer(
        tokenizer=tokenize_zh,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=1,
    )
    grouped_docs = []
    grouped_labels = []
    for topic_id in sorted(int(topic) for topic in np.unique(labels) if int(topic) != -1):
        topic_docs = [doc for doc, label in zip(docs, labels) if int(label) == topic_id]
        if topic_docs:
            grouped_labels.append(topic_id)
            grouped_docs.append(" ".join(topic_docs))
    if not grouped_docs:
        return []

    matrix = vectorizer.fit_transform(grouped_docs)
    terms = vectorizer.get_feature_names_out()
    word_lists = []
    for row in matrix.toarray():
        top_indices = row.argsort()[::-1]
        words = [terms[index] for index in top_indices if row[index] > 0 and terms[index] not in STOP_WORDS]
        if words:
            word_lists.append(words[:top_n_words])
    return word_lists


def topic_diversity_score(word_lists: list[list[str]]) -> float:
    words = [word for word_list in word_lists for word in word_list]
    if not words:
        return np.nan
    return len(set(words)) / len(words)


def topic_coherence_npmi(docs: list[str], word_lists: list[list[str]], eps: float = 1e-12) -> float:
    tokenized_docs = [set(tokenize_zh(doc)) for doc in docs]
    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return np.nan

    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    pair_scores = []
    for words in word_lists:
        words = [word for word in dict.fromkeys(words) if doc_freq.get(word, 0) > 0]
        if len(words) < 2:
            continue
        for i, word_i in enumerate(words[:-1]):
            docs_i = doc_freq[word_i]
            for word_j in words[i + 1 :]:
                docs_j = doc_freq[word_j]
                co_docs = sum(1 for tokens in tokenized_docs if word_i in tokens and word_j in tokens)
                p_i = docs_i / n_docs
                p_j = docs_j / n_docs
                p_ij = (co_docs + eps) / n_docs
                pair_scores.append(np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij)))
    return float(np.mean(pair_scores)) if pair_scores else np.nan


def topic_keywords(topic_model: BERTopic, topic_id: int, top_n_words: int = N_WORDS) -> str:
    words = topic_model.get_topic(topic_id) or []
    cleaned = [str(word).strip() for word, _ in words if str(word).strip() and str(word).strip() not in STOP_WORDS]
    return " / ".join(cleaned[:top_n_words])


def add_composite_score(metrics_df: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "silhouette_score": True,
        "davies_bouldin_index": False,
        "calinski_harabasz_index": True,
        "topic_coherence_npmi": True,
        "topic_diversity": True,
    }
    scored = metrics_df.copy()
    norm_columns = []
    for column, higher_is_better in score_columns.items():
        values = scored[column].astype(float)
        valid = values.dropna()
        norm_column = f"{column}_norm"
        norm_columns.append(norm_column)
        if valid.empty:
            scored[norm_column] = np.nan
            continue
        min_value = valid.min()
        max_value = valid.max()
        if np.isclose(max_value, min_value):
            normalized = pd.Series(0.5, index=scored.index)
        else:
            normalized = (values - min_value) / (max_value - min_value)
        if not higher_is_better:
            normalized = 1 - normalized
        scored[norm_column] = normalized
    scored["composite_score"] = scored[norm_columns].mean(axis=1, skipna=True)
    return scored


def run_kmeans_k_sweep(dataset: pd.DataFrame, embeddings: np.ndarray) -> tuple[int, pd.DataFrame]:
    docs = dataset["document"].tolist()
    reduced_embeddings = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    ).fit_transform(embeddings)
    rows = []
    for k in K_SWEEP_VALUES:
        if k >= len(docs):
            continue
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(reduced_embeddings)
        word_lists = topic_word_lists_from_labels(docs, labels)
        rows.append(
            {
                "k": k,
                "n_documents": int(len(docs)),
                "n_topics": int(k),
                "outlier_count": 0,
                "outlier_rate": 0.0,
                "silhouette_score": safe_cluster_metric(silhouette_score, embeddings, labels, metric="cosine"),
                "davies_bouldin_index": safe_cluster_metric(davies_bouldin_score, embeddings, labels),
                "calinski_harabasz_index": safe_cluster_metric(calinski_harabasz_score, embeddings, labels),
                "topic_coherence_npmi": topic_coherence_npmi(docs, word_lists),
                "topic_diversity": topic_diversity_score(word_lists),
            }
        )
    sweep_df = add_composite_score(pd.DataFrame(rows))
    sweep_df = sweep_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    best_k = int(sweep_df.loc[0, "k"])
    sweep_df.to_csv(OUTPUT_DIR / "kmeans_k_sweep_metrics.csv", index=False, encoding="utf-8-sig")
    return best_k, sweep_df


def evaluate_model(
    experiment: Experiment,
    dataset: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    topic_model: BERTopic,
) -> dict[str, object]:
    labels = np.asarray(labels)
    valid_labels = labels[labels != -1]
    unique_topics = np.unique(valid_labels)
    outlier_count = int(np.sum(labels == -1))
    word_lists = topic_word_lists(topic_model, labels)
    return {
        "experiment_id": experiment.experiment_id,
        "data_variant": experiment.data_variant,
        "clusterer": experiment.clusterer,
        "embedding": experiment.embedding_name,
        "n_documents": int(len(dataset)),
        "n_topics": int(len(unique_topics)),
        "outlier_count": outlier_count,
        "outlier_rate": outlier_count / len(dataset) if len(dataset) else np.nan,
        "silhouette_score": safe_cluster_metric(silhouette_score, embeddings, labels, metric="cosine"),
        "davies_bouldin_index": safe_cluster_metric(davies_bouldin_score, embeddings, labels),
        "calinski_harabasz_index": safe_cluster_metric(calinski_harabasz_score, embeddings, labels),
        "topic_coherence_npmi": topic_coherence_npmi(dataset["document"].tolist(), word_lists),
        "topic_diversity": topic_diversity_score(word_lists),
    }


def summarize_topics(experiment: Experiment, dataset: pd.DataFrame, labels: np.ndarray, topic_model: BERTopic) -> pd.DataFrame:
    rows = []
    assigned = dataset.copy()
    assigned["topic"] = labels
    topic_info = topic_model.get_topic_info()
    topic_counts = assigned["topic"].value_counts().to_dict()
    for topic_id in sorted(topic_counts):
        topic_docs = assigned.loc[assigned["topic"].eq(topic_id)]
        examples = " || ".join(topic_docs["document"].head(3).tolist())
        category_mix = "；".join(f"{k}({v})" for k, v in topic_docs["category"].value_counts().head(5).items())
        info_row = topic_info.loc[topic_info["Topic"].eq(topic_id)]
        representative_docs = ""
        if not info_row.empty and "Representative_Docs" in info_row.columns:
            representative_docs = str(info_row.iloc[0]["Representative_Docs"])
        rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "topic": int(topic_id),
                "count": int(topic_counts[topic_id]),
                "share": topic_counts[topic_id] / len(assigned),
                "keywords": topic_keywords(topic_model, int(topic_id)),
                "category_mix": category_mix,
                "examples": examples,
                "representative_docs": representative_docs,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    experiment: Experiment,
    raw: pd.DataFrame,
    raw_notes: pd.DataFrame,
    embedding_models: dict[str, SentenceTransformer],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if experiment.data_variant == "raw_notes_xlsx_title_desc":
        raw = raw_notes
    dataset = build_dataset(raw, experiment)
    docs = dataset["document"].tolist()
    if len(docs) < 2:
        raise ValueError(f"{experiment.experiment_id} 可用文本不足")

    model = embedding_models.get(experiment.embedding_name)
    if model is None:
        model = load_embedding_model(experiment.embedding_name)
        embedding_models[experiment.embedding_name] = model

    cache_path = OUTPUT_DIR / f"{experiment.data_variant}_{experiment.embedding_name}_embeddings.npy"
    embeddings = encode_documents(model, docs, cache_path)
    if len(embeddings) != len(docs):
        raise ValueError(f"{experiment.experiment_id} embedding 数量和文本数量不一致")

    topic_model = make_topic_model(experiment)
    labels, _ = topic_model.fit_transform(docs, embeddings)
    labels = np.asarray(labels)

    assignments = dataset[
        ["station", "note_id", "title", "description", "category", "category_group", "document"]
    ].copy()
    assignments["experiment_id"] = experiment.experiment_id
    assignments["topic"] = labels
    assignments["topic_keywords"] = [topic_keywords(topic_model, int(topic)) if int(topic) != -1 else "outlier" for topic in labels]

    metrics = evaluate_model(experiment, dataset, embeddings, labels, topic_model)
    topic_summary = summarize_topics(experiment, dataset, labels, topic_model)
    return metrics, topic_summary, assignments


def main() -> None:
    global SELECTED_KMEANS_K

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_records()
    raw_notes = load_notes_records()
    embedding_models: dict[str, SentenceTransformer] = {}
    metrics_rows = []
    topic_summaries = []
    assignments = []

    kmeans_experiment = EXPERIMENTS[0]
    kmeans_dataset = build_dataset(raw, kmeans_experiment)
    qwen_model = load_embedding_model(kmeans_experiment.embedding_name)
    embedding_models[kmeans_experiment.embedding_name] = qwen_model
    qwen_cache_path = OUTPUT_DIR / f"{kmeans_experiment.data_variant}_{kmeans_experiment.embedding_name}_embeddings.npy"
    qwen_embeddings = encode_documents(qwen_model, kmeans_dataset["document"].tolist(), qwen_cache_path)
    SELECTED_KMEANS_K, k_sweep_df = run_kmeans_k_sweep(kmeans_dataset, qwen_embeddings)
    print(
        f"\nSelected KMeans k={SELECTED_KMEANS_K} from kmeans_k_sweep_metrics.csv",
        flush=True,
    )
    print(k_sweep_df.head(10).to_string(index=False), flush=True)

    for experiment in EXPERIMENTS:
        print(f"\\n=== Running {experiment.experiment_id} ===", flush=True)
        metrics, topic_summary, assignment = run_experiment(experiment, raw, raw_notes, embedding_models)
        metrics_rows.append(metrics)
        topic_summaries.append(topic_summary)
        assignments.append(assignment)
        print(pd.DataFrame([metrics]).to_string(index=False), flush=True)

    metrics_df = pd.DataFrame(metrics_rows)
    topic_summary_df = pd.concat(topic_summaries, ignore_index=True)
    assignment_df = pd.concat(assignments, ignore_index=True)

    metrics_df.to_csv(OUTPUT_DIR / "experiment_metrics.csv", index=False, encoding="utf-8-sig")
    topic_summary_df.to_csv(OUTPUT_DIR / "topic_summaries.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(OUTPUT_DIR / "topic_assignments.csv", index=False, encoding="utf-8-sig")

    print("\\nSaved:")
    print(OUTPUT_DIR / "experiment_metrics.csv")
    print(OUTPUT_DIR / "topic_summaries.csv")
    print(OUTPUT_DIR / "topic_assignments.csv")


if __name__ == "__main__":
    main()
