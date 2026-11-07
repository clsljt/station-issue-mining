"""
Sentiment filtering for cleaned station comments.

Input: a DataFrame with id/text columns.
Output: the same id/text rows plus sentiment columns, filtered to neutral/negative by default.

Jupyter example
---------------
import sys
sys.path.insert(0, "/Users/clsljt/Documents/hhgh/station/scripts")

from sentiment_filter import filter_neutral_negative

filtered_df = filter_neutral_negative(cleaned_df)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_STATION_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SENTIMENT_MODEL = DEFAULT_STATION_DIR / "senModel"


def _get_device(device: str = "auto") -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_sentiment_model(model_dir: str | Path = DEFAULT_SENTIMENT_MODEL, device: str = "auto"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Sentiment model directory not found: {model_dir}")

    runtime_device = _get_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(runtime_device)
    model.eval()
    return tokenizer, model, runtime_device


def analyze_sentiment_df(
    df: pd.DataFrame,
    text_col: str = "content",
    id_col: str = "id",
    model_dir: str | Path = DEFAULT_SENTIMENT_MODEL,
    batch_size: int = 32,
    max_length: int = 256,
    positive_threshold: float = 0.65,
    negative_threshold: float = 0.55,
    device: str = "auto",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add sentiment columns to an id/text DataFrame.

    The local Erlangshen model is binary: Negative/Positive. To get a neutral
    class, this function treats low-confidence predictions as neutral:

    - p_negative >= negative_threshold -> negative
    - p_positive >= positive_threshold -> positive
    - otherwise -> neutral
    """
    import torch

    missing = [column for column in [id_col, text_col] if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    tokenizer, model, runtime_device = _load_sentiment_model(model_dir=model_dir, device=device)

    ids = df[id_col].tolist()
    texts = df[text_col].fillna("").astype(str).tolist()
    rows = []

    batch_starts = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            batch_starts = tqdm(batch_starts, total=(len(texts) + batch_size - 1) // batch_size, desc="Sentiment")
        except ImportError:
            pass

    for start in batch_starts:
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(runtime_device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for offset, prob in enumerate(probs):
            p_negative = float(prob[0])
            p_positive = float(prob[1])
            if p_negative >= negative_threshold:
                sentiment = "negative"
            elif p_positive >= positive_threshold:
                sentiment = "positive"
            else:
                sentiment = "neutral"

            rows.append(
                {
                    "id": ids[start + offset],
                    "content": batch_texts[offset],
                    "sentiment": sentiment,
                    "p_negative": p_negative,
                    "p_positive": p_positive,
                }
            )

    return pd.DataFrame(rows)


def filter_neutral_negative(
    df: pd.DataFrame,
    text_col: str = "content",
    id_col: str = "id",
    model_dir: str | Path = DEFAULT_SENTIMENT_MODEL,
    keep_sentiments: tuple[str, ...] = ("negative", "neutral"),
    batch_size: int = 32,
    max_length: int = 256,
    positive_threshold: float = 0.65,
    negative_threshold: float = 0.55,
    device: str = "auto",
    keep_scores: bool = True,
    preserve_input_columns: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Return rows whose sentiment is neutral or negative by default."""
    scored = analyze_sentiment_df(
        df=df,
        text_col=text_col,
        id_col=id_col,
        model_dir=model_dir,
        batch_size=batch_size,
        max_length=max_length,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        device=device,
        show_progress=show_progress,
    )
    if preserve_input_columns:
        base = df.reset_index(drop=True).copy()
        for column in ["sentiment", "p_negative", "p_positive"]:
            base[column] = scored[column].to_numpy()
        filtered = base[base["sentiment"].isin(keep_sentiments)].copy()
    else:
        filtered = scored[scored["sentiment"].isin(keep_sentiments)].copy()
    if not keep_scores:
        filtered = filtered[[id_col, text_col]]
    return filtered.reset_index(drop=True)
