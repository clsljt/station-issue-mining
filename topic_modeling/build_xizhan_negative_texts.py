"""Build per-note aggregates of negative Beijing West Station texts.

The script reads the two Excel exports and image-caption JSON under
``data/XiZhan``.  It cleans each text with the project's existing cleaner,
scores every remaining note title, note body, comment, and image caption with
the local sentiment model, keeps only ``negative`` rows, and aggregates them
by note ID. Negative comments are retained only when their note body is also
negative.

Example
-------
conda run -n py312 python scripts/build_xizhan_negative_texts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "XiZhan"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "0623" / "xizhan_negative_texts.csv"

# Reuse the project's established cleaning and sentiment implementations.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from clean_station_data import clean_text_id_dataframe  # noqa: E402
from sentiment_filter import analyze_sentiment_df  # noqa: E402


def _as_text(value: object) -> str:
    """Return a safe text value for exported cells that may be empty."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_note_texts(notes_path: Path) -> pd.DataFrame:
    """Return title and body as separately scoreable texts for each note."""
    notes = pd.read_excel(notes_path, dtype={"note_id": str})
    required = {"note_id", "title", "desc"}
    missing = required - set(notes.columns)
    if missing:
        raise KeyError(f"{notes_path.name} is missing columns: {sorted(missing)}")

    ids = notes["note_id"].map(_as_text)
    return pd.concat(
        [
            pd.DataFrame({"id": ids, "text": notes["title"].map(_as_text), "kind": "title"}),
            pd.DataFrame({"id": ids, "text": notes["desc"].map(_as_text), "kind": "content"}),
        ],
        ignore_index=True,
    )


def load_note_metrics(notes_path: Path) -> pd.DataFrame:
    """Return the engagement metrics attached to each original note."""
    notes = pd.read_excel(notes_path, dtype={"note_id": str})
    metric_columns = ["note_id", "liked_count", "comment_count", "share_count", "upload_time"]
    missing = set(metric_columns) - set(notes.columns)
    if missing:
        raise KeyError(f"{notes_path.name} is missing columns: {sorted(missing)}")
    metrics = notes[metric_columns].copy().rename(columns={"note_id": "id"})
    metrics["id"] = metrics["id"].map(_as_text)
    for column in ["liked_count", "comment_count", "share_count"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce").fillna(0).astype(int)
    metrics["upload_time"] = pd.to_datetime(metrics["upload_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    metrics["upload_time"] = metrics["upload_time"].fillna("")
    return metrics


def load_comment_texts(comments_path: Path) -> pd.DataFrame:
    """Return one text per comment, keyed by its parent note ID."""
    comments = pd.read_excel(comments_path, dtype={"note_id": str})
    required = {"note_id", "content"}
    missing = required - set(comments.columns)
    if missing:
        raise KeyError(f"{comments_path.name} is missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {"id": comments["note_id"].map(_as_text), "text": comments["content"].map(_as_text), "kind": "comment"}
    )


def load_caption_texts(captions_path: Path) -> pd.DataFrame:
    """Return one image-caption text per note ID from ``note_id/image_name`` keys."""
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    if not isinstance(captions, dict):
        raise ValueError(f"{captions_path.name} must be a JSON object mapping image paths to captions")

    rows = []
    for image_path, caption in captions.items():
        note_id = _as_text(image_path).split("/", maxsplit=1)[0]
        if note_id:
            rows.append({"id": note_id, "text": _as_text(caption), "kind": "fig"})
    return pd.DataFrame(rows, columns=["id", "text", "kind"])


def score_negative_texts(texts: pd.DataFrame, batch_size: int, device: str) -> pd.DataFrame:
    """Clean and retain only model-negative texts, excluding short comments."""
    clean_input = texts.rename(columns={"text": "content"}).copy()
    clean_input["row_id"] = range(len(clean_input))
    # Use a temporary unique ID so rows removed by the cleaner can be mapped
    # back to their original note ID and source type without index ambiguity.
    cleaned = clean_text_id_dataframe(clean_input[["row_id", "content"]], id_col="row_id")
    cleaned["row_id"] = cleaned["id"].astype(int)
    source = clean_input.set_index("row_id")
    cleaned["id"] = cleaned["row_id"].map(source["id"])
    cleaned["kind"] = cleaned["row_id"].map(source["kind"])
    # Very short comments are usually acknowledgements or conversational
    # fragments; exclude them after cleaning, so emojis and XHS markup do not
    # count toward the four-character minimum. Notes and figure captions stay.
    cleaned = cleaned.loc[
        ~((cleaned["kind"].eq("comment")) & (cleaned["content"].str.len() < 4))
    ].reset_index(drop=True)
    scored = analyze_sentiment_df(
        cleaned,
        batch_size=batch_size,
        negative_threshold=0.55,
        positive_threshold=0.65,
        device=device,
        show_progress=True,
    )
    scored["kind"] = cleaned["kind"].to_numpy()
    return scored.loc[scored["sentiment"].eq("negative"), ["id", "content", "kind"]].reset_index(drop=True)


def _json_list(values: pd.Series) -> str:
    """Serialize non-empty unique text values as a CSV-safe list representation."""
    unique = list(dict.fromkeys(value for value in values.astype(str) if value.strip()))
    return json.dumps(unique, ensure_ascii=False)


def _single_text(values: pd.Series) -> str:
    """Return the first non-empty value for a one-text-per-note field."""
    return next((value for value in values.astype(str) if value.strip()), "")


def aggregate_by_id(negative: pd.DataFrame) -> pd.DataFrame:
    """Build id/title/content/comment/fig, gating comments on negative bodies."""
    negative_content_ids = set(negative.loc[negative["kind"].eq("content"), "id"])
    negative = negative.loc[
        ~negative["kind"].eq("comment") | negative["id"].isin(negative_content_ids)
    ].copy()
    single_texts = (
        negative.loc[negative["kind"].isin(["title", "content"])]
        .groupby(["id", "kind"], sort=True)["content"]
        .agg(_single_text)
        .unstack("kind")
    )
    list_texts = (
        negative.loc[negative["kind"].isin(["comment", "fig"])]
        .groupby(["id", "kind"], sort=True)["content"]
        .agg(_json_list)
        .unstack("kind")
    )
    grouped = single_texts.join(list_texts, how="outer").reset_index()
    for column in ["title", "content"]:
        if column not in grouped:
            grouped[column] = ""
    for column in ["comment", "fig"]:
        if column not in grouped:
            grouped[column] = "[]"
    grouped[["title", "content"]] = grouped[["title", "content"]].fillna("")
    grouped[["comment", "fig"]] = grouped[["comment", "fig"]].fillna("[]")
    grouped = grouped.loc[grouped["content"].ne("")]
    return grouped[["id", "title", "content", "comment", "fig"]].sort_values("id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=DEFAULT_DATA_DIR / "XiZhan_notes.xlsx")
    parser.add_argument("--comments", type=Path, default=DEFAULT_DATA_DIR / "XiZhan_mments.xlsx")
    parser.add_argument("--captions", type=Path, default=DEFAULT_DATA_DIR / "xizhan_media_image_captions.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [args.notes, args.comments, args.captions]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")

    texts = pd.concat(
        [load_note_texts(args.notes), load_comment_texts(args.comments), load_caption_texts(args.captions)],
        ignore_index=True,
    )
    negative = score_negative_texts(texts, batch_size=args.batch_size, device=args.device)
    result = aggregate_by_id(negative)
    result = result.merge(load_note_metrics(args.notes), on="id", how="left", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Scored texts: {len(texts):,}")
    print(f"Negative texts: {len(negative):,}")
    print(f"Output IDs: {len(result):,}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
