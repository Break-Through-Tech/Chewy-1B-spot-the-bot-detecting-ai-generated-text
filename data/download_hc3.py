"""Download the raw, combined HC3 dataset from its pinned Parquet snapshot.

The snapshot contains an ``all`` Parquet file plus one file per domain. Load only
``all/train/0000.parquet``: loading the dataset repository without an explicit
file also reads the five domain copies and duplicates every question.
"""
from pathlib import Path

from datasets import load_dataset

DATASET_REPO = "Hello-SimpleAI/HC3"
DATASET_REVISION = "7444b1cb1469246e7b5d191b03f69b7e7c951134"
DATASET_FILE = (
    f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
    f"{DATASET_REVISION}/all/train/0000.parquet"
)
EXPECTED_ROWS = 24_322
EXPECTED_COLUMNS = ["id", "question", "human_answers", "chatgpt_answers", "source"]
OUT_PATH = Path(__file__).resolve().with_name("hc3_all.csv")


def main():
    dataset = load_dataset(
        "parquet",
        data_files={"train": DATASET_FILE},
        split="train",
    )
    df = dataset.to_pandas()

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected HC3 columns: {list(df.columns)}; expected {EXPECTED_COLUMNS}"
        )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} combined HC3 questions, got {len(df):,}. "
            "Check that only the pinned all/train Parquet file is loaded."
        )
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("HC3 question IDs must be present and unique")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows (one per question) to {OUT_PATH}")
    print(df["source"].value_counts().sort_index())


if __name__ == "__main__":
    main()
