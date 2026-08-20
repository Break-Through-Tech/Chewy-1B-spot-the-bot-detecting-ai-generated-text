"""
Download the HC3 (Human ChatGPT Comparison Corpus) dataset from Hugging Face.

Why this script exists
----------------------
The full HC3 dataset is ~147 MB, so we do NOT commit it to this public repo.
Run this script to pull the raw dataset from the Hugging Face Hub and save it to disk, exactly as published. Do your own preprocessing in your notebook.

A tiny 42-row preview (`data/hc3_sample.csv`) IS committed so you can eyeball the structure without downloading anything.

Raw structure (one row per question)
------------------------------------
    id, question, human_answers, chatgpt_answers, source
Heads-up: `human_answers` and `chatgpt_answers` are LISTS (a question can have
several answers).
"""
from datasets import load_dataset

OUT_PATH = "data/hc3_all.csv"


def main():
    # `trust_remote_code=True` is required: HC3 ships a small custom loading
    # script. If your `datasets` version complains, upgrade it, or load the
    # auto-generated Parquet mirror instead (see data/README.md).
    ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
    df = ds["train"].to_pandas()

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows (one per question) to {OUT_PATH}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
