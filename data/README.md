# Data

We do **not** commit the full HC3 dataset (about 147 MB) to this public repo. Pull the raw combined dataset with the script here and use the small committed preview to inspect its structure first.

| File | What it is |
|------|------------|
| `hc3_sample.csv` | A committed 42-row preview (balanced human/AI across the five domains), already expanded to one answer per row. |
| `download_hc3.py` | Downloads the pinned HC3 Parquet snapshot's combined `all` file and saves it as `hc3_all.csv`. |
| `hc3_all.csv` | The generated raw dataset: one row per question, with nested human and ChatGPT answer lists. This file is not committed. |

## Quick start

From the repository root:

```bash
python -m pip install "datasets>=2.14" pandas
python data/download_hc3.py
```

The script loads only `all/train/0000.parquet` from a fixed Hugging Face commit. The Hub repository also contains a separate Parquet file for each domain; loading the entire repository would include those rows a second time. The script checks the expected 24,322 rows, schema, and unique question IDs before writing the CSV.

In the CSV, `human_answers` and `chatgpt_answers` are serialized lists. Parse them with `ast.literal_eval` after importing `ast`.

## Data dictionary

### Raw HC3 (combined `all` file)

One row per question. The answer fields are lists because a question may have several human answers.

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Stable row ID in the pinned snapshot. |
| `question` | string | Question or prompt that was answered. |
| `human_answers` | list[string] | One or more human-written answers. |
| `chatgpt_answers` | list[string] | One or more ChatGPT-generated answers. |
| `source` | string | Domain: `finance`, `medicine`, `open_qa`, `reddit_eli5`, or `wiki_csai`. |
