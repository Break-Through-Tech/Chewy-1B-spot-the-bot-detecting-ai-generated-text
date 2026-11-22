# Data

We do **not** commit the full dataset (~147 MB) to this public repo. Pull it from
Hugging Face with the script here, and use the small committed preview to see the
structure first.

| File | What it is |
|------|------------|
| `hc3_sample.csv` | A **committed** 42-row preview (balanced human/AI across all 5 domains) so you can inspect the format without downloading anything. Real HC3 rows, already "exploded" (one answer per row). |
| `download_hc3.py` | Pulls the **raw** full HC3 from Hugging Face and saves it as `hc3_all.csv`, exactly as published (no cleaning). Not committed — you generate it. |
| `hc3_all.csv` | The full **raw** dataset (one row per question) **you create** by running the script. Do your own preprocessing in your notebook. |

## Quick start

```bash
pip install "datasets>=2.14" pandas
python data/download_hc3.py     # writes data/hc3_all.csv
```

> Note: in the raw CSV, `human_answers` / `chatgpt_answers` are saved as
> stringified lists. Parse a column back with
> `df["human_answers"].apply(ast.literal_eval)` (`import ast`).

Or in Colab, load it directly:

```python
from datasets import load_dataset
ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
```

## Data dictionary

### Raw HC3 (as it comes from Hugging Face — the `all` config)

One row per **question**. The answer fields are **lists** (a question may have several human answers), which is why we explode them during prep.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Row identifier. |
| `question` | string | The question / prompt that was answered. |
| `human_answers` | list[string] | One or more human-written answers. |
| `chatgpt_answers` | list[string] | One or more ChatGPT-generated answers. |
| `source` | string | Domain: `finance`, `medicine`, `open_qa`, `reddit_eli5`, or `wiki_csai`. (Present only in the `all` config; the per-domain configs omit it because the domain is implied.) |
