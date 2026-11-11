#!/usr/bin/env python3
"""Evaluate five TF-IDF classifiers and profile held-out errors and shared misses."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import vstack


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluation"
MODEL_NAMES = ["FFNN", "KNN", "Linear SVM", "Logistic Regression", "MultinomialNB"]


def formatting(text: str) -> dict[str, int]:
    return {
        "has_newline": int("\n" in text),
        "has_list_marker": int(bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text))),
        "has_markdown": int(bool(re.search(r"```|(?:^|\s)#{1,6}\s|\*\*|__", text))),
        "has_url": int(bool(re.search(r"https?://|www\.", text, re.I))),
        "has_code_or_formula": int(bool(re.search(r"```|`[^`]+`|[=<>]{2}|\$[^$]+\$", text))),
        "has_repeated_punctuation": int(bool(re.search(r"[!?.,;:]{2,}", text))),
        "has_parentheses": int("(" in text or ")" in text),
        "first_person": int(bool(re.search(r"\b(i|we|my|our|me|us)\b", text, re.I))),
        "second_person": int(bool(re.search(r"\b(you|your|yours)\b", text, re.I))),
        "contractions": int(bool(re.search(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m)\b", text, re.I))),
        "hedging_language": int(bool(re.search(r"\b(maybe|perhaps|probably|possibly|might|could|seems?|appears?)\b", text, re.I))),
        "conversational_marker": int(bool(re.search(r"\b(yeah|well|honestly|basically|lol|imo|tbh|i mean|you know)\b", text, re.I))),
        "question_mark": int("?" in text),
    }


def length_band(n: int) -> str:
    if n < 20:
        return "Very short (<20 words)"
    if n < 100:
        return "Short (20–99 words)"
    if n < 300:
        return "Medium (100–299 words)"
    return "Very long (300+ words)"


def top_terms(texts: pd.Series, labels: pd.Series, positive: int, limit: int = 12) -> list[tuple[str, int]]:
    # Compare failed cases against correctly classified cases of the same true class.
    stop = set("the a an and or but if then than to of in on at for with from by as is are was were be been being it this that these those i you we they he she my your our their me him her them do does did have has had can could would should will may might not no yes so very just about into over under what which who when where why how its also".split())
    pos, neg = Counter(), Counter()
    for text, label in zip(texts, labels):
        tokens = set(re.findall(r"[a-z]{3,}", str(text).lower())) - stop
        (pos if int(label) == positive else neg).update(tokens)
    npos, nneg = max(1, int((labels == positive).sum())), max(1, int((labels != positive).sum()))
    ranked = []
    for word, count in pos.items():
        if count < 3:
            continue
        other = neg[word]
        lift = ((count + 0.5) / npos) / ((other + 0.5) / nneg)
        ranked.append((word, count, lift))
    return [(w, c) for w, c, _ in sorted(ranked, key=lambda x: (x[2], x[1]), reverse=True)[:limit]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data" / "processed"
    train, val, test = [pd.read_csv(data / f"{s}.csv") for s in ("train", "validation", "test")]
    vec_path = ROOT / "backend" / "models" / "tfidf_vectorizer.joblib"
    vectorizer = joblib.load(vec_path)
    Xtr, Xv, Xt = [vectorizer.transform(df.text.fillna("").astype(str)) for df in (train, val, test)]
    ytr, yv, yt = [df.target.astype(int).to_numpy() for df in (train, val, test)]
    Xfit, yfit = vstack([Xtr, Xv]), np.concatenate([ytr, yv])
    predictions: dict[str, np.ndarray] = {}

    # The checked-in FFNN is the project model. Predict in batches to bound dense memory use.
    from tensorflow.keras.models import load_model
    net = load_model(ROOT / "backend" / "models" / "feedforward_neural_network.keras")
    batches = []
    for start in range(0, Xt.shape[0], 64):
        batches.extend(np.asarray(net.predict(Xt[start:start + 64].toarray(), verbose=0)).reshape(-1))
    predictions["FFNN"] = (np.asarray(batches) >= 0.5).astype(int)

    models = {
        # The checked-in KNN notebook's executed result uses best_k=275.
        "KNN": KNeighborsClassifier(n_neighbors=275, weights="distance", metric="cosine", algorithm="brute", n_jobs=-1),
        "Linear SVM": LinearSVC(C=1, class_weight="balanced", dual="auto", max_iter=10000, random_state=42),
        "Logistic Regression": LogisticRegression(C=50, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=0.01, fit_prior=True),
    }
    for name, clf in models.items():
        clf.fit(Xfit, yfit)
        if name == "KNN":
            predictions[name] = (clf.predict_proba(Xt)[:, 1] >= 0.475).astype(int)
        else:
            predictions[name] = clf.predict(Xt).astype(int)

    records = test[["id", "source", "text", "label", "target"]].copy().reset_index(drop=True)
    for name, pred in predictions.items():
        records[name] = pred
    records["error_count"] = (records[MODEL_NAMES].to_numpy() != yt[:, None]).sum(axis=1)
    records["all_wrong"] = records.error_count.eq(len(MODEL_NAMES))
    records["word_count"] = records.text.fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    records["length_band"] = records.word_count.map(length_band)
    records["error_type"] = np.where((yt == 0) & (records[MODEL_NAMES].to_numpy() == 1).any(axis=1), "False positive by ≥1 model", "")
    records.loc[(yt == 1) & (records[MODEL_NAMES].to_numpy() == 0).any(axis=1), "error_type"] = "False negative by ≥1 model"
    records.to_csv(out / "tfidf_error_predictions.csv", index=False)

    metrics = []
    for name in MODEL_NAMES:
        p = predictions[name]
        tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel()
        metrics.append({"model": name, "accuracy": accuracy_score(yt, p), "macro_precision": precision_score(yt, p, average="macro", zero_division=0), "macro_recall": recall_score(yt, p, average="macro", zero_division=0), "macro_f1": f1_score(yt, p, average="macro", zero_division=0), "false_positives": int(fp), "false_negatives": int(fn), "true_negatives": int(tn), "true_positives": int(tp)})
    pd.DataFrame(metrics).to_csv(out / "tfidf_error_metrics.csv", index=False)

    profiles = records.copy()
    for k, v in pd.DataFrame([formatting(str(t)) for t in profiles.text]).items():
        profiles[k] = v
    wrong_any = profiles.error_count.gt(0)
    allright = ~wrong_any
    lines = ["# TF-IDF shared-error and robustness review", "", "## Scope and method", "", f"Analysis uses the fixed held-out test split ({len(test):,} answers; {int((yt==0).sum()):,} human and {int((yt==1).sum()):,} AI). TF-IDF uses the checked-in vectorizer. FFNN uses the checked-in Keras model; the other four classifiers are fit on train + validation. Settings follow executed notebook configurations where available: KNN k=275 and threshold 0.475, Linear SVM C=1, Logistic Regression C=50, and MultinomialNB alpha=0.01. The KNN value differs from the prose in `MODEL_COMPARISON.md`, which says k=400. These are benchmark diagnostics, not evidence of real-world authorship detection.", "", "## Held-out results", "", "| Model | Accuracy | Macro F1 | False positives | False negatives |", "|---|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['model']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['false_positives']:,} | {row['false_negatives']:,} |")
    lines += ["", "## Shared mistakes", "", f"{int(records.all_wrong.sum()):,} examples are misclassified by all five models. The complete set is in `tfidf_error_predictions.csv`; a compact sample of up to 50 examples is in `tfidf_shared_mistakes.csv`. Error count means number of models wrong (0–5).", ""]
    shared = records.loc[records.all_wrong].copy()
    shared.head(50).to_csv(out / "tfidf_shared_mistakes.csv", index=False)
    for count in range(5, 0, -1):
        lines.append(f"- Wrong for {count}/5 models: {(records.error_count == count).sum():,}")
    lines += ["", "## Length and formatting patterns", "", "Rates below are the share of examples in each bucket misclassified by at least one model, compared with the overall any-model error rate. Small buckets can be noisy.", "", f"Overall any-model error rate: {wrong_any.mean():.1%}; all-five error rate: {records.all_wrong.mean():.1%}.", "", "| Pattern | n | Any-model error rate | All-five error rate |", "|---|---:|---:|---:|"]
    feature_cols = ["length_band", "has_newline", "has_list_marker", "has_markdown", "has_url", "has_code_or_formula", "has_repeated_punctuation", "has_parentheses", "first_person", "second_person", "contractions", "hedging_language", "conversational_marker", "question_mark"]
    for col in feature_cols:
        groups = profiles.groupby(col, dropna=False)
        for key, g in groups:
            if len(g) >= 10:
                lines.append(f"| {col.replace('_',' ')} = {key} | {len(g):,} | {g.error_count.gt(0).mean():.1%} | {g.all_wrong.mean():.1%} |")
    lines += ["", "## Error direction and vocabulary clues", "", "False positives are human answers labeled AI; false negatives are AI answers labeled human. These lexical clues are descriptive associations among errors with minimum frequency three, not causal explanations or stable authorship signals.", ""]
    for label, true_class in [("False positives (human → AI)", 0), ("False negatives (AI → human)", 1)]:
        same_class = profiles[profiles.target == true_class]
        error_mask = same_class.error_count.gt(0).astype(int)
        subset = same_class[error_mask == 1]
        terms = top_terms(same_class.text, error_mask, 1)
        lines.append(f"### {label}")
        lines.append(f"- Examples: {len(subset):,}; median word count: {subset.word_count.median():.0f}.")
        lines.append("- Frequent content words in this error group: " + (", ".join(f"`{w}` ({n})" for w, n in terms) if terms else "not enough repeated terms"))
        lines.append("")
    lines += ["## Sample unanimous errors", "", "Examples below were misclassified by all five models. Inspect the CSV for full responses.", ""]
    for _, row in shared.head(5).iterrows():
        excerpt = re.sub(r"\s+", " ", str(row.text))[:220].replace("|", "\\|")
        lines.append(f"- Test id `{row.id}`; true label `{row.label}`; {int(row.word_count)} words; source `{row.source}`: “{excerpt}…”")
    lines += ["", "## Source breakdown", "", "| Source | n | Any-model errors | False positives | False negatives |", "|---|---:|---:|---:|---:|"]
    for source, group in profiles.groupby("source", dropna=False):
        any_error = group.error_count.gt(0)
        fp = ((group.target == 0) & any_error).sum()
        fn = ((group.target == 1) & any_error).sum()
        lines.append(f"| {source} | {len(group):,} | {any_error.sum():,} ({any_error.mean():.1%}) | {fp:,} | {fn:,} |")
    lines.append("")
    lines += ["## Robustness assessment", "", "Do not use these models as standalone evidence that a response was AI-written. This evaluation is an HC3-derived, fixed test split and can reflect benchmark topic, source, preprocessing, and collection artifacts. Shared errors mark examples where ensembling by agreement offers no protection. Error-rate differences by length or formatting are observational; they do not establish causation. Before use beyond this benchmark, evaluate a time-separated and domain-separated external set, report class-specific false-positive rates at the intended threshold, inspect false positives and false negatives with human reviewers, and measure calibration and subgroup variation.", "", "## Output files", "", "- `tfidf_error_metrics.csv`: held-out metrics and confusion counts by model.", "- `tfidf_error_predictions.csv`: every test response, label, all five predictions, error count/type, and length band.", "- `tfidf_shared_mistakes.csv`: up to 50 examples misclassified by every model.", "- `tfidf_error_analysis.md`: this report."]
    (out / "tfidf_error_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote analysis artifacts to {out}")


if __name__ == "__main__":
    main()
