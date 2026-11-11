#!/usr/bin/env python3
"""Summarize per-model TF-IDF error rates by response length, style, format, and terms."""
from collections import Counter
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/evaluation/tfidf_error_predictions.csv"
OUT = ROOT / "data/evaluation/tfidf_model_error_patterns.md"
MODELS = ["FFNN", "KNN", "Linear SVM", "Logistic Regression", "MultinomialNB"]
STOP = set("the a an and or but if then than to of in on at for with from by as is are was were be been being it this that these those i you we they he she my your our their me him her them do does did have has had can could would should will may might not no yes so very just about into over under what which who when where why how its also".split())


def flags(text):
    text = str(text)
    return {
        "Newline": "\n" in text,
        "List marker": bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)),
        "Markdown": bool(re.search(r"```|(?:^|\s)#{1,6}\s|\*\*|__", text)),
        "URL": bool(re.search(r"https?://|www\.", text, re.I)),
        "Code/formula": bool(re.search(r"```|`[^`]+`|[=<>]{2}|\$[^$]+\$", text)),
        "Repeated punctuation": bool(re.search(r"[!?.,;:]{2,}", text)),
        "Parentheses": "(" in text or ")" in text,
        "First person": bool(re.search(r"\b(i|we|my|our|me|us)\b", text, re.I)),
        "Second person": bool(re.search(r"\b(you|your|yours)\b", text, re.I)),
        "Contractions": bool(re.search(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m)\b", text, re.I)),
        "Hedging": bool(re.search(r"\b(maybe|perhaps|probably|possibly|might|could|seems?|appears?)\b", text, re.I)),
        "Conversational marker": bool(re.search(r"\b(yeah|well|honestly|basically|lol|imo|tbh|i mean|you know)\b", text, re.I)),
        "Question mark": "?" in text,
    }


def terms(frame, fail_mask, direction):
    base = frame[frame.target == direction]
    failures = fail_mask.loc[base.index]
    bad, good = Counter(), Counter()
    for text, failed in zip(base.text, failures):
        words = set(re.findall(r"[a-z]{3,}", str(text).lower())) - STOP
        (bad if failed else good).update(words)
    nb, ng = max(1, int(failures.sum())), max(1, int((~failures).sum()))
    scored = []
    for word, count in bad.items():
        if count >= 4:
            lift = ((count + .5) / nb) / ((good[word] + .5) / ng)
            scored.append((lift, word, count))
    return ", ".join(f"{w} ({n}, {lift:.1f}×)" for lift, w, n in sorted(scored, reverse=True)[:8]) or "No term reached minimum frequency"


def main():
    df = pd.read_csv(SOURCE).reset_index(drop=True)
    df["word_count"] = df.text.fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    df["length"] = pd.cut(df.word_count, [-1, 19, 99, 299, float("inf")], labels=["Very short (<20)", "Short (20–99)", "Medium (100–299)", "Very long (300+)"])
    feat = pd.DataFrame([flags(t) for t in df.text])
    lines = ["# Per-model error patterns", "", "Rates use the fixed held-out test set. For each model, a false positive is a human response predicted AI; a false negative is an AI response predicted human. Length and style rates are observational and can reflect topic/source mix. Vocabulary terms are enriched among a model's errors versus correctly classified responses of the same true class; minimum four error examples, ranked by document-frequency lift.", ""]
    for model in MODELS:
        pred = df[model].astype(int)
        fp = (df.target.eq(0) & pred.eq(1))
        fn = (df.target.eq(1) & pred.eq(0))
        err = fp | fn
        lines += [f"## {model}", "", f"Errors: {int(err.sum()):,}/{len(df):,} ({err.mean():.1%}); false positives: {int(fp.sum()):,}/{int(df.target.eq(0).sum()):,} humans ({fp.sum()/max(1,df.target.eq(0).sum()):.1%}); false negatives: {int(fn.sum()):,}/{int(df.target.eq(1).sum()):,} AI ({fn.sum()/max(1,df.target.eq(1).sum()):.1%}).", "", "### Length", "", "| Length | n | Errors | FP | FN |", "|---|---:|---:|---:|---:|"]
        for band, g in df.groupby("length", observed=False):
            if len(g):
                e = err.loc[g.index]; p = fp.loc[g.index]; n = fn.loc[g.index]
                lines.append(f"| {band} | {len(g):,} | {e.sum():,} ({e.mean():.1%}) | {p.sum():,} | {n.sum():,} |")
        lines += ["", "### Writing style and formatting", "", "| Pattern present | n | Error rate | FP | FN |", "|---|---:|---:|---:|---:|"]
        for name in feat.columns:
            mask = feat[name].astype(bool)
            n = int(mask.sum())
            if n >= 10:
                e = err[mask]; p = fp[mask]; f = fn[mask]
                lines.append(f"| {name} | {n:,} | {e.mean():.1%} | {p.sum():,} | {f.sum():,} |")
        lines += ["", "### Vocabulary associated with errors", "", f"- False positives: {terms(df, fp, 0)}", f"- False negatives: {terms(df, fn, 1)}", ""]
    lines += ["## Reading this report", "", "A high error rate in a category flags a review priority; it does not show the feature caused the error. Style and format indicators can overlap, and small groups are unstable. Use the response-level CSV to inspect examples and source/domain before drawing conclusions.", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
