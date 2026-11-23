
"""
Export README-ready figures for Spot the Bot.

Run from the repository root:
    python scripts/export_readme_figures.py

Figures:
1. HC3 vs modern macro F1 comparison (grouped bars)
2. HC3 vs modern RoBERTa confusion matrices
3. Modern external ROC curves
4. Modern RoBERTa prediction score distributions
5. HC3 performance by answer length (two-panel line chart)

Uses saved evaluation results. No model retraining.

For the length plot, first export prediction_rows from
notebooks/evaluation/hc3_model_comparison.ipynb:

    pd.concat(prediction_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "hc3_test_predictions.csv",
        index=False
    )

Outputs: docs/images/results/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
TEST = ROOT / "data" / "processed" / "test.csv"
OUTPUT = ROOT / "docs" / "images" / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)

DPI = 200


# ---------------------------------------------------------
# Spot the Bot — Shared Visualization Theme
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "hc3": "#7DA9D9",          # Soft blue
    "modern": "#F4A261",       # Warm coral
    "roberta": "#7DA9D9",      # Soft blue
    "ffnn": "#8BCF9B",         # Sage green
    "tfidf_ffnn": "#8BCF9B",
    "tfidf_other": "#F2C879",  # Pastel gold
    "embedding": "#C6A0DC",    # Lavender
    "human": "#89B4E0",        # Pastel blue
    "ai": "#F4A261",           # Soft coral
}

# Use one consistent sequential color scale
HEATMAP_CMAP = "YlGnBu"

# Individual model colors for multi-model plots
MODEL_COLORS = {
    "RoBERTa": COLORS["roberta"],
    "TF-IDF FFNN": COLORS["tfidf_ffnn"],
    "TF-IDF Linear SVM": "#E8B45B",
    "TF-IDF Logistic Regression": "#D99A5B",
    "TF-IDF Multinomial Naive Bayes": "#BFA56C",
    "TF-IDF KNN": "#D9C79E",
    "Embeddings FFNN": "#A98BCC",
    "Embeddings Linear SVM": "#C6A0DC",
    "Embeddings Logistic Regression": "#9074B5",
}

sns.set_theme(
    style="white",
    context="notebook",
    font_scale=1.0
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
})

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": DPI,
})

MODEL_ORDER = [
    "RoBERTa",
    "TF-IDF FFNN",
    "TF-IDF Linear SVM",
    "TF-IDF Logistic Regression",
    "Embeddings FFNN",
    "Embeddings Linear SVM",
    "Embeddings Logistic Regression",
]

SHORT_NAMES = {
    "RoBERTa": "RoBERTa",
    "TF-IDF FFNN": "TF-IDF FFNN",
    "TF-IDF Linear SVM": "TF-IDF SVM",
    "TF-IDF Logistic Regression": "TF-IDF LR",
    "Embeddings FFNN": "Embedding FFNN",
    "Embeddings Linear SVM": "Embedding SVM",
    "Embeddings Logistic Regression": "Embedding LR",
}


def save_figure(fig, filename):
    path = OUTPUT / filename
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        print(f"SKIP: Missing {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------
# 1. HC3 vs Modern: Grouped horizontal bar chart
# ---------------------------------------------------------

def plot_hc3_vs_modern():
    hc3 = load_csv("hc3_model_metrics.csv")
    modern = load_csv("modern_seven_model_comparison.csv")

    if hc3 is None or modern is None:
        return

    hc3 = hc3.set_index("model")
    modern = modern.set_index("model")

    models = [
        m for m in MODEL_ORDER
        if m in hc3.index and m in modern.index
    ]

    if not models:
        raise ValueError("No matching model names.")

    hc3_scores = hc3.loc[models, "macro_f1"].to_numpy()
    modern_scores = modern.loc[models, "macro_f1"].to_numpy()

    y = np.arange(len(models))
    height = 0.34

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.barh(
        y - height / 2,
        hc3_scores,
        height,
        color=COLORS["hc3"],
        label="HC3 held-out test",
    )

    ax.barh(
        y + height / 2,
        modern_scores,
        height,
        color=COLORS["modern"],
        label="Modern external",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_NAMES[m] for m in models])
    ax.invert_yaxis()

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro F1")
    ax.set_title(
        "Model Generalization: HC3 vs Modern External Data",
        weight="bold",
        pad=18,
    )

    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)

    for i, (original, external) in enumerate(
        zip(hc3_scores, modern_scores)
    ):
        ax.text(
            original + 0.012,
            i - height / 2,
            f"{original:.3f}",
            va="center",
            fontsize=9,
        )
        ax.text(
            external + 0.012,
            i + height / 2,
            f"{external:.3f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "hc3_vs_modern_macro_f1.png")



# ---------------------------------------------------------
# HC3 vs Modern: Distribution Shift Confusion Matrices
# ---------------------------------------------------------

def plot_distribution_shift_confusion_matrices():
    hc3 = load_csv("hc3_model_metrics.csv")

    if hc3 is None:
        return

    # Model configurations
    models = [
        {
            "name": "RoBERTa",
            "modern_file": "modern_roberta_predictions.csv",
            "prediction_col": "roberta_prediction",
            "output": "roberta_hc3_vs_modern_confusion.png",
        },
        {
            "name": "TF-IDF FFNN",
            "modern_file": "modern_tfidf_predictions.csv",
            "prediction_col": "tf_idf_ffnn_prediction",
            "output": "tfidf_ffnn_hc3_vs_modern_confusion.png",
        },
    ]

    for config in models:

        modern = load_csv(config["modern_file"])

        if modern is None:
            continue

        # Restrict modern evaluation to human + ChatGPT
        modern = modern[
            modern["model"].isin(["human", "chatgpt"])
        ].copy()

        # HC3 confusion matrix from saved model metrics
        row = hc3.loc[
            hc3["model"] == config["name"]
        ]

        if row.empty:
            print(f"SKIP: Missing {config['name']} HC3 metrics")
            continue

        row = row.iloc[0]

        hc3_cm = np.array([
            [int(row["tn"]), int(row["fp"])],
            [int(row["fn"]), int(row["tp"])],
        ])

        # Modern external confusion matrix
        modern_cm = confusion_matrix(
            modern["target"],
            modern[config["prediction_col"]],
            labels=[0, 1],
        )

        # Create figure with automatic spacing
        fig, axes = plt.subplots(
            1, 2,
            figsize=(11.5, 5.3),
            layout="constrained",
            facecolor="white",
        )

        titles = [
            "HC3 Held-Out Test",
            "Modern External Dataset",
        ]

        for ax, cm, title in zip(
            axes,
            [hc3_cm, modern_cm],
            titles,
        ):

            # Normalize by actual class
            normalized = (
                cm / cm.sum(axis=1, keepdims=True)
            )

            im = ax.imshow(
                normalized,
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )

            # Remove unwanted gridlines
            ax.grid(False, which="both")
            ax.minorticks_off()

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels(["Human", "AI"])
            ax.set_yticklabels(["Human", "AI"])

            ax.set_xlabel("Predicted label", labelpad=8)

            # No individual y-axis labels:
            # We'll use one shared label instead.
            ax.set_ylabel("")

            ax.set_title(
                title,
                fontsize=13,
                fontweight="semibold",
                pad=14,
            )

            ax.tick_params(length=0)

            # Display raw counts and row percentages
            for i in range(2):
                for j in range(2):

                    value = normalized[i, j]

                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:,}\n({value:.1%})",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=(
                            "white"
                            if value > 0.65
                            else "#263238"
                        ),
                    )

            for spine in ax.spines.values():
                spine.set_visible(False)

        # Shared y-axis label — prevents overlap
        fig.supylabel(
            "Actual label",
            fontsize=12,
            x=0.01,
        )

        # Shared colorbar
        fig.colorbar(
            im,
            ax=axes,
            shrink=0.78,
            pad=0.025,
            label="Proportion within actual class",
        )

        fig.suptitle(
            f"{config['name']}: HC3 vs Modern External Dataset",
            fontsize=16,
            fontweight="bold",
        )

        save_figure(fig, config["output"])



# ---------------------------------------------------------
# Modern External ROC Curves — All AI vs ChatGPT Only
# ---------------------------------------------------------

def plot_modern_roc():
    roberta = load_csv("modern_roberta_predictions.csv")
    tfidf = load_csv("modern_tfidf_predictions.csv")
    embeddings = load_csv("modern_embeddings_predictions.csv")

    if any(df is None for df in [roberta, tfidf, embeddings]):
        return

    score_columns = {
        "RoBERTa": (roberta, "roberta_score"),
        "TF-IDF FFNN": (tfidf, "tf_idf_ffnn_score"),
        "TF-IDF Linear SVM": (
            tfidf, "tf_idf_linear_svm_score"
        ),
        "TF-IDF Logistic Regression": (
            tfidf, "tf_idf_logistic_regression_score"
        ),
        "Embeddings FFNN": (
            embeddings, "embeddings_ffnn_score"
        ),
        "Embeddings Linear SVM": (
            embeddings, "embeddings_linear_svm_score"
        ),
        "Embeddings Logistic Regression": (
            embeddings, "embeddings_logistic_regression_score"
        ),
    }

    # Use the same model colors in both plots
    colors = {
        "RoBERTa": "#7DA9D9",
        "TF-IDF FFNN": "#8BCF9B",
        "TF-IDF Linear SVM": "#E8B45B",
        "TF-IDF Logistic Regression": "#D99A5B",
        "Embeddings FFNN": "#A98BCC",
        "Embeddings Linear SVM": "#C6A0DC",
        "Embeddings Logistic Regression": "#9074B5",
    }

    # Two evaluation subsets
    subsets = [
        (
            "All Modern AI Models",
            ["human", "chatgpt", "claude", "gemini"],
        ),
        (
            "Modern ChatGPT Only",
            ["human", "chatgpt"],
        ),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    summary = []

    for ax, (title, included_sources) in zip(
        axes, subsets
    ):

        for model_name, (df, score_col) in score_columns.items():

            # Filter each model's existing predictions
            subset = df.loc[
                df["model"].isin(included_sources)
            ].copy()

            y_true = subset["target"].to_numpy()
            y_score = subset[score_col].to_numpy()

            # Calculate ROC and AUC for this subset
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            summary.append({
                "evaluation": title,
                "model": model_name,
                "roc_auc": auc,
            })

            ax.plot(
                fpr,
                tpr,
                linewidth=2.2,
                color=colors[model_name],
                label=f"{SHORT_NAMES[model_name]} ({auc:.3f})",
            )

        # Random classifier reference
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#999999",
            linewidth=1.2,
            alpha=0.8,
        )

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            pad=14,
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        # Clean, subtle grid
        ax.grid(alpha=0.12)

        ax.legend(
            loc="lower right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#DDDDDD",
        )

    fig.suptitle(
        "Modern External Evaluation — ROC Curve Comparison",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "modern_roc_curves.png",
    )

    # Also save a numeric comparison for the README
    summary_df = pd.DataFrame(summary)

    comparison = summary_df.pivot(
        index="model",
        columns="evaluation",
        values="roc_auc",
    )

    comparison = comparison.reindex(MODEL_ORDER)

    comparison.to_csv(
        DATA / "modern_roc_auc_comparison.csv"
    )

    print("\nModern ROC-AUC Comparison:")
    print(comparison.round(4).to_string())

# ---------------------------------------------------------
# 4. Modern RoBERTa prediction score distributions
# ---------------------------------------------------------

def plot_modern_score_distributions():
    df = load_csv("modern_roberta_predictions.csv")

    if df is None:
        return

    human = df.loc[
        df["target"] == 0, "roberta_score"
    ].to_numpy()

    ai = df.loc[
        df["target"] == 1, "roberta_score"
    ].to_numpy()

    # Log scale reveals differences when scores are
    # strongly concentrated near zero.
    epsilon = 1e-8

    human_log = np.log10(np.clip(human, epsilon, 1))
    ai_log = np.log10(np.clip(ai, epsilon, 1))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bins = np.linspace(-8, 0, 49)

    ax.hist(
        human_log,
        bins=bins,
        color=COLORS["human"],
        alpha=0.65,
        label=f"Human (n={len(human)})",
    )

    ax.hist(
        ai_log,
        bins=bins,
        color=COLORS["ai"],
        alpha=0.55,
        label=f"AI (n={len(ai)})",
    )

    ax.axvline(
        np.log10(0.5),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Original 0.5 threshold",
    )

    ax.set_xlabel("Predicted AI score (log10 scale)")
    ax.set_ylabel("Number of responses")
    ax.set_title(
        "RoBERTa Prediction Scores on Modern External Data",
        weight="bold",
        pad=15,
    )

    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, "modern_roberta_score_distribution.png")


# ---------------------------------------------------------
# 5. HC3 performance by answer length
# ---------------------------------------------------------

def plot_length_performance():
    predictions = load_csv("hc3_test_predictions.csv")

    if predictions is None:
        print(
            "Run hc3_model_comparison.ipynb and export "
            "prediction_rows before generating the length plot."
        )
        return

    if not TEST.exists():
        print("SKIP length plot: missing test.csv")
        return

    test = pd.read_csv(TEST)

    required = {"test_row", "model", "target", "prediction"}
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    selected_models = ["RoBERTa", "TF-IDF FFNN"]

    predictions = predictions.loc[
        predictions["model"].isin(selected_models)
    ].copy()

    if predictions.empty:
        raise ValueError("Selected models not found.")

    # Match predictions to the original HC3 test rows.
    test["word_count"] = (
        test["text"]
        .fillna("")
        .astype(str)
        .str.findall(r"\b[\w']+\b")
        .str.len()
    )

    bins = [0, 50, 100, 200, 400, np.inf]
    labels = [
        "0–49",
        "50–99",
        "100–199",
        "200–399",
        "400+",
    ]

    test["length_bin"] = pd.cut(
        test["word_count"],
        bins=bins,
        labels=labels,
        right=False,
    )

    predictions = predictions.merge(
        test[["target", "length_bin"]].rename(
            columns={"target": "expected_target"}
        ),
        left_on="test_row",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    if predictions["expected_target"].isna().any():
        raise ValueError("Unmatched prediction rows.")

    if not (
        predictions["target"] == predictions["expected_target"]
    ).all():
        raise ValueError(
            "Prediction labels do not match HC3 test rows."
        )

    # Avoid duplicate or partial model evaluations.
    for model in selected_models:
        subset = predictions[predictions["model"] == model]

        if (
            len(subset) != len(test)
            or subset["test_row"].nunique() != len(test)
        ):
            raise ValueError(
                f"Incomplete or duplicate predictions for {model}."
            )

    records = []

    for (model, bucket), group in predictions.groupby(
        ["model", "length_bin"],
        observed=True,
    ):
        human = group[group["target"] == 0]
        ai = group[group["target"] == 1]

        records.append({
            "model": model,
            "length_bin": str(bucket),
            "samples": len(group),
            "human_samples": len(human),
            "ai_samples": len(ai),
            "human_recall": (
                (human["prediction"] == 0).mean()
                if len(human) else np.nan
            ),
            "ai_recall": (
                (ai["prediction"] == 1).mean()
                if len(ai) else np.nan
            ),
            "macro_f1": f1_score(
                group["target"],
                group["prediction"],
                average="macro",
                labels=[0, 1],
                zero_division=0,
            ),
        })

    metrics = pd.DataFrame(records)

    metrics.to_csv(
        DATA / "hc3_length_bucket_metrics.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True,
    )

    plots = [
        ("human_recall", "Human Recall"),
        ("ai_recall", "AI Recall"),
    ]

    for ax, (metric, title) in zip(axes, plots):
        for model in selected_models:
            subset = (
                metrics[metrics["model"] == model]
                .set_index("length_bin")
                .reindex(labels)
            )

            color = (
                COLORS["roberta"]
                if model == "RoBERTa"
                else COLORS["ffnn"]
            )

            ax.plot(
                labels,
                subset[metric].to_numpy() * 100,
                marker="o",
                markersize=7,
                linewidth=2.5,
                color=color,
                label=model,
            )

        ax.set_title(title, weight="bold")
        ax.set_xlabel("Answer length (words)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("Recall (%)")
    axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle(
        "HC3 Test Performance by Answer Length",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "hc3_performance_by_answer_length.png",
    )

    print("\nAnswer-length evaluation:")
    print(metrics.to_string(index=False))


# ---------------------------------------------------------
# Run all exports
# ---------------------------------------------------------

def main():
    print("\nExporting Spot the Bot README figures...\n")

    plot_hc3_vs_modern()
    plot_distribution_shift_confusion_matrices()
    plot_modern_roc()
    plot_modern_score_distributions()
    plot_length_performance()

    print("\nFinished.")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()

