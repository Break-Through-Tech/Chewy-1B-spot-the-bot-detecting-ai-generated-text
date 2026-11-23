"""Export README-ready evaluation figures. Run from any working directory.

Usage:
    python scripts/export_readme_figures.py

For the answer-length plot, first run the HC3 model-comparison notebook and
export its in-memory `prediction_rows` list into data/evaluation/hc3_test_predictions.csv:

    pd.concat(prediction_rows, ignore_index=True).to_csv(
        RESULTS_DIR / 'hc3_test_predictions.csv', index=False
    )

Do not upload the per-response prediction CSV if privacy or repository policy
requires keeping it local. Only PNGs are required for the README.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'evaluation'
TEST = ROOT / 'data' / 'processed' / 'test.csv'
OUTPUT = ROOT / 'docs' / 'images'
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'savefig.dpi': 200})


def save(fig, filename):
    target = OUTPUT / filename
    fig.savefig(target, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', target.relative_to(ROOT))


def plot_hc3_summary():
    path = DATA / 'hc3_model_metrics.csv'
    if not path.exists():
        print('SKIP HC3 summary: missing', path)
        return
    df = pd.read_csv(path).sort_values('macro_f1')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df['model'], df['macro_f1'])
    ax.set(xlim=(0, 1.02), xlabel='Macro F1', title='HC3 held-out test — model comparison')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    save(fig, 'hc3_model_macro_f1.png')


def plot_modern_summary():
    path = DATA / 'modern_seven_model_comparison.csv'
    if not path.exists():
        print('SKIP modern summary: missing', path)
        return
    df = pd.read_csv(path).sort_values('macro_f1')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df['model'], df['macro_f1'])
    ax.set(xlim=(0, 1), xlabel='Macro F1', title='Modern external dataset — model comparison')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    save(fig, 'modern_model_macro_f1.png')


def plot_length_performance():
    path = DATA / 'hc3_test_predictions.csv'
    if not path.exists():
        print('SKIP answer-length plot: missing', path)
        print('Run the HC3 comparison notebook and export prediction_rows (instructions above).')
        return
    if not TEST.exists():
        print('SKIP answer-length plot: missing', TEST)
        return
    test = pd.read_csv(TEST)
    predictions = pd.read_csv(path)
    required = {'test_row', 'model', 'target', 'prediction'}
    if not required.issubset(predictions.columns):
        raise ValueError(f'Missing prediction columns: {required - set(predictions.columns)}')
    selected = ['RoBERTa', 'TF-IDF FFNN']
    predictions = predictions[predictions['model'].isin(selected)].copy()
    if predictions.empty:
        raise ValueError('No RoBERTa / TF-IDF FFNN predictions found')
    if not test['text'].notna().all():
        raise ValueError('Test data has missing text')
    test['word_count'] = test['text'].astype(str).str.findall(r"\b[\w']+\b").str.len()
    bins = [0, 50, 100, 200, 400, np.inf]
    labels = ['<50', '50–99', '100–199', '200–399', '400+']
    test['length_bin'] = pd.cut(test['word_count'], bins=bins, labels=labels, right=False)
    # Preserve exact held-out test row order and verify labels align.
    predictions = predictions.merge(
        test[['target', 'length_bin']].rename(columns={'target':'expected_target'}),
        left_on='test_row', right_index=True, how='left', validate='many_to_one'
    )
    if predictions['expected_target'].isna().any() or not (predictions['target'] == predictions['expected_target']).all():
        raise ValueError('Predictions and held-out test file are not row-aligned. Do not plot.')
    rows = []
    for (model, length_bin), grp in predictions.groupby(['model', 'length_bin'], observed=True):
        human = grp[grp['target'] == 0]
        ai = grp[grp['target'] == 1]
        rows.append({
            'model': model, 'length_bin': str(length_bin), 'n': len(grp),
            'human_n': len(human), 'ai_n': len(ai),
            'human_recall': (human['prediction'] == 0).mean() if len(human) else np.nan,
            'ai_recall': (ai['prediction'] == 1).mean() if len(ai) else np.nan,
            'macro_f1': f1_score(grp['target'], grp['prediction'], average='macro', labels=[0,1], zero_division=0),
        })
    result = pd.DataFrame(rows)
    result.to_csv(DATA / 'hc3_length_bucket_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ['human_recall', 'ai_recall'], ['Human recall', 'AI recall']):
        for model in selected:
            s = result[result['model'] == model].set_index('length_bin').reindex(labels)
            ax.plot(labels, s[metric].to_numpy() * 100, marker='o', linewidth=2, label=model)
        ax.set(xlabel='Response length (words)', title=title, ylim=(0, 101))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Recall (%)')
    axes[1].legend(loc='lower right')
    fig.suptitle('HC3 held-out test — model reliability by answer length')
    fig.tight_layout()
    save(fig, 'hc3_performance_by_answer_length.png')
    print('Saved', (DATA/'hc3_length_bucket_metrics.csv').relative_to(ROOT))
    print(result.to_string(index=False))


if __name__ == '__main__':
    plot_hc3_summary()
    plot_modern_summary()
    plot_length_performance()
