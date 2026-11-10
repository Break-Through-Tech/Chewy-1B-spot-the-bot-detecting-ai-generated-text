# Modern Dataset Drift Analysis — Findings

**Models evaluated:** 7 classifiers (TF-IDF, embeddings, RoBERTa)

## 1. Initial Evaluation: Modern Human vs. All AI Generators

We evaluated seven HC3-trained classifiers against a modern external dataset containing 600 responses:

- 150 human
- 150 ChatGPT
- 150 Claude
- 150 Gemini

All models were evaluated using their original prediction rules, without retraining or threshold adjustments.

### 1.1 Overall Classification Performance

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| TF-IDF FFNN | 0.268 | 0.227 | **0.857** |
| TF-IDF Logistic Regression | 0.277 | 0.241 | 0.737 |
| TF-IDF Linear SVM | 0.275 | 0.238 | 0.735 |
| RoBERTa | 0.320 | 0.299 | 0.709 |
| Embeddings FFNN | 0.305 | 0.289 | 0.549 |
| Embeddings Logistic Regression | **0.383** | **0.383** | 0.544 |
| Embeddings Linear SVM | 0.375 | 0.374 | 0.529 |

**Key observations:**

- Embeddings Logistic Regression achieved the highest accuracy (38.3%) and macro-F1 (0.383), despite having relatively weak ROC-AUC (0.544).
- TF-IDF FFNN achieved the highest ROC-AUC (0.857), substantially outperforming RoBERTa (0.709).
- The TF-IDF models generally demonstrated stronger ranking performance than the embedding-based models.
- All models performed poorly at their original classification thresholds.

The modern dataset contains 75% AI-generated responses, making accuracy particularly sensitive to models incorrectly predicting AI responses as human.

### 1.2 Generator-Specific AI Recall

The following results show the proportion of responses from each AI generator correctly identified as AI.

| Model | ChatGPT | Claude | Gemini | Human FPR |
|---|---:|---:|---:|---:|
| Embeddings FFNN | 11.33% | 15.33% | 4.00% | 8.67% |
| Embeddings Linear SVM | 26.00% | 32.00% | 10.67% | 18.67% |
| Embeddings Logistic Regression | **29.33%** | **32.00%** | 9.33% | 17.33% |
| TF-IDF FFNN | 4.00% | 3.33% | 0.00% | **0.00%** |
| TF-IDF Linear SVM | 1.33% | 9.33% | 0.00% | 0.67% |
| TF-IDF Logistic Regression | 2.67% | 9.33% | 0.00% | 1.33% |
| RoBERTa | 9.33% | 18.67% | 1.33% | 1.33% |

**Key observations:**

- Gemini was consistently the most difficult generator to detect at the original decision thresholds.
- The three TF-IDF models detected 0% of Gemini responses.
- Embedding-based classifiers generally detected more ChatGPT and Claude responses, but also produced more human false positives.
- TF-IDF FFNN achieved 0% human false positives while detecting only 4% of ChatGPT responses.
- RoBERTa detected Claude more successfully than ChatGPT or Gemini, suggesting generator-dependent behavior.

### 1.3 Initial Interpretation

The results revealed a discrepancy between ranking performance and classification performance.

Although TF-IDF FFNN had the lowest overall accuracy (26.8%), it achieved the highest ROC-AUC (0.857). This indicates that its underlying scores retained substantial ability to rank AI-generated responses above human responses, even though its original threshold produced very few positive AI classifications.

This motivated further investigation into the models' score distributions and decision thresholds.

However, the evaluation alone cannot isolate the effects of model-generation changes over time from domain differences, prompt differences, or other dataset shifts.

## 2. Controlled Comparison: Modern ChatGPT vs. Human

Because the original HC3 training data contains ChatGPT-generated responses, we restricted the modern evaluation to ChatGPT and human responses to create a more directly comparable test.

**Subset:** 300 responses (150 ChatGPT, 150 human).

| Model | Accuracy | Macro-F1 | ROC-AUC | AI Recall |
|---|---:|---:|---:|---:|
| TF-IDF FFNN | 52.0% | 0.3763 | **0.8981** | 4.0% |
| TF-IDF Linear SVM | 50.3% | 0.3464 | 0.7531 | 1.3% |
| TF-IDF Logistic Regression | 50.7% | 0.3590 | 0.7621 | 2.7% |
| Embeddings FFNN | 51.3% | 0.4206 | 0.6075 | 11.3% |
| Embeddings Linear SVM | 53.7% | 0.4983 | 0.5670 | 26.0% |
| Embeddings Logistic Regression | **56.0%** | **0.5263** | 0.5969 | **29.3%** |
| RoBERTa | 54.0% | 0.4254 | 0.8073 | 9.3% |

**Key discovery:** TF-IDF FFNN achieved the highest ROC-AUC (0.8981), outperforming RoBERTa (0.8073), despite detecting only 4% of modern ChatGPT responses at its original threshold.

Restricting the evaluation increased RoBERTa's ROC-AUC from 0.7088 to 0.8073, while TF-IDF FFNN's ROC-AUC increased from approximately 0.857 to 0.8981.

This suggests ranking performance varies across AI generators.

The increase in accuracy partly reflects the change from an AI-heavy dataset to a balanced dataset, rather than necessarily improved model behavior.

## 3. Score Distribution Analysis

We visualized TF-IDF FFNN and RoBERTa prediction scores using standard histograms and logarithmic-scale histograms.

Both models assigned very low AI scores to most modern ChatGPT responses, often far below the default 0.5 threshold.

The logarithmic scale revealed differences hidden in the original graphs:

- **TF-IDF FFNN:** Human responses generally received lower scores than ChatGPT responses, with visible separation and some overlap.
- **RoBERTa:** Human responses were tightly clustered near zero, while ChatGPT responses were more widely distributed, with substantial overlap.

**Interpretation:** The models retained some ability to rank modern ChatGPT responses above human responses, even though their original classification thresholds produced poor recall.

The logarithmic transformation changed only the visualization. It did not modify model scores, predictions, or weights.

## 4. ROC Curves and Experimental Threshold Analysis

We compared ROC curves to examine AI recall at different human false-positive rates.

TF-IDF FFNN generally outperformed RoBERTa across much of the ROC curve, particularly around the 10% false-positive range.

We then calculated thresholds corresponding to maximum human false-positive rates of 5% and 10%.

| Model | Maximum Human FPR | Threshold | AI Recall | Actual Human FPR |
|---|---:|---:|---:|---:|
| TF-IDF FFNN | 5% | 0.02242509 | 26.0% | 4.7% |
| TF-IDF FFNN | 10% | 0.00309986 | **64.7%** | 10.0% |
| RoBERTa | 5% | 0.00182704 | 23.3% | 4.7% |
| RoBERTa | 10% | 0.00033405 | 50.0% | 10.0% |

At the 10% false-positive limit, TF-IDF FFNN detected approximately 97 of 150 modern ChatGPT responses, compared with 75 of 150 for RoBERTa.

These experimental thresholds increased AI recall without modifying the trained models.

However, lowering the thresholds also increased human false positives.

**Important:** These thresholds were selected using the same modern evaluation examples. Results are exploratory and are not independent estimates of performance on unseen data.

## 5. Conclusions

1. All seven HC3-trained classifiers demonstrated substantial degradation when evaluated against modern AI-generated responses using their original decision rules.
2. Generator-specific results revealed substantial differences in detection performance, with Gemini particularly difficult to detect.
3. TF-IDF FFNN achieved the strongest ROC-AUC on both the full modern dataset (0.857) and the ChatGPT-only subset (0.8981).
4. Restricting the evaluation to modern ChatGPT versus human responses revealed stronger ranking performance for several models, particularly TF-IDF FFNN and RoBERTa.
5. Poor AI recall at the default threshold does not necessarily mean the model has lost all ability to distinguish AI from human text.
6. Score distributions revealed that many modern AI responses received very low AI scores, suggesting decision-threshold mismatch contributes to classification failures.
7. Experimental threshold adjustments substantially increased modern ChatGPT recall, but also increased human false positives.
8. The simpler TF-IDF FFNN outperformed RoBERTa in ROC-AUC and in recall at the evaluated false-positive limits on this particular subset.

## 6. Limitations and Next Steps

**Limitations:**
- Only 150 modern ChatGPT and 150 human responses were used in the controlled comparison.
- The modern dataset was used for exploratory threshold selection, so the adjusted results are not unbiased held-out performance estimates.
- Differences between datasets may reflect generator changes, domain differences, text characteristics, or other distribution shifts.
- We have not established whether the experimental modern thresholds would improve or damage HC3 performance.
- Strong ROC-AUC does not guarantee reliable classifications at a practically acceptable false-positive rate.

**Next steps:**
- Preserve original HC3-trained models and reported evaluation results.
- Examine HC3 test-set score distributions for TF-IDF FFNN and RoBERTa.
- Compare HC3 distributions with modern ChatGPT distributions.
- Optionally evaluate threshold sensitivity on HC3 without changing the original models.
- Keep modern drift analysis as a supplementary robustness experiment rather than retraining the models on modern data.

**Overall takeaway:** The modern drift experiment revealed that poor classification performance can coexist with meaningful ranking ability. The findings highlight the importance of evaluating model generalization, score distributions, ROC-AUC, threshold sensitivity, and false-positive trade-offs rather than relying on accuracy alone.