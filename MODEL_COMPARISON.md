# Classical Model Comparison

This is the shared repository record for classifiers trained on the HC3 answer-level dataset. The saved project train/validation/test splits are question-grouped and use `target=0` for human answers and `target=1` for ChatGPT answers. Classical models should use the same text representation and split for a fair comparison.

## Logistic Regression baseline

**Data:** 67,232 training answers, 8,417 validation answers, and 8,417 test answers. The test set contains 5,719 human answers and 2,698 ChatGPT answers. The fixed split has no question IDs shared across train, validation, and test.

**Features:** Word-level TF-IDF, unigram and bigram range, `min_df=2`, `max_features=100000`, sublinear term frequency, Unicode accent stripping, `float32`. The vectorizer is fit on training data for validation tuning, then refit on train plus validation for final evaluation.

**Classifier:** `LogisticRegression(C=50, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)`. C was selected by validation macro-F1 from `[0.1, 1, 4, 10, 20, 50, 100]`.

### Held-out test metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.9899 |
| Macro precision | 0.9887 |
| Macro recall | 0.9881 |
| Macro F1 | 0.9884 |
| Majority-class accuracy | 0.6795 |

Confusion matrix (rows = true class, columns = predicted class; order = human, ChatGPT):

|  | Predicted human | Predicted ChatGPT |
|---|---:|---:|
| True human | 5,680 | 39 |
| True ChatGPT | 46 | 2,652 |

### Per-domain test results

| Domain | Answers | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Finance | 853 | 0.9883 | 0.9882 |
| Medicine | 262 | 1.0000 | 1.0000 |
| Open Q&A | 470 | 0.9319 | 0.9124 |
| Reddit ELI5 | 6,664 | 0.9947 | 0.9930 |
| Wikipedia CS | 168 | 0.9524 | 0.9523 |

### Findings and limitations

The baseline substantially exceeds the test majority-class accuracy. It makes 85 errors: 39 human answers flagged as ChatGPT and 46 ChatGPT answers classified as human. Performance is lower on Open Q&A than on the larger Reddit ELI5 domain; the Wikipedia CS test group is also small, so those domain estimates are less stable.

The strongest learned terms include “including,” “important,” and “such as” for ChatGPT and `URL_0`, “basically,” and “etc.” for human answers. These are corpus-level associations, not reliable authorship evidence. High HC3 performance should not be read as evidence that the detector generalizes to modern models, new topics, or real-world writing; the source text and benchmark are limited. Review false positives and false negatives and evaluate robustness before using the model outside this benchmark.

## K-Nearest Neighbors (KNN)

**Data:** Used the same train, validation, and test splits as Logistic Regression, with 67,232 training answers, 8,417 validation answers, and 8,417 test answers.

**Features:** Used the same word-level TF-IDF settings as Logistic Regression, including unigrams and bigrams, `min_df=2`, `max_features=100000`, sublinear TF, Unicode accent stripping, and `float32`. The vectorizer was fit on training data during tuning and then refit on training plus validation for final testing.

**Classifier:** `KNeighborsClassifier(n_neighbors=400, weights="distance", metric="cosine", algorithm="brute", n_jobs=-1)`. I tested different K values and voting weights using validation macro-F1 to choose the best configuration. Larger K values generally performed better, with K=400 giving the best results. I also tested different classification thresholds and selected `0.475`.

### Held-out test metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.9353 |
| Macro precision | 0.9208 |
| Macro recall | 0.9339 |
| Macro F1 | 0.9269 |
| Majority-class accuracy | 0.6795 |

Confusion matrix (rows = actual class, columns = predicted class; order = human, ChatGPT):

| | Predicted human | Predicted ChatGPT |
|---|---:|---:|
| True human | 5,362 | 357 |
| True ChatGPT | 188 | 2,510 |

### Findings and limitations

KNN reached 93.53% accuracy and 92.69% macro-F1 on the test set. It performed better than the majority-class baseline, but Logistic Regression still performed better overall, with 98.99% accuracy and 98.84% macro-F1.

Increasing K made a pretty big difference in performance. The smaller K values performed noticeably worse, while larger values improved the results until they started leveling off. Using distance-weighted voting and adjusting the classification threshold also helped.

KNN made 545 incorrect predictions, including 357 human answers classified as ChatGPT and 188 ChatGPT answers classified as human. It also takes more computation to make predictions since it needs to compare answers against training examples. These results are specific to the HC3 dataset, so they don't necessarily mean the model would perform as well on other datasets or newer AI-generated text. Earlier KNN configurations were also evaluated on the test set during development, so the final test results weren't from a completely untouched test set.

## Other classical models

Add Linear SVM and Multinomial Naive Bayes results here using the same split, target labels, TF-IDF settings, metric definitions, and confusion-matrix class order.