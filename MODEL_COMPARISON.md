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

## Other classical models

Add Linear SVM and Multinomial Naive Bayes results here using the same split, target labels, TF-IDF settings, metric definitions, and confusion-matrix class order.
