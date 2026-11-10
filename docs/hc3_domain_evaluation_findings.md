# HC3 Domain Evaluation

## Purpose and scope

This analysis compares **nine previously trained human-versus-AI text classifiers** across five source domains in the HC3 test set. The goal was to identify domain-specific strengths, weaknesses, and differences between feature representations, rather than to retrain models or optimize thresholds. This is an evaluation of performance **within HC3**, not evidence of reliability on newly generated AI text or in real-world deployment.

The target labels are **Human = 0** and **AI = 1**. The held-out test set contains **12,632 responses**. The domain breakdown is:

| Domain | Human | AI | Total |
|---|---:|---:|---:|
| Finance | 590 | 693 | 1,283 |
| Medicine | 187 | 201 | 388 |
| Open QA | 178 | 532 | 710 |
| Reddit ELI5 | 7,497 | 2,500 | 9,997 |
| Wiki CS/AI | 127 | 127 | 254 |
| **Total** | **8,579** | **4,053** | **12,632** |

Reddit ELI5 makes up about **79%** of the test set; Wiki CS/AI is the smallest domain at about **2%**. These differences matter when interpreting aggregate results and the stability of domain-specific error rates.

## Models compared

- **TF-IDF:** Feedforward neural network (FFNN), linear SVM, logistic regression, multinomial Naive Bayes, and K-nearest neighbors (KNN).
- **Sentence embeddings:** FFNN, linear SVM, and logistic regression using **all-MiniLM-L6-v2** embeddings (384 dimensions).
- **Transformer:** Fine-tuned **RoBERTa** sequence classifier, evaluated with a maximum input length of 384 tokens.

The KNN evaluation used its saved **0.475** classification threshold. RoBERTa predictions used the higher of the two output logits (equivalent to a 0.5 softmax threshold). The embedding FFNN used a **provisional 0.5 threshold**, because its original threshold was not confirmed in saved metadata; that detail should be verified against its training notebook before treating the comparison as final. All other models used their loaded prediction behavior. The embedding arrays were treated as aligned with the test rows; the available metadata did not independently verify row order.

## Metrics

- **Accuracy:** Fraction of all examples classified correctly.
- **Precision (for a class):** Among examples predicted as that class, the fraction actually belonging to it.
- **Recall (for a class):** Among examples actually belonging to that class, the fraction correctly identified.
- **F1 (for a class):** Harmonic mean of that class's precision and recall.
- **Macro F1:** Arithmetic average of **Human F1** and **AI F1**. It weights the two classes equally even when their sample counts differ. A value of 0.99 is a high combined F1 score, **not** literally 99% accuracy.
- **Human false-positive rate (FPR):** Among genuinely human responses, the proportion incorrectly classified as AI. With Human = 0 and AI = 1, this is `FP / (TN + FP)`, or `1 − human recall`.

Higher macro F1 is better; lower human FPR is better. The two metrics answer different questions: macro F1 summarizes balanced class performance, while human FPR isolates the risk of falsely flagging human writing.

## Macro F1 by domain

| Model | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CS/AI |
|---|---:|---:|---:|---:|---:|
| TF-IDF FFNN | **0.9953** | **1.0000** | 0.9132 | 0.9919 | **0.9290** |
| TF-IDF Linear SVM | 0.9906 | 0.9974 | 0.9274 | 0.9911 | 0.9172 |
| TF-IDF Logistic Regression | 0.9882 | 0.9974 | 0.9201 | 0.9893 | 0.9172 |
| TF-IDF Multinomial NB | 0.9629 | 0.9974 | 0.8324 | 0.9391 | 0.7758 |
| TF-IDF KNN | 0.7835 | 0.9767 | 0.8350 | 0.9440 | 0.7480 |
| Embedding FFNN | 0.9293 | 0.9044 | 0.7641 | 0.9190 | 0.7225 |
| Embedding Linear SVM | 0.8715 | 0.7237 | 0.7322 | 0.8489 | 0.6179 |
| Embedding Logistic Regression | 0.8601 | 0.7117 | 0.7056 | 0.8472 | 0.6349 |
| RoBERTa | 0.9890 | 0.9974 | **0.9944** | **0.9936** | 0.8966 |

**Highest macro F1 per domain:** TF-IDF FFNN in Finance, Medicine, and Wiki CS/AI; RoBERTa in Open QA and Reddit ELI5.

### Macro F1 findings

1. **TF-IDF FFNN led in three of five domains.** Its 1.0000 macro F1 in Medicine reflects perfect predictions on that particular 388-row test subset, not guaranteed perfect generalization.
2. **RoBERTa excelled in Open QA**, scoring **0.9944**, versus **0.9132** for TF-IDF FFNN and **0.9274** for TF-IDF Linear SVM. It also narrowly led Reddit ELI5.
3. **Linear SVM and logistic regression were competitive with the TF-IDF FFNN** in several domains, showing that relatively simple classifiers can work extremely well with high-dimensional sparse text features.
4. **All three MiniLM embedding-based models scored below the strongest TF-IDF and RoBERTa models** across the five domains. This does not establish that embeddings are generally inferior; the result depends on these models, their training procedures, their feature representations, and this dataset.
5. **Wiki CS/AI had the lowest macro F1 for every model except TF-IDF KNN**, for which Finance was slightly lower. This suggests a recurring domain-specific challenge rather than a problem unique to RoBERTa.

## Human false-positive rate by domain

Values below are percentages of **actual human** responses incorrectly classified as AI.

| Model | Finance | Medicine | Open QA | Reddit ELI5 | Wiki CS/AI |
|---|---:|---:|---:|---:|---:|
| TF-IDF FFNN | **0.68%** | **0.00%** | 7.87% | 0.45% | **11.02%** |
| TF-IDF Linear SVM | 1.19% | **0.00%** | 6.18% | **0.41%** | 12.60% |
| TF-IDF Logistic Regression | 1.69% | **0.00%** | 7.30% | 0.56% | 12.60% |
| TF-IDF Multinomial NB | 7.97% | 0.53% | 29.78% | 0.91% | 40.16% |
| TF-IDF KNN | 43.39% | 4.81% | 27.53% | 2.83% | 45.67% |
| Embedding FFNN | 8.64% | 11.23% | 23.03% | 5.47% | 41.73% |
| Embedding Linear SVM | 19.32% | 44.39% | 26.97% | 11.60% | 59.06% |
| Embedding Logistic Regression | 21.36% | 46.52% | 26.40% | 11.90% | 55.12% |
| RoBERTa | 2.37% | 0.53% | **1.12%** | 0.64% | 20.47% |

### False-positive findings

- **TF-IDF FFNN was particularly strong at avoiding false accusations of human writing**, with the lowest human FPR in Finance and Wiki CS/AI and a tie for lowest in Medicine. TF-IDF Linear SVM narrowly led Reddit ELI5, while RoBERTa led Open QA.
- **RoBERTa's Wiki CS/AI FPR was 20.47% (26 of 127 human responses)**, compared with **11.02% (14 of 127)** for TF-IDF FFNN. That is a difference of **9.45 percentage points**, or roughly **1.86 times** the FFNN's rate. The gap is notable despite RoBERTa's strong performance in other domains.
- **The embedding-based models also struggled with Wiki CS/AI**, with human FPRs ranging from **41.73% to 59.06%**. Their error rates were substantially higher than RoBERTa's, so the *direction* of the Wiki difficulty was shared but the *magnitude* was not the same.
- **Reddit ELI5 had low human FPR for the strongest TF-IDF and RoBERTa models**. It also had far more examples than any other domain, which might contribute to performance, but the domain's training counts and content differences would need further analysis to establish why.
- **Naive Bayes and KNN were considerably less reliable in some domains**, especially Finance, Open QA, and Wiki CS/AI. This demonstrates that the choice of classifier matters even when using the same TF-IDF representation.

## Interpretation and limitations

**Why might Wiki CS/AI be difficult?** One possible hypothesis is that formal, technical, encyclopedia-style human writing shares vocabulary or structural patterns with AI-generated explanations. Another is that the small domain-specific training sample limits the models' ability to distinguish the two classes. Neither explanation has been demonstrated by this analysis. Topic, source formatting, answer length, class balance, and annotation artifacts could also matter.

**Why might TF-IDF perform so well?** TF-IDF represents words and word combinations, which can capture lexical, topical, and stylistic signals. The results are consistent with those signals being useful on HC3. However, the metrics do not prove that the TF-IDF models learned *style* rather than topic-specific or dataset-specific shortcuts. MiniLM sentence embeddings encode different information, and RoBERTa can use contextual token relationships, but representation alone does not establish the cause of performance differences.

**Small-sample caution:** Wiki CS/AI has only **127 human examples**. Each additional false positive changes its measured human FPR by about **0.79 percentage points**. Medicine also has a relatively small test set. The domain differences are informative, but uncertainty should be considered before generalizing them.

**Dataset scope:** These are **HC3 test-set results**. The project's separate modern-data evaluation showed that strong HC3 scores do not necessarily transfer to newer AI-generated text. These domain results should not be presented as proof of robust general-purpose AI detection.

## Potential future exploration (outside current project scope)

The most interesting follow-up would be **error analysis of the Wiki CS/AI human responses**:

1. Identify the **26 Wiki human responses RoBERTa misclassified** and compare them with the **14 misclassified by TF-IDF FFNN**.
2. Separate **shared errors** from responses only RoBERTa gets wrong. Examine answer length, formatting, technical vocabulary, and phrasing without assuming any one factor caused the errors.
3. Compare these patterns with a better-performing domain, such as Reddit ELI5 or Finance, to distinguish possible source-specific effects from general model behavior.
4. If pursued as a separate research extension, inspect **per-domain training counts** and consider uncertainty intervals or controlled experiments to test whether data quantity, domain style, or representation explains the gap.

These are **ideas for future investigation**, not requirements or planned additions to the current Chewy deliverable.

## Overall takeaway

Across nine classifiers and five HC3 domains, **TF-IDF FFNN and RoBERTa were the strongest overall contenders**, but their strengths differed by source. TF-IDF FFNN led macro F1 in three domains and generally minimized false positives against human writing. RoBERTa delivered exceptional Open QA performance but had a conspicuous Wiki CS/AI weakness. TF-IDF Linear SVM remained a close competitor to the FFNN, making it a valuable, comparatively simple baseline. The clearest lesson is that **high overall classification performance can conceal meaningful domain-specific failure modes**, especially when a dataset is dominated by one source.
