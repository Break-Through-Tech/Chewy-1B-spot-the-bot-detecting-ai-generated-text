# Per-model error patterns

Rates use the fixed held-out test set. For each model, a false positive is a human response predicted AI; a false negative is an AI response predicted human. Length and style rates are observational and can reflect topic/source mix. Vocabulary terms are enriched among a model's errors versus correctly classified responses of the same true class; minimum four error examples, ranked by document-frequency lift.

## FFNN

Errors: 133/12,632 (1.1%); false positives: 66/8,579 humans (0.8%); false negatives: 67/4,053 AI (1.7%).

### Length

| Length | n | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Very short (<20) | 888 | 11 (1.2%) | 6 | 5 |
| Short (20–99) | 4,824 | 54 (1.1%) | 27 | 27 |
| Medium (100–299) | 6,077 | 56 (0.9%) | 23 | 33 |
| Very long (300+) | 843 | 12 (1.4%) | 10 | 2 |

### Writing style and formatting

| Pattern present | n | Error rate | FP | FN |
|---|---:|---:|---:|---:|
| List marker | 86 | 1.2% | 1 | 0 |
| Markdown | 65 | 0.0% | 0 | 0 |
| URL | 72 | 1.4% | 1 | 0 |
| Code/formula | 306 | 2.0% | 2 | 4 |
| Repeated punctuation | 828 | 0.4% | 1 | 2 |
| Parentheses | 4,522 | 0.9% | 30 | 12 |
| First person | 4,749 | 0.6% | 19 | 11 |
| Second person | 6,613 | 0.6% | 15 | 22 |
| Contractions | 4,916 | 0.4% | 6 | 12 |
| Hedging | 3,294 | 0.9% | 14 | 14 |
| Conversational marker | 1,905 | 0.8% | 11 | 4 |
| Question mark | 1,334 | 0.4% | 2 | 4 |

### Vocabulary associated with errors

- False positives: awarded (4, 105.5×), greatest (4, 33.2×), varied (4, 28.3×), fiction (4, 27.0×), models (4, 20.4×), located (5, 18.4×), approach (4, 14.7×), largest (5, 14.0×)
- False negatives: diego (4, 107.1×), san (4, 31.5×), died (4, 9.1×), english (4, 6.8×), century (4, 6.8×), rule (4, 6.2×), television (5, 5.3×), his (11, 5.0×)

## KNN

Errors: 832/12,632 (6.6%); false positives: 584/8,579 humans (6.8%); false negatives: 248/4,053 AI (6.1%).

### Length

| Length | n | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Very short (<20) | 888 | 22 (2.5%) | 18 | 4 |
| Short (20–99) | 4,824 | 238 (4.9%) | 214 | 24 |
| Medium (100–299) | 6,077 | 468 (7.7%) | 257 | 211 |
| Very long (300+) | 843 | 104 (12.3%) | 95 | 9 |

### Writing style and formatting

| Pattern present | n | Error rate | FP | FN |
|---|---:|---:|---:|---:|
| List marker | 86 | 2.3% | 2 | 0 |
| Markdown | 65 | 4.6% | 3 | 0 |
| URL | 72 | 23.6% | 17 | 0 |
| Code/formula | 306 | 19.0% | 54 | 4 |
| Repeated punctuation | 828 | 7.0% | 55 | 3 |
| Parentheses | 4,522 | 8.9% | 345 | 56 |
| First person | 4,749 | 6.7% | 254 | 65 |
| Second person | 6,613 | 7.2% | 333 | 141 |
| Contractions | 4,916 | 6.4% | 247 | 68 |
| Hedging | 3,294 | 6.9% | 155 | 72 |
| Conversational marker | 1,905 | 6.4% | 107 | 15 |
| Question mark | 1,334 | 8.2% | 71 | 38 |

### Vocabulary associated with errors

- False positives: undergraduate (4, 123.2×), subsidiary (4, 123.2×), itemize (4, 123.2×), faculty (4, 123.2×), awards (4, 123.2×), appreciation (4, 123.2×), issuer (10, 95.8×), award (7, 68.5×)
- False negatives: theorem (4, 46.0×), paradox (6, 39.9×), separated (4, 27.6×), cube (4, 27.6×), mathematicians (4, 19.7×), truly (6, 18.1×), sequence (6, 18.1×), bounce (6, 18.1×)

## Linear SVM

Errors: 141/12,632 (1.1%); false positives: 65/8,579 humans (0.8%); false negatives: 76/4,053 AI (1.9%).

### Length

| Length | n | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Very short (<20) | 888 | 7 (0.8%) | 3 | 4 |
| Short (20–99) | 4,824 | 49 (1.0%) | 21 | 28 |
| Medium (100–299) | 6,077 | 66 (1.1%) | 24 | 42 |
| Very long (300+) | 843 | 19 (2.3%) | 17 | 2 |

### Writing style and formatting

| Pattern present | n | Error rate | FP | FN |
|---|---:|---:|---:|---:|
| List marker | 86 | 0.0% | 0 | 0 |
| Markdown | 65 | 0.0% | 0 | 0 |
| URL | 72 | 2.8% | 2 | 0 |
| Code/formula | 306 | 2.0% | 2 | 4 |
| Repeated punctuation | 828 | 0.5% | 2 | 2 |
| Parentheses | 4,522 | 1.0% | 31 | 15 |
| First person | 4,749 | 0.7% | 22 | 11 |
| Second person | 6,613 | 0.7% | 20 | 24 |
| Contractions | 4,916 | 0.5% | 11 | 14 |
| Hedging | 3,294 | 0.9% | 15 | 16 |
| Conversational marker | 1,905 | 0.9% | 11 | 7 |
| Question mark | 1,334 | 0.7% | 4 | 5 |

### Vocabulary associated with errors

- False positives: greatest (4, 33.7×), varied (4, 28.8×), fiction (4, 27.4×), underlying (5, 22.2×), located (5, 18.7×), wide (7, 15.2×), century (7, 15.0×), approach (4, 14.9×)
- False negatives: russian (4, 24.8×), served (4, 11.0×), rule (5, 6.8×), points (4, 6.5×), century (4, 6.0×), his (14, 5.7×), learning (5, 4.3×), top (5, 3.9×)

## Logistic Regression

Errors: 161/12,632 (1.3%); false positives: 81/8,579 humans (0.9%); false negatives: 80/4,053 AI (2.0%).

### Length

| Length | n | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Very short (<20) | 888 | 8 (0.9%) | 4 | 4 |
| Short (20–99) | 4,824 | 56 (1.2%) | 25 | 31 |
| Medium (100–299) | 6,077 | 76 (1.3%) | 33 | 43 |
| Very long (300+) | 843 | 21 (2.5%) | 19 | 2 |

### Writing style and formatting

| Pattern present | n | Error rate | FP | FN |
|---|---:|---:|---:|---:|
| List marker | 86 | 1.2% | 1 | 0 |
| Markdown | 65 | 0.0% | 0 | 0 |
| URL | 72 | 2.8% | 2 | 0 |
| Code/formula | 306 | 2.0% | 2 | 4 |
| Repeated punctuation | 828 | 0.7% | 4 | 2 |
| Parentheses | 4,522 | 1.3% | 43 | 14 |
| First person | 4,749 | 0.9% | 28 | 14 |
| Second person | 6,613 | 0.9% | 29 | 28 |
| Contractions | 4,916 | 0.7% | 21 | 14 |
| Hedging | 3,294 | 1.2% | 21 | 17 |
| Conversational marker | 1,905 | 0.9% | 12 | 6 |
| Question mark | 1,334 | 0.7% | 4 | 5 |

### Vocabulary associated with errors

- False positives: awarded (4, 85.8×), december (5, 31.2×), greatest (4, 27.0×), senate (4, 25.5×), varied (4, 23.0×), located (5, 15.0×), tasks (4, 14.5×), underlying (4, 14.1×)
- False negatives: truth (4, 34.4×), russian (4, 23.5×), saying (4, 11.5×), served (4, 10.4×), thousands (4, 9.5×), began (4, 8.8×), rule (5, 6.4×), his (15, 5.8×)

## MultinomialNB

Errors: 616/12,632 (4.9%); false positives: 202/8,579 humans (2.4%); false negatives: 414/4,053 AI (10.2%).

### Length

| Length | n | Errors | FP | FN |
|---|---:|---:|---:|---:|
| Very short (<20) | 888 | 38 (4.3%) | 34 | 4 |
| Short (20–99) | 4,824 | 157 (3.3%) | 112 | 45 |
| Medium (100–299) | 6,077 | 388 (6.4%) | 38 | 350 |
| Very long (300+) | 843 | 33 (3.9%) | 18 | 15 |

### Writing style and formatting

| Pattern present | n | Error rate | FP | FN |
|---|---:|---:|---:|---:|
| List marker | 86 | 1.2% | 1 | 0 |
| Markdown | 65 | 0.0% | 0 | 0 |
| URL | 72 | 2.8% | 2 | 0 |
| Code/formula | 306 | 4.6% | 5 | 9 |
| Repeated punctuation | 828 | 2.3% | 12 | 7 |
| Parentheses | 4,522 | 3.9% | 95 | 81 |
| First person | 4,749 | 3.7% | 25 | 151 |
| Second person | 6,613 | 4.2% | 42 | 239 |
| Contractions | 4,916 | 3.2% | 16 | 139 |
| Hedging | 3,294 | 4.7% | 18 | 137 |
| Conversational marker | 1,905 | 2.3% | 16 | 27 |
| Question mark | 1,334 | 4.9% | 5 | 61 |

### Vocabulary associated with errors

- False positives: awards (4, 373.2×), surveillance (4, 74.6×), inc (4, 28.7×), october (4, 24.9×), april (4, 19.6×), founded (7, 18.9×), approaches (4, 17.8×), contributions (5, 16.9×)
- False negatives: leap (4, 79.1×), infinite (10, 61.5×), paradox (7, 43.9×), paradoxes (4, 26.4×), forever (13, 26.4×), contradictory (4, 26.4×), gravitational (9, 23.9×), experiment (6, 22.9×)

## Reading this report

A high error rate in a category flags a review priority; it does not show the feature caused the error. Style and format indicators can overlap, and small groups are unstable. Use the response-level CSV to inspect examples and source/domain before drawing conclusions.
