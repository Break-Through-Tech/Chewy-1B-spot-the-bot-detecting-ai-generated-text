# TF-IDF shared-error and robustness review

## Scope and method

Analysis uses the fixed held-out test split (12,632 answers; 8,579 human and 4,053 AI). TF-IDF uses the checked-in vectorizer. FFNN uses the checked-in Keras model; the other four classifiers are fit on train + validation. Settings follow executed notebook configurations where available: KNN k=275 and threshold 0.475, Linear SVM C=1, Logistic Regression C=50, and MultinomialNB alpha=0.01. The KNN value differs from the prose in `MODEL_COMPARISON.md`, which says k=400. These are benchmark diagnostics, not evidence of real-world authorship detection.

## Held-out results

| Model | Accuracy | Macro F1 | False positives | False negatives |
|---|---:|---:|---:|---:|
| FFNN | 0.989 | 0.988 | 66 | 67 |
| KNN | 0.934 | 0.926 | 584 | 248 |
| Linear SVM | 0.989 | 0.987 | 65 | 76 |
| Logistic Regression | 0.987 | 0.985 | 81 | 80 |
| MultinomialNB | 0.951 | 0.943 | 202 | 414 |

## Shared mistakes

46 examples are misclassified by all five models. The complete set is in `tfidf_error_predictions.csv`; a compact sample of up to 50 examples is in `tfidf_shared_mistakes.csv`. Error count means number of models wrong (0–5).

- Wrong for 5/5 models: 46
- Wrong for 4/5 models: 39
- Wrong for 3/5 models: 60
- Wrong for 2/5 models: 254
- Wrong for 1/5 models: 809

## Length and formatting patterns

Rates below are the share of examples in each bucket misclassified by at least one model, compared with the overall any-model error rate. Small buckets can be noisy.

Overall any-model error rate: 9.6%; all-five error rate: 0.4%.

| Pattern | n | Any-model error rate | All-five error rate |
|---|---:|---:|---:|
| length band = Medium (100–299 words) | 6,077 | 11.7% | 0.3% |
| length band = Short (20–99 words) | 4,824 | 6.9% | 0.3% |
| length band = Very long (300+ words) | 843 | 14.1% | 0.8% |
| length band = Very short (<20 words) | 888 | 5.2% | 0.5% |
| has newline = 0 | 12,632 | 9.6% | 0.4% |
| has list marker = 0 | 12,546 | 9.6% | 0.4% |
| has list marker = 1 | 86 | 3.5% | 0.0% |
| has markdown = 0 | 12,567 | 9.6% | 0.4% |
| has markdown = 1 | 65 | 4.6% | 0.0% |
| has url = 0 | 12,560 | 9.5% | 0.4% |
| has url = 1 | 72 | 23.6% | 0.0% |
| has code or formula = 0 | 12,326 | 9.3% | 0.3% |
| has code or formula = 1 | 306 | 21.6% | 1.3% |
| has repeated punctuation = 0 | 11,804 | 9.7% | 0.4% |
| has repeated punctuation = 1 | 828 | 8.2% | 0.1% |
| has parentheses = 0 | 8,110 | 8.7% | 0.4% |
| has parentheses = 1 | 4,522 | 11.1% | 0.3% |
| first person = 0 | 7,883 | 9.7% | 0.5% |
| first person = 1 | 4,749 | 9.3% | 0.2% |
| second person = 0 | 6,019 | 9.3% | 0.5% |
| second person = 1 | 6,613 | 9.8% | 0.2% |
| contractions = 0 | 7,716 | 10.3% | 0.5% |
| contractions = 1 | 4,916 | 8.5% | 0.2% |
| hedging language = 0 | 9,338 | 9.4% | 0.3% |
| hedging language = 1 | 3,294 | 10.0% | 0.4% |
| conversational marker = 0 | 10,727 | 9.9% | 0.4% |
| conversational marker = 1 | 1,905 | 7.9% | 0.3% |
| question mark = 0 | 11,298 | 9.4% | 0.4% |
| question mark = 1 | 1,334 | 10.8% | 0.3% |

## Error direction and vocabulary clues

False positives are human answers labeled AI; false negatives are AI answers labeled human. These lexical clues are descriptive associations among errors with minimum frequency three, not causal explanations or stable authorship signals.

### False positives (human → AI)
- Examples: 691; median word count: 115.
- Frequent content words in this error group: `invests` (5), `subsidiary` (4), `awards` (4), `faculty` (4), `undergraduate` (4), `robotics` (4), `itemize` (4), `appreciation` (4), `issuer` (10), `cyclical` (3), `toothpastes` (3), `secretaries` (3)

### False negatives (AI → human)
- Examples: 517; median word count: 162.
- Frequent content words in this error group: `leap` (4), `liturgical` (4), `infinite` (10), `stuff` (3), `climb` (3), `marble` (3), `marbles` (3), `maybe` (3), `elliptic` (3), `slotted` (3), `experimenting` (3), `asteroid` (3)

## Sample unanimous errors

Examples below were misclassified by all five models. Inspect the CSV for full responses.

- Test id `481`; true label `human`; 89 words; source `reddit_eli5`: “A zit is an infection of the pore. It typically forms when the pore is clogged and infection ensures. Some are bigger because the infection - and your bodies response to it - are " larger " ( e.g. more stuff ). Additiona…”
- Test id `1224`; true label `human`; 54 words; source `reddit_eli5`: “Clindamycin, for example, can cause oesophagitis ( inflammation of your oesophagus ) if not taken with a lot of water. Many other medications can cause that condition too, due to the consistency of the tablet or capsule …”
- Test id `17207`; true label `human`; 54 words; source `open_qa`: “Lobbying (also lobby) is the act of attempting to influence decisions made by officials in the government, most often legislators or members of regulatory agencies.Lobbying is done by many different types of people and o…”
- Test id `17263`; true label `human`; 46 words; source `open_qa`: “Wildlife such as ibex live in the higher peaks to elevations of, and plants such as Edelweiss grow in rocky areas in lower elevations as well as in higher elevations.At present the region is home to 14 million people and…”
- Test id `17568`; true label `human`; 27 words; source `open_qa`: “Castillo San Felipe del Morro also known as Fort San Felipe del Morro or Morro Castle, is a 16th-century citadel located in San Juan, Puerto Rico.…”

## Source breakdown

| Source | n | Any-model errors | False positives | False negatives |
|---|---:|---:|---:|---:|
| finance | 1,283 | 264 (20.6%) | 259 | 5 |
| medicine | 388 | 11 (2.8%) | 10 | 1 |
| open_qa | 710 | 139 (19.6%) | 72 | 67 |
| reddit_eli5 | 9,997 | 716 (7.2%) | 281 | 435 |
| wiki_csai | 254 | 78 (30.7%) | 69 | 9 |

## Robustness assessment

Do not use these models as standalone evidence that a response was AI-written. This evaluation is an HC3-derived, fixed test split and can reflect benchmark topic, source, preprocessing, and collection artifacts. Shared errors mark examples where ensembling by agreement offers no protection. Error-rate differences by length or formatting are observational; they do not establish causation. Before use beyond this benchmark, evaluate a time-separated and domain-separated external set, report class-specific false-positive rates at the intended threshold, inspect false positives and false negatives with human reviewers, and measure calibration and subgroup variation.

## Output files

- `tfidf_error_metrics.csv`: held-out metrics and confusion counts by model.
- `tfidf_error_predictions.csv`: every test response, label, all five predictions, error count/type, and length band.
- `tfidf_shared_mistakes.csv`: up to 50 examples misclassified by every model.
- `tfidf_error_analysis.md`: this report.
