# Spot the Bot: Detecting AI-Generated Text

**Company / Org:** Chewy.com  
**Challenge Advisor:** Rishabh Jain, rishab1300@gmail.com   
**AI Studio Coach:** Anshul Rehpade, anshul.rehpade@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Chewy.com

Chewy.com is a leading online retailer in the pet supplies industry, focusing on delivering high-quality products and services for pet owners. Our commitment to customer satisfaction drives our innovative approach across various departments, including technology and customer support.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use a public corpus of paired human-written and AI-generated (ChatGPT) answers — the HC3 dataset, spanning five domains (everyday/open-domain Q&A, Reddit ELI5 explanations, finance, medicine, and Wikipedia computer-science topics) and natural language processing with supervised machine learning and deep learning (TF-IDF feature engineering, logistic regression, and a feedforward neural network built in Keras), followed by an adversarial-robustness evaluation to build a classifier that distinguishes human-written from AI-generated text and to measure how well it holds up when the AI text is lightly paraphrased to evade detection. This will help address the growing challenge of content authenticity and academic integrity, determining whether text was written by a person or a machine and reveal where automated AI-text detectors fail in the real world.

### Success Criteria
- Primary metric: strong classification performance on the held-out test set — macro-F1 and accuracy, reported against a majority-class baseline, with precision/recall and a confusion matrix.   
- Distinctive success criterion: a robustness curve showing how detector performance degrades when AI text is paraphrased, plus a written analysis of why it breaks (the real-world insight).
Reproducibility: a clean, well-documented Colab notebook and public GitHub repo the team can share on LinkedIn, with a short final report covering method, results, and limitations (including dataset drift — see stretch/notes).   
- Real-world demo (the capstone): a working, publicly viewable web app — a Streamlit frontend calling a lightweight API that serves the trained model — where anyone can paste text and get a "human vs. AI" prediction with a confidence score. Deployed free on Streamlit Community Cloud or Hugging Face Spaces, this is the piece that makes the project feel like a real product and showcases end-to-end ML skills (serving + frontend) on the students' portfolios.

### Stretch Goals
- Adversarial generation: use a small/open LLM to generate the paraphrased "humanized" test cases automatically, rather than hand-crafting them.
- Dataset-drift test: HC3 was built on an early-2023 model (GPT-3.5). Collect a small modern sample and test whether the detector still works — a great lesson in why detectors go stale.
Bias/fairness check: measure whether the detector performs unevenly across HC3's five domains (medicine, finance, open-domain Q&A, Reddit ELI5, Wikipedia CS), tying into the course's bias-mitigation module.   
- Pretrained embeddings: swap TF-IDF for transformer sentence-embeddings and quantify the lift.   
- Make the demo interactive/explainable: in the deployed app, let users paraphrase text and watch the prediction flip in real time, or highlight which words pushed the model toward "AI" vs. "human" — turning the robustness finding into something a viewer can actually feel.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data understanding & preparation | Load the HC3 dataset from Hugging Face; run EDA comparing human vs. ChatGPT answers (length, vocabulary richness, punctuation/structure patterns); clean and normalize text; build a stratified train/validation/test split; engineer baseline features (TF-IDF). Deliverable: a documented, reproducible data-prep notebook plus an EDA summary. |
| **October** | Modeling & evaluation (baseline → deep) | Train and compare classic classifiers from the course (logistic regression as the primary baseline, with decision tree / k-NN for comparison) and establish the evaluation harness (accuracy, precision, recall, macro-F1, confusion matrix); then build a feedforward neural network in Keras and compare it against the baselines. Run error analysis to see which texts fool the models, and export/save the best model so it's ready to be served. Deliverable: a working detector (classic + neural) with a clear metrics report and a saved model file. |
| **November** | Robustness experiment + deploy it as a real app (the novel core) | First, the key experiment: test the detector against paraphrased / "humanized" AI text and measure how much accuracy/F1 drops, plus a per-domain breakdown (does it hold up on medical vs. legal vs. everyday text?). Then take it to production: wrap the saved model in a lightweight API (e.g., FastAPI) and build a Streamlit frontend that calls it — a simple page where a user pastes text and instantly sees a "human vs. AI" verdict with a confidence score — deployed for free on Streamlit Community Cloud or Hugging Face Spaces. This gives students hands-on experience with model serving and frontend–backend integration, the end-to-end workflow they'll see in industry. Deliverable: a robustness analysis with failure-mode write-up, plus a live demo app. (Scope note: the deploy step sits on top of an already-complete model, so it can't endanger the core result. If time runs short, the simplest viable version is a self-contained Streamlit app that loads the model directly — no separate API — so the team always ships a working demo.) |

**Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.


## 📊 Dataset

**Name and Source:** HC3 — Human ChatGPT Comparison Corpus (English), available via Hugging Face  
**Format:** JSONL on the Hub (auto-converted to Parquet); we provide a prepared CSV via a script  
**Size:** ~147 MB (≈24,300 questions; ~85k+ answers once expanded) — well under 1 GB  
**Location:** https://huggingface.co/datasets/Hello-SimpleAI/HC3  
**In this repo:** see the [`data/`](data) folder — a small committed preview (`data/hc3_sample.csv`), a `download_hc3.py` script to pull the full set, and a [data dictionary](data/README.md).

### How to get the data
A tiny 42-row preview is committed so you can see the format immediately. To pull the full set look into data/README.md

### Key details — limitations & preprocessing
- **Explode the nested lists first.** `human_answers` and `chatgpt_answers` are *lists* per question — flatten to one row per answer with a `0/1` label before modeling. 
- **Split by question, not by answer.** The same question appears on both the human and AI side, so an answer-level random split leaks topics across train/test and inflates scores. Group by `question` (or `id`) when splitting.
- **Length is a confound.** ChatGPT answers are systematically longer and more uniform, so a classifier can "cheat" by learning length instead of AI-ness — exactly the weakness the paraphrase-robustness experiment is meant to expose. Watch for it in EDA and error analysis.
- **Model drift.** The AI answers are GPT-3.5-era (early 2023); a detector trained on HC3 may not generalize to newer models.
- **Documentation:** [paper (arXiv:2301.07597)](https://arxiv.org/abs/2301.07597)


## 🛠️ Suggested Approach

**ML Problem Type:** Classification, NLP, Deep Learning / Neural Networks, Large Language Models (LLMs)/ Generative AI, Transfer Learning / Pre-trained Models

**Recommended Libraries:** *(all free, all run on Google Colab — no local setup or GPU required)*

| Phase | Libraries | What you'll use them for |
|-------|-----------|--------------------------|
| Data & EDA | `datasets` (Hugging Face), `pandas`, `numpy`, `matplotlib` / `seaborn` | Pull HC3 from the Hub, explode the answer lists, and chart human-vs-AI differences (length, vocabulary, punctuation). |
| Text features | `scikit-learn` (`TfidfVectorizer`, `CountVectorizer`), `nltk` *(optional)* | Turn text into numeric features (TF-IDF); optional tokenization/stopword tools. |
| Classic models | `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`) | The baseline detectors and the primary logistic-regression model. |
| Deep model | `tensorflow` / `keras` | The feedforward neural network to compare against the baselines. |
| Evaluation | `scikit-learn.metrics`, `matplotlib` | Accuracy, precision/recall, macro-F1, confusion matrix, and the robustness curve. |
| Save / serve | `joblib` or `pickle`, `fastapi` *(optional)*, `streamlit` | Save the trained model, wrap it in an API, and build the demo web app. |
| Stretch (embeddings / paraphrasing) | `sentence-transformers`, `transformers` | Swap TF-IDF for transformer embeddings; generate "humanized"/paraphrased test cases. |

> Start simple: `pandas` + `scikit-learn` gets you a working detector. Only add Keras once the baseline works, and treat `sentence-transformers`/`transformers` as stretch goals.

**Evaluation Metrics:**
- **Macro-F1** — the primary metric. Averages the F1 of the "human" and "AI" classes equally, so the model can't win just by favoring the bigger class.
- **Accuracy** — easy to read, but always report it *next to a majority-class baseline* (predicting the most common label) so you can see the real lift.
- **Precision & Recall (per class)** — precision = "when it says AI, how often is it right"; recall = "of all AI text, how much did it catch." Report both for each class.
- **Confusion matrix** — the 2×2 grid of human/AI predictions vs. truth; the fastest way to see *how* the model fails.
- **Robustness curve (the distinctive metric)** — accuracy / macro-F1 measured as the AI text is progressively paraphrased, plus a **per-domain breakdown** (does it hold up on medicine vs. finance vs. ELI5?). This is the real-world insight the project is built around.
- *(Optional, for a probability-based view)* **ROC-AUC** — how well the model's confidence separates the two classes regardless of threshold.


## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [AI Detectors Are Biased Against Non-Native English Writers (Stanford HAI)](https://hai.stanford.edu/news/ai-detectors-biased-against-non-native-english-writers) — a short, accessible explainer on *why* AI-text detectors are unreliable and unfair; sets up the exact real-world question your robustness experiment investigates.
- [OpenAI Scuttles Its AI-Written Text Detector Over "Low Rate of Accuracy" (TechCrunch)](https://techcrunch.com/2023/07/25/openai-scuttles-ai-written-text-detector-over-low-rate-of-accuracy/) — a real-world case study: even OpenAI shut down its own detector, motivating why measuring where detectors *fail* is the interesting part of this project.

**Technical Tutorials:**
- [Working With Text Data (scikit-learn tutorial)](https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html) — walks through the core pipeline for this project: turning text into TF-IDF features and training/evaluating a classifier.
- [TfidfVectorizer API reference (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) — the official docs for the exact tool you'll use to build features; see `ngram_range`, `min_df`, `max_features`, and `stop_words`.
- [Basic Text Classification (TensorFlow/Keras tutorial)](https://www.tensorflow.org/tutorials/keras/text_classification) — an end-to-end binary text classifier in Keras, the template for the feedforward neural network you'll compare against the baselines.
- [Get Started with Streamlit (official docs)](https://docs.streamlit.io/get-started) — install, build your first app, and run it; this is how you'll turn the trained model into the live "human vs. AI" demo.

**Code Examples:**
- [HC3 dataset card on Hugging Face](https://huggingface.co/datasets/Hello-SimpleAI/HC3) — the dataset home page with ready-to-copy `datasets` loading code and a description of every field you'll work with.
- [Hello-SimpleAI/chatgpt-comparison-detection (GitHub)](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) — the official HC3 repo with the authors' own detectors (QA, single-text, and linguistic-feature versions) as reference implementations.
- [Host a Streamlit app on Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit) — a complete starter example (a small classifier app) showing how to deploy your demo for free, including the `app.py` and `requirements.txt` setup.

**Other:**
- [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection (arXiv:2301.07597)](https://arxiv.org/abs/2301.07597) — the HC3 paper: how the dataset was built, what distinguishes human vs. ChatGPT answers, and the detection experiments your project extends.

*Feel free to explore beyond these, and share anything interesting you find with me!*


## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* **Discord** — message me in our team channel within Break Through Tech's Discord space.
* **Email** — rishab1300@gmail.com (please copy your teammates and AI Studio Coach).
* I'll try to respond as soon as possible. For urgent questions, please also reach out to your AI Studio Coach.

**Recommended free coding / collaboration tools**
All free and runnable on a school laptop — no paid services or GPU purchase required.

**Coding & compute**
- **Google Colab** — free cloud notebooks (with GPU) for EDA, feature engineering, and train
ing; opens directly from GitHub, so there's nothing to install locally.
- **VS Code** *(optional)* — a full editor for writing the Streamlit / API app instead of no
tebook cells.

**Core Python libraries**
- **pandas / numpy** — load and reshape the HC3 data.
- **scikit-learn** — TF-IDF features and the baseline models (logistic regression, decision
tree, k-NN) plus all evaluation metrics.
- **TensorFlow / Keras** — the feedforward neural network.
- **matplotlib / seaborn** — EDA charts, the confusion matrix, and the robustness curve.
- **Hugging Face `datasets`** — download the HC3 dataset from the Hub.

**Deployment / hosting (the demo)**
- **Streamlit** + **Streamlit Community Cloud** — build and host the "paste text → human vs.
 AI" app for free.

**Collaboration**
- **Discord** — the team's day-to-day chat channel.
- **GitHub Issues + Projects** — task tracking and weekly planning.


## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
