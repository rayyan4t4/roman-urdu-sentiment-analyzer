# Roman Urdu Sentiment Analyzer

Predicts sentiment (Positive / Negative / Neutral) of Roman Urdu text — the way Pakistanis actually write online, which English sentiment tools can't read.

## Data

- Source: [ayeshasameer/roman-urdu-sentiment-analysis](https://huggingface.co/datasets/ayeshasameer/roman-urdu-sentiment-analysis) (Hugging Face, MIT license, public — no API key needed)
- Splits: 33,619 train / 3,736 validation / 4,151 test (a few empty-after-cleaning rows dropped per split)
- Class balance (train): Neutral 16,247 (48%), Positive 9,303 (28%), Negative 8,069 (24%)
- Short texts: median 36 characters

## Method

Light cleaning (lowercase, strip URLs/mentions/hashtags, collapse whitespace) → TF-IDF word unigrams+bigrams (50k features, min_df=2, sublinear TF) → classifiers. LogisticRegression's C tuned on the validation set (grid 0.25–4.0, best C=2.0). All models evaluated on the held-out **test** split.

## Results (test set)

| Model | macro-F1 | Accuracy |
|---|---|---|
| LogisticRegression (C=2.0) | **0.6887** | **0.7099** |
| LinearSVC | 0.6862 | 0.7034 |
| MultinomialNB | 0.5856 | 0.6572 |
| Majority-class baseline | 0.2164 | 0.4806 |

Per-class F1 (LogisticRegression): Negative 0.6008, Neutral 0.7586, Positive 0.7067. Neutral dominates the data, so the model leans on it — Negative is the hardest class.

A transformer fine-tune (e.g. distilbert-base-multilingual-cased) was deliberately skipped: on CPU-only it would take hours for a marginal gain over TF-IDF, and the honest comparison here is the classic baseline suite.

## Train

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python train.py
```

Produces `tfidf_vectorizer.joblib`, `model.joblib`, `labels.json`, `metrics.json`.

## Run the app

```bash
./venv/bin/streamlit run streamlit_app.py
```

Text in → predicted sentiment + confidence bar chart. Note: the app needs a `no_proxy='localhost,127.0.0.1,::1'` workaround only at dataset download time; inference is local.
