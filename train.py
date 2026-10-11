import json
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def frame(split, ds):
    df = ds[split].to_pandas()
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def main():
    ds = load_dataset("ayeshasameer/roman-urdu-sentiment-analysis")

    train_df = frame("train", ds)
    val_df = frame("validation", ds)
    test_df = frame("test", ds)

    labels = sorted(train_df["label"].unique())
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    for df in (train_df, val_df, test_df):
        df["y"] = df["label"].map(label_to_id)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(train_df["text"])
    X_val = vec.transform(val_df["text"])
    X_test = vec.transform(test_df["text"])
    y_train, y_val, y_test = train_df["y"].values, val_df["y"].values, test_df["y"].values

    def scores(name, clf, params=None):
        if params:
            clf.set_params(**params)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        return name, clf, {
            "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "per_class_f1": {lab: round(float(v), 4) for lab, v in zip(labels, f1_score(y_test, pred, average=None))},
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }

    results = {}
    models = {}

    name, clf, m = scores("majority", DummyClassifier(strategy="most_frequent"))
    results[name], models[name] = m, clf

    best, best_c = None, None
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(X_train, y_train)
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        if best is None or f1 > best:
            best, best_c, best_clf = f1, c, clf
    name, clf, m = scores("logistic_regression", best_clf)
    m["best_C"] = best_c
    results[name], models[name] = m, clf

    name, clf, m = scores("multinomial_nb", MultinomialNB())
    results[name], models[name] = m, clf

    name, clf, m = scores("linear_svc", LinearSVC())
    results[name], models[name] = m, clf

    joblib.dump(vec, "tfidf_vectorizer.joblib")
    joblib.dump(models["logistic_regression"], "model.joblib")
    with open("labels.json", "w") as f:
        json.dump(label_to_id, f)

    metrics = {
        "splits": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "class_distribution": {
            s: {str(k): int(v) for k, v in Counter(d["label"]).items()}
            for s, d in (("train", train_df), ("validation", val_df), ("test", test_df))
        },
        "label_mapping": label_to_id,
        "results": results,
        "transformer_stretch": "skipped: distilbert fine-tuning on 33k rows is too slow on CPU-only; TF-IDF baselines are the honest comparison here",
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in results.items()}, indent=2))
    print("best_C:", best_c)


if __name__ == "__main__":
    main()
