import json
import re

import joblib
import streamlit as st


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load():
    vec = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("model.joblib")
    labels = json.load(open("labels.json"))
    id_to_label = {v: k for k, v in labels.items()}
    return vec, model, id_to_label


vec, model, id_to_label = load()

st.title("Roman Urdu Sentiment Analyzer")
st.write("Type a sentence in Roman Urdu — the model predicts Positive, Negative, or Neutral.")

text = st.text_input("Your text", placeholder="yeh drama bohat acha hai")

if st.button("Analyze") and text.strip():
    cleaned = clean(text)
    proba = model.predict_proba(vec.transform([cleaned]))[0]
    pred = int(proba.argmax())
    st.subheader(f"{id_to_label[pred]} ({proba[pred]:.1%} confidence)")
    st.bar_chart({id_to_label[i]: float(p) for i, p in enumerate(proba)})
