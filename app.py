import json, pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from utils import predict_proba, LABELS

st.set_page_config(page_title="Review Sentiment Dashboard", page_icon="🛒", layout="wide")

config = json.load(open("config.json"))
MAX_LEN = config["max_length"]

@st.cache_resource
def get_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_model(name):
    return load_model(f"models/{name}.keras")

tokenizer = get_tokenizer()

st.title("🛒 Customer Review Sentiment Analysis")
st.caption("LSTM / BiLSTM / GRU models trained on Amazon Fine Food Reviews")

names = config["models"]
model_name = st.sidebar.selectbox("Model", names, index=names.index(config["best_model"]))
model = get_model(model_name)
st.sidebar.info(f"Best model on validation data: **{config['best_model']}**")

tab1, tab2 = st.tabs(["✍️ Single review", "📁 CSV batch dashboard"])

with tab1:
    review = st.text_area("Enter a product review", height=150,
                          placeholder="e.g. The coffee tastes great but the delivery was very late.")
    if st.button("Predict sentiment", type="primary"):
        if not review.strip():
            st.error("Please type a review first.")
        else:
            probs = predict_proba([review], model, tokenizer, MAX_LEN)[0]
            label = LABELS[int(np.argmax(probs))]
            icon = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}[label]
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicted sentiment", f"{icon} {label}", f"{probs.max():.1%} confidence")
            c2.bar_chart(pd.DataFrame({"Probability": probs}, index=LABELS))

with tab2:
    st.write("Upload a CSV with columns **Product** and **Review**.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if not {"Product", "Review"}.issubset(df.columns):
            st.error("The CSV must contain the columns 'Product' and 'Review'.")
        else:
            df = df.dropna(subset=["Review"]).copy()
            with st.spinner("Predicting..."):
                probs = predict_proba(df["Review"].astype(str).tolist(), model, tokenizer, MAX_LEN)
            df["Predicted Sentiment"] = [LABELS[i] for i in probs.argmax(axis=1)]
            df["Confidence"] = probs.max(axis=1).round(3)

            counts = pd.crosstab(df["Product"], df["Predicted Sentiment"]).reindex(columns=LABELS, fill_value=0)
            pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Reviews", len(df))
            for col, lab in zip((m2, m3, m4), LABELS):
                col.metric(lab, int((df["Predicted Sentiment"] == lab).sum()))

            left, right = st.columns([2, 1])
            with left:
                st.subheader("Product-wise sentiment")
                st.bar_chart(counts)
            with right:
                st.subheader("Needs attention")
                st.caption("Products ranked by share of negative reviews")
                st.dataframe(pct.sort_values("Negative", ascending=False)[["Negative"]].rename(columns={"Negative": "% negative"}))

            st.subheader("Percentage breakdown")
            st.dataframe(pct)
            st.subheader("Predictions")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download predictions (CSV)", df.to_csv(index=False), "predictions.csv", "text/csv")
