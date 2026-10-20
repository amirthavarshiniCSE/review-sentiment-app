import json, pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from utils import predict_proba, LABELS

st.set_page_config(page_title="Review Sentiment", layout="centered")

COLORS = {"Negative": "#b3261e", "Neutral": "#9a7b1c", "Positive": "#2e7d32"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; }
h1 { font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.sub { color: #6b6b66; margin-bottom: 1.8rem; font-size: 0.95rem; }
.card { background: #ffffff; border: 1px solid #e4e1d8; border-radius: 6px; padding: 1.2rem 1.4rem; margin-top: 1rem; }
.small { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b6b66; }
.big { font-size: 1.9rem; font-weight: 600; margin: 0.1rem 0 0.9rem 0; }
.row { display: flex; align-items: center; gap: 0.8rem; margin: 0.35rem 0; font-size: 0.9rem; }
.lab { width: 4.5rem; color: #3b3b37; }
.track { flex: 1; height: 8px; background: #efede6; border-radius: 4px; overflow: hidden; }
.fill { height: 100%; border-radius: 4px; }
.pct { width: 2.8rem; text-align: right; color: #3b3b37; font-variant-numeric: tabular-nums; }
.stat { border-top: 2px solid #1d1d1b; padding-top: 0.5rem; }
.stat .n { font-size: 1.6rem; font-weight: 600; }
.stat .t { font-size: 0.8rem; color: #6b6b66; }
</style>
""", unsafe_allow_html=True)

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

names = config["models"]
with st.sidebar:
    st.markdown("**Model**")
    model_name = st.selectbox("Model", names, index=names.index(config["best_model"]), label_visibility="collapsed")
    st.caption(f"Best on validation data: {config['best_model']}")
model = get_model(model_name)

st.title("Review Sentiment")
st.markdown('<div class="sub">Classifies product reviews as positive, neutral or negative. '
            'Trained on the Amazon Fine Food Reviews dataset.</div>', unsafe_allow_html=True)

EXAMPLES = {
    "Positive": "The coffee is fresh and rich, and it arrived on time.",
    "Negation": "The product is not good at all, I would not recommend it.",
    "Sarcasm": "Great, another product that broke after one day.",
}
st.session_state.setdefault("review_text", "")
def set_text(t):
    st.session_state["review_text"] = t

def result_card(label, probs):
    bars = "".join(
        f'<div class="row"><span class="lab">{l}</span><div class="track">'
        f'<div class="fill" style="width:{p*100:.1f}%;background:{COLORS[l]}"></div></div>'
        f'<span class="pct">{p:.0%}</span></div>' for l, p in zip(LABELS, probs))
    st.markdown(
        f'<div class="card" style="border-left:4px solid {COLORS[label]}">'
        f'<div class="small">Predicted sentiment</div>'
        f'<div class="big" style="color:{COLORS[label]}">{label}</div>{bars}</div>',
        unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single review", "Batch analysis"])

with tab1:
    st.caption("Try an example")
    cols = st.columns(len(EXAMPLES))
    for c, (name, text) in zip(cols, EXAMPLES.items()):
        c.button(name, on_click=set_text, args=(text,), use_container_width=True)

    review = st.text_area("Review", key="review_text", height=140,
                          label_visibility="collapsed", placeholder="Type or paste a review")
    if st.button("Analyse", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            probs = predict_proba([review], model, tokenizer, MAX_LEN)[0]
            result_card(LABELS[int(np.argmax(probs))], probs)

with tab2:
    st.caption("Upload a CSV with two columns: Product and Review.")
    template = pd.DataFrame({"Product": ["Green Tea"], "Review": ["Excellent taste and quality"]})
    st.download_button("Download CSV template", template.to_csv(index=False), "template.csv", "text/csv")
    file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if file is not None:
        df = pd.read_csv(file)
        if not {"Product", "Review"}.issubset(df.columns):
            st.error("The file needs the columns Product and Review.")
        else:
            df = df.dropna(subset=["Review"]).copy()
            with st.spinner("Analysing reviews"):
                probs = predict_proba(df["Review"].astype(str).tolist(), model, tokenizer, MAX_LEN)
            df["Predicted Sentiment"] = [LABELS[i] for i in probs.argmax(axis=1)]
            df["Confidence"] = probs.max(axis=1).round(3)

            tot = len(df)
            stats = st.columns(4)
            stats[0].markdown(f'<div class="stat"><div class="n">{tot}</div><div class="t">Reviews</div></div>', unsafe_allow_html=True)
            for col, lab in zip(stats[1:], LABELS[::-1]):
                n = int((df["Predicted Sentiment"] == lab).sum())
                col.markdown(f'<div class="stat" style="border-color:{COLORS[lab]}"><div class="n">{n}</div>'
                             f'<div class="t">{lab} ({n/tot:.0%})</div></div>', unsafe_allow_html=True)

            st.markdown("#### Sentiment by product")
            counts = pd.crosstab(df["Product"], df["Predicted Sentiment"]).reindex(columns=LABELS, fill_value=0)
            st.bar_chart(counts, color=[COLORS[l] for l in LABELS])

            st.markdown("#### Products ranked by share of negative reviews")
            pct = counts.div(counts.sum(axis=1), axis=0)
            rank = pd.DataFrame({"Reviews": counts.sum(axis=1), "Negative share": (pct["Negative"] * 100).round(0)})
            rank = rank.sort_values("Negative share", ascending=False)
            st.dataframe(rank, use_container_width=True,
                         column_config={"Negative share": st.column_config.ProgressColumn(
                             "Negative share", format="%.0f%%", min_value=0, max_value=100)})

            st.markdown("#### All predictions")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
