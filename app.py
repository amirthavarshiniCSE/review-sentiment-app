import json, pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from utils import predict_proba, LABELS

st.set_page_config(page_title="Review Sentiment Analysis", page_icon="🍓", layout="centered")

COLORS = {"Negative": "#c0503a", "Neutral": "#d9a82e", "Positive": "#4a8a60"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, label, p, button, textarea, input { font-family: Arial, 'Helvetica Neue', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* cream background with soft leaf / dot pattern */
.stApp {
  background-color: #fbf3dc;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cellipse cx='30' cy='30' rx='12' ry='5' fill='%23ece8c8'/%3E%3Cellipse cx='90' cy='78' rx='12' ry='5' fill='%23ece8c8'/%3E%3Ccircle cx='88' cy='22' r='3' fill='%23f2d9c9'/%3E%3Ccircle cx='28' cy='92' r='3' fill='%23f2d9c9'/%3E%3C/svg%3E");
}
.block-container { max-width: 900px; padding-top: 1.2rem; padding-bottom: 4rem; position: relative; z-index: 2; }

/* floating food decorations */
.deco { position: fixed; z-index: 1; pointer-events: none; opacity: 0.9; filter: drop-shadow(0 3px 4px rgba(120,80,30,.25)); }
.d1 { top: 14%; left: 3%;  font-size: 76px; transform: rotate(-12deg); }
.d2 { top: 28%; right: 3%; font-size: 70px; transform: rotate(8deg); }
.d3 { top: 50%; left: 4%;  font-size: 56px; transform: rotate(-8deg); }
.d4 { top: 72%; right: 3%; font-size: 80px; transform: rotate(12deg); }
.d5 { top: 42%; right: 2%; font-size: 34px; transform: rotate(20deg); }
.d6 { top: 84%; left: 3%;  font-size: 52px; transform: rotate(10deg); }
@media (max-width: 1250px) { .deco { display: none; } }

/* hero banner */
.hero { background: #c0503a; border-radius: 14px 14px 18px 18px; box-shadow: 0 8px 24px rgba(120,60,30,.25); overflow: hidden; margin-bottom: 1.6rem; }
.hero-in { padding: 2.2rem 2.5rem 2.2rem 2.5rem; }
.eyebrow { color: #f0b83a; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; }
.hero h1 { font-family: 'Newsreader', Georgia, serif; font-weight: 500; font-size: 3.2rem; line-height: 1.1; color: #fff; margin: 0.9rem 0 0.9rem 0; padding: 0; }
.hero p { color: #fbe9e3; font-size: 1rem; margin: 0; }
.stripe { height: 11px; background: linear-gradient(90deg, #e0b13a 0 33.3%, #4a8a60 33.3% 66.6%, #e0b13a 66.6% 100%); }

/* model card */
.st-key-modelcard { background: #faf1c8; border: 2px solid #e0c34f; border-radius: 12px; padding: 1.5rem 1.7rem; }
.badge { display: inline-block; background: #4a8a60; color: #fff; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; padding: 0.35rem 0.8rem; border-radius: 999px; }
.bestname { font-family: 'Newsreader', Georgia, serif; font-size: 1.7rem; color: #4a7a52; margin-top: 0.5rem; }
.bestf1 { font-size: 0.85rem; color: #6b6040; }
.st-key-modelcard label p { font-weight: 700; color: #3b3a30; }
.st-key-modelcard [data-baseweb="select"] > div { background: #ffffff; border: 1px solid #d9b93f; border-radius: 6px; }

/* section cards */
.st-key-sec1, .st-key-sec2 { background: #ffffff; border-top: 8px solid #c0503a; border-radius: 16px 16px 12px 12px; padding: 1.8rem 2rem 2rem 2rem; box-shadow: 0 8px 22px rgba(120,80,30,.12); }
.sec { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.4rem; }
.num { width: 38px; height: 38px; min-width: 38px; background: #c0503a; color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Newsreader', serif; font-weight: 600; }
.sec-title { font-family: 'Newsreader', Georgia, serif; font-size: 1.8rem; color: #2a2623; }
.sec-desc { color: #55504a; margin: 0 0 1.4rem 3.4rem; font-size: 1rem; }
.fieldlab { font-weight: 700; font-size: 0.92rem; color: #2f2b27; margin-bottom: 0.3rem; }

textarea { background: #fffaf0 !important; color: #2a2623 !important; -webkit-text-fill-color: #2a2623 !important; caret-color: #c0503a !important; border: 1px solid #ecdfbd !important; border-radius: 8px !important; }
textarea::placeholder { color: #9a8f7a !important; -webkit-text-fill-color: #9a8f7a !important; opacity: 1 !important; }
[data-baseweb="textarea"], [data-baseweb="textarea"] > div { background: #fffaf0 !important; }
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] input { color: #2a2623 !important; -webkit-text-fill-color: #2a2623 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #3b3a30 !important; }
[data-testid="stFileUploaderDropzone"] button { background: #c0503a !important; border: none !important; border-radius: 8px !important; transition: all .2s ease; }
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; fill: #ffffff !important; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] button:hover { background: #cf5a42 !important; transform: translateY(-1px); box-shadow: 0 0 8px rgba(255,170,90,.9), 0 0 22px rgba(240,120,60,.6); }
[data-testid="stCaptionContainer"], .stMarkdown, .stMarkdown p, label p { color: #2f2b27; }
textarea:focus { border-color: #c0503a !important; box-shadow: 0 0 0 2px rgba(192,80,58,.2) !important; }

/* example chips */
.st-key-ex_pos button, .st-key-ex_neg button, .st-key-ex_sar button, .st-key-ex_mix button {
  border-radius: 999px; font-weight: 700; font-size: 0.92rem; padding: 0.3rem 1.1rem; border: 1px solid transparent; transition: all .2s ease;
}
.st-key-ex_pos button { background: #dfeee3; color: #2f6b45; }
.st-key-ex_neg button { background: #f6dcd5; color: #b5482f; }
.st-key-ex_sar button { background: #faf0c2; color: #7a5d00; border-color: #d9b93f; }
.st-key-ex_mix button { background: #ece8df; color: #4f4a42; }
.st-key-ex_pos button:hover { box-shadow: 0 0 14px rgba(74,138,96,.65); }
.st-key-ex_neg button:hover { box-shadow: 0 0 14px rgba(192,80,58,.6); }
.st-key-ex_sar button:hover { box-shadow: 0 0 14px rgba(217,168,46,.8); }
.st-key-ex_mix button:hover { box-shadow: 0 0 12px rgba(120,110,95,.5); }
.stButton button:hover { transform: translateY(-1px); }

/* glowing primary button */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: #c0503a; color: #fff; border: none; border-radius: 8px; padding: 0.7rem 1.6rem; font-weight: 700;
  animation: glow 2.6s ease-in-out infinite; transition: transform .15s ease, box-shadow .2s ease;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: #cf5a42; color: #fff; transform: translateY(-2px);
  box-shadow: 0 0 8px rgba(255,170,90,.95), 0 0 28px rgba(240,120,60,.75);
}
@keyframes glow {
  0%, 100% { box-shadow: 0 0 6px rgba(240,130,70,.35), 0 0 14px rgba(240,130,70,.25); }
  50%      { box-shadow: 0 0 12px rgba(250,160,80,.65), 0 0 28px rgba(240,120,60,.45); }
}
[data-testid="stDownloadButton"] button { border: 1px solid #d9b93f; background: #fff8dc; color: #6b5200; font-weight: 600; border-radius: 8px; }
[data-testid="stDownloadButton"] button:hover { box-shadow: 0 0 14px rgba(217,168,46,.7); }
[data-testid="stFileUploaderDropzone"] { background: #fffaf0; border: 1px dashed #d9b93f; border-radius: 10px; }

/* result + stats */
.res { margin-top: 1.4rem; padding: 1.1rem 1.4rem; border-radius: 10px; background: #fffaf0; border: 1px solid #ecdfbd; }
.small { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em; color: #7a7061; }
.big { font-family: 'Newsreader', Georgia, serif; font-size: 2.4rem; margin: 0.1rem 0 0.8rem 0; }
.row { display: flex; align-items: center; gap: 0.8rem; margin: 0.4rem 0; font-size: 0.92rem; }
.lab { width: 4.6rem; color: #3b3a30; }
.track { flex: 1; height: 9px; background: #f0e8d0; border-radius: 5px; overflow: hidden; }
.fill { height: 100%; border-radius: 5px; }
.pct { width: 2.8rem; text-align: right; color: #3b3a30; font-variant-numeric: tabular-nums; }
.stat { border-top: 3px solid #2a2623; padding-top: 0.5rem; }
.stat .n { font-family: 'Newsreader', serif; font-size: 2rem; }
.stat .t { font-size: 0.82rem; color: #6b6558; }
h4 { font-family: 'Newsreader', Georgia, serif !important; font-weight: 500 !important; }
</style>
<div class="deco d1">🍕</div><div class="deco d2">🍔</div><div class="deco d3">🍓</div>
<div class="deco d4">🍩</div><div class="deco d5">🍋</div><div class="deco d6">🥑</div>
""", unsafe_allow_html=True)

config = json.load(open("config.json"))
MAX_LEN = config["max_length"]
names = config["models"]
best = config["best_model"]

@st.cache_resource
def get_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_model(name):
    return load_model(f"models/{name}.keras")

tokenizer = get_tokenizer()

# ---------- Hero ----------
st.markdown(
    '<div class="hero"><div class="hero-in"><div class="eyebrow">Customer Review Sentiment Analysis</div>'
    '<h1>Reading what customers say</h1>'
    '<p>Classifies reviews as Negative, Neutral, or Positive using a recurrent neural network.</p></div>'
    '<div class="stripe"></div></div>', unsafe_allow_html=True)

# ---------- Model chooser ----------
with st.container(key="modelcard"):
    left, right = st.columns([2.2, 1])
    with left:
        model_name = st.selectbox("Choose a model", names, index=names.index(best),
                                  format_func=lambda m: f"{m} (best)" if m == best else m)
    with right:
        f1 = config.get("best_f1")
        f1_html = f'<div class="bestf1">Test macro F1: {f1:.3f}</div>' if isinstance(f1, (int, float)) else ""
        st.markdown(f'<div style="text-align:right"><span class="badge">BEST MODEL</span>'
                    f'<div class="bestname">{best}</div>{f1_html}</div>', unsafe_allow_html=True)
model = get_model(model_name)

st.write("")

EXAMPLES = {
    "ex_pos": ("Clear positive", "The coffee is fresh and rich, and it arrived on time."),
    "ex_neg": ("Negation", "The product is not good at all, I would not recommend it."),
    "ex_sar": ("Sarcasm", "Great, another product that broke after one day."),
    "ex_mix": ("Mixed", "Tastes fine but it is too expensive for what you get."),
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
        f'<div class="res" style="border-left:5px solid {COLORS[label]}">'
        f'<div class="small">Predicted sentiment</div>'
        f'<div class="big" style="color:{COLORS[label]}">{label}</div>{bars}</div>',
        unsafe_allow_html=True)

# ---------- Section 1 ----------
with st.container(key="sec1"):
    st.markdown('<div class="sec"><div class="num">1</div><div class="sec-title">Single review prediction text box</div></div>'
                '<div class="sec-desc">Type or paste one review to see its predicted sentiment and the model\'s confidence for each class.</div>'
                '<div class="fieldlab">Review text</div>', unsafe_allow_html=True)
    review = st.text_area("Review text", key="review_text", height=130, label_visibility="collapsed",
                          placeholder="For example: Not bad at all, I would buy it again.")
    cols = st.columns([1.35, 1, 1, 1, 2.2])
    for c, (k, (name, text)) in zip(cols, EXAMPLES.items()):
        c.button(name, on_click=set_text, args=(text,), key=k)
    st.write("")
    if st.button("Analyse sentiment", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            probs = predict_proba([review], model, tokenizer, MAX_LEN)[0]
            result_card(LABELS[int(np.argmax(probs))], probs)

st.write("")

# ---------- Section 2 ----------
with st.container(key="sec2"):
    st.markdown('<div class="sec"><div class="num">2</div><div class="sec-title">Batch CSV dashboard</div></div>'
                '<div class="sec-desc">Upload a CSV with the columns Product and Review to see sentiment trends for each product.</div>',
                unsafe_allow_html=True)
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
