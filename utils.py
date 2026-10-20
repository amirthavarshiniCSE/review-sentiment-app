import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

LABELS = ["Negative", "Neutral", "Positive"]

NEGATIONS = {"not", "never", "no", "nothing", "neither", "nor", "hardly", "barely", "without"}
CLAUSE_BREAKS = {"but", "however", "although", "though", "yet"}
PUNCT = set(".,!?;:")

def clean_text(text, window=3):
    """Lowercase, strip HTML/noise, expand n't contractions and mark negation scope.
    'this is not good, but cheap' -> 'this is not not_good but cheap'
    The next `window` words after a negation word get a 'not_' prefix, stopping at
    punctuation or a contrast word such as 'but'."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"http\S+", " ", text)          # URLs
    text = text.replace("won't", "will not").replace("can't", "can not").replace("cannot", "can not")
    text = re.sub(r"n't\b", " not", text)         # don't, isn't, didn't ...
    text = text.replace("'", "")
    tokens = re.findall(r"[a-z0-9]+|[.,!?;:]", text)
    out, remaining = [], 0
    for tok in tokens:
        if tok in PUNCT or tok in CLAUSE_BREAKS:
            remaining = 0
            if tok in CLAUSE_BREAKS:
                out.append(tok)
            continue
        if tok in NEGATIONS:
            out.append(tok)
            remaining = window
        elif remaining > 0:
            out.append("not_" + tok)
            remaining -= 1
        else:
            out.append(tok)
    return " ".join(out)

def predict_proba(texts, model, tokenizer, max_length, batch_size=256):
    """Returns an (n, 3) array of class probabilities."""
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=max_length, padding="post", truncating="post")
    return model.predict(padded, batch_size=batch_size, verbose=0)

def predict_labels(texts, model, tokenizer, max_length):
    probs = predict_proba(texts, model, tokenizer, max_length)
    return [LABELS[i] for i in probs.argmax(axis=1)], probs
