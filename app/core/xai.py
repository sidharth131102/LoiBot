from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
import os

USE_SHAP = os.getenv("USE_SHAP", "false").lower() == "true"

def word_importance_tfidf(text: str, top_k: int = 8) -> Dict[str, List]:
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    X = vec.fit_transform([text])
    feats = vec.get_feature_names_out()
    scores = X.toarray()[0]
    idxs = scores.argsort()[::-1][:top_k]
    words = [feats[i] for i in idxs]
    vals = [float(scores[i]) for i in idxs]
    if vals and max(vals) > 0:
        m = max(vals)
        vals = [v/m for v in vals]
    return {"words": words, "scores": vals}

def get_word_importance(text: str, top_k: int = 8):
    return word_importance_tfidf(text, top_k=top_k)
