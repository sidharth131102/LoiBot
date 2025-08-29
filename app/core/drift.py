import json
from pathlib import Path
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from app.models.loaders import embed_texts

BASE_PATH = Path("data/train.json")

_baseline_vec = None
_baseline_vocab = set()

def _load_baseline(n_samples: int = 2000):
    global _baseline_vec, _baseline_vocab
    if not BASE_PATH.exists():
        _baseline_vec = np.zeros((1, 768))
        _baseline_vocab = set()
        return

    texts = []
    try:
        with open(BASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            for rec in data[:n_samples]:
                if "reference" in rec:
                    texts.append(rec["reference"])
    except Exception:
        with open(BASE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "reference" in rec:
                        texts.append(rec["reference"])
                except Exception:
                    continue
                if len(texts) >= n_samples:
                    break

    if not texts:
        _baseline_vec = np.zeros((1, 768))
        _baseline_vocab = set()
        return

    _baseline_vec = embed_texts(texts)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words="english")
    vec.fit(texts)
    _baseline_vocab = set(vec.get_feature_names_out())

_load_baseline()

def _embedding_drift(new_emb: np.ndarray) -> float:
    if _baseline_vec is None or _baseline_vec.shape[0] == 0:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=min(8, len(_baseline_vec)), metric="cosine").fit(_baseline_vec)
    dists, _ = nbrs.kneighbors(new_emb)
    mean_dist = float(np.mean(dists))
    drift_pct = max(0.0, min(100.0, mean_dist * 100.0))
    return round(drift_pct, 2)

def _mine_new_concepts(text: str, top_k=4) -> List[str]:
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    X = vec.fit_transform([text])
    feats = vec.get_feature_names_out()
    scores = X.toarray()[0]
    idxs = scores.argsort()[::-1]
    out = []
    for i in idxs:
        term = feats[i]
        if term not in _baseline_vocab:
            out.append(term)
        if len(out) >= top_k:
            break
    return out

def drift_pipeline(text: str, top_k: int = 4) -> Dict[str, any]:
    emb = embed_texts([text])
    drift = _embedding_drift(emb)
    new_concepts = _mine_new_concepts(text, top_k=top_k)
    return {"drift_percentage": drift, "new_concepts": new_concepts}
