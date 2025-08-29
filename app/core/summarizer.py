from typing import Dict, Any, List
from app.utils.preprocess import clean_text, detect_lang, truncate_for_model
from app.models.loaders import (
    get_legal_pegasus, get_lexlm_bart, run_seq2seq,
    embed_with_legalbert
)
from app.core.retriever import retrieve_context
from app.core.translator import translate_pipeline
from app.core.xai import get_word_importance
from sklearn.metrics.pairwise import cosine_similarity


# ---------- NEW: helper function ----------
def chunk_text(text: str, chunk_size=800, overlap=100) -> List[str]:
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks, start = [], 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _build_context_block(text: str, references: List[str]) -> str:
    ctx = "\n\n[Retrieved References]\n" + "\n".join(references)
    return text + ctx


def _rank_summaries(candidates: Dict[str, str], src_text: str) -> str:
    keys = list(candidates.keys())
    vals = list(candidates.values())
    src_emb = embed_with_legalbert([src_text])   # ✅ use Legal-BERT for ranking
    sum_emb = embed_with_legalbert(vals)
    sims = cosine_similarity(src_emb, sum_emb)[0]
    best_idx = int(sims.argmax())
    return keys[best_idx]


def summarize_en(text: str) -> Dict[str, Any]:
    # 1. Retrieval
    rag = retrieve_context(text, top_k=5)
    refs = [r["reference"] for r in rag["references"]]

    text_with_ctx = _build_context_block(text, refs)
    text_with_ctx = truncate_for_model(text_with_ctx, max_chars=8000)

    # 2. Chunking
    chunks = chunk_text(text_with_ctx, chunk_size=800, overlap=100)

    peg = get_legal_pegasus()
    tok2, m2 = get_lexlm_bart()

    summaries_peg, summaries_lex = [], []

    for chunk in chunks:
        try:
            s1 = peg(chunk, max_length=256, min_length=64, do_sample=False)[0]["summary_text"]
        except Exception as e:
            s1 = f"[Pegasus error: {str(e)}]"
        try:
            s2 = run_seq2seq(tok2, m2, chunk, max_new_tokens=256)
        except Exception as e:
            s2 = f"[LexLM error: {str(e)}]"

        summaries_peg.append(s1)
        summaries_lex.append(s2)

    # 3. Combine summaries → final candidates
    combined_peg = " ".join(summaries_peg)
    combined_lex = " ".join(summaries_lex)

    candidates = {"legal_pegasus": combined_peg, "lexlm_hybrid": combined_lex}

    best_key = _rank_summaries(candidates, text)
    best_summary = candidates[best_key]

    # 4. XAI explanation (✅ fixed structured output)
    raw_expl = get_word_importance(text, top_k=8)  # e.g. [("party", 0.12), ...]
    explanation = {
        "words": [w for w, _ in raw_expl],
        "scores": [float(s) for _, s in raw_expl]
    }

    return {
        "summary": best_summary,
        "candidates": candidates,
        "retrieved_refs": refs,
        "explanation": explanation
    }


def summarize_non_en(text: str, lang: str) -> Dict[str, Any]:
    en_text = translate_pipeline(text, src_lang=lang, tgt_lang="en")
    result = summarize_en(en_text)
    return result


def summarize_pipeline(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    lang = detect_lang(text)
    if lang == "en":
        return summarize_en(text)
    else:
        return summarize_non_en(text, lang=lang)
