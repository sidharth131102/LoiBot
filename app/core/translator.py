from typing import Optional
from app.models.loaders import get_mbart50, mbart_lang_code
import torch

def translate_pipeline(text: str, src_lang: Optional[str] = "auto", tgt_lang: str = "en"):
    tok, model = get_mbart50()
    if src_lang == "auto" or src_lang is None:
        src_lang = "en"
    try:
        tok.src_lang = mbart_lang_code(src_lang)
    except Exception:
        pass
    forced_bos = tok.convert_tokens_to_ids(mbart_lang_code(tgt_lang))
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    gen = model.generate(**enc, forced_bos_token_id=forced_bos, num_beams=4, max_new_tokens=256)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]
