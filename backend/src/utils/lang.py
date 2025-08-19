from langdetect import detect
from typing import Tuple

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def choose_mbart_codes(src_lang: str, mbart_map: dict) -> Tuple[str, str]:
    src = mbart_map.get(src_lang, "en_XX")
    tgt = mbart_map.get("en", "en_XX")
    return src, tgt
