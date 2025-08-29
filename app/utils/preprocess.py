import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def detect_lang(t: str) -> str:
    try:
        code = detect(t)
        return code
    except Exception:
        return "en"

def truncate_for_model(t: str, max_chars=8000):
    return t[:max_chars]
