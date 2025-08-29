import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"
LEGAL_PEGASUS = "nsi319/legal-pegasus"
LEXLM_HYBRID = "MikaSie/LexLM_Longformer_BART_hybrid_V1"
MBART50 = "facebook/mbart-large-50-many-to-many-mmt"

DEVICE = 0 if torch.cuda.is_available() else -1

# -------- Retrieval Embeddings (for Pinecone) --------
_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(texts):
    """Embed texts with mpnet (for Pinecone ingestion + query)."""
    return _mpnet.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# -------- Legal-BERT (used only for ranking + keywords) --------
@lru_cache(maxsize=1)
def get_legalbert_embedder():
    tok = AutoTokenizer.from_pretrained(LEGAL_BERT)
    model = AutoModel.from_pretrained(LEGAL_BERT)
    model.eval()
    return tok, model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_with_legalbert(texts):
    """Optional: Legal-BERT embeddings (for ranking summaries)."""
    tok, model = get_legalbert_embedder()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**enc)
        pooled = mean_pooling(outputs, enc["attention_mask"])
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

# -------- Summarization Models --------
@lru_cache(maxsize=1)
def get_legal_pegasus():
    return pipeline("summarization", model=LEGAL_PEGASUS, tokenizer=LEGAL_PEGASUS, device=DEVICE)

@lru_cache(maxsize=1)
def get_lexlm_bart():
    tok = AutoTokenizer.from_pretrained(LEXLM_HYBRID)
    model = AutoModelForSeq2SeqLM.from_pretrained(LEXLM_HYBRID)
    return tok, model

def run_seq2seq(tok, model, text, max_new_tokens=256):
    inputs = tok(text, truncation=True, max_length=2048, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=0.9,
        early_stopping=True
    )
    return tok.batch_decode(outputs, skip_special_tokens=True)[0]

@lru_cache(maxsize=1)
def get_mbart50():
    tok = AutoTokenizer.from_pretrained(MBART50)
    model = AutoModelForSeq2SeqLM.from_pretrained(MBART50)
    return tok, model

def mbart_lang_code(lang):
    mapping = {
        "en": "en_XX", "de": "de_DE", "fr": "fr_XX", "es": "es_XX", "it": "it_IT",
        "nl": "nl_XX", "pl": "pl_PL", "pt": "pt_XX", "ro": "ro_RO", "cs": "cs_CZ",
        "hu": "hu_HU", "bg": "bg_BG", "hr": "hr_HR", "sk": "sk_SK", "sl": "sl_SI",
        "et": "et_EE", "lv": "lv_LV", "lt": "lt_LT", "mt": "mt_MT", "sv": "sv_SE",
        "da": "da_DK", "fi": "fi_FI", "el": "el_GR"
    }
    return mapping.get(lang, "en_XX")
