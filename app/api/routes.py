from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from app.core.summarizer import summarize_pipeline
from app.core.translator import translate_pipeline
from app.core.drift import drift_pipeline

router = APIRouter()

class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1)
    task: str = Field(..., pattern="^(summarization|translation|both|drift)$")
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = "en"
    top_k: Optional[int] = 4

# ✅ Structured explanation schema
class Explanation(BaseModel):
    words: List[str]
    scores: List[float]

class ProcessResponse(BaseModel):
    summary: Optional[str] = None
    translation: Optional[str] = None
    explanation: Optional[Explanation] = None
    drift_percentage: Optional[float] = None
    new_concepts: Optional[List[str]] = None
    candidates: Optional[Dict[str, str]] = None
    retrieved_ids: Optional[List[str]] = None

@router.post("/process/", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if req.task == "translation":
        try:
            out = translate_pipeline(
                req.text, src_lang=req.src_lang or "auto", tgt_lang=req.tgt_lang or "en"
            )
        except Exception as e:
            return ProcessResponse(translation=f"Error: {str(e)}")
        return ProcessResponse(translation=out)

    if req.task == "summarization":
        try:
            result = summarize_pipeline(req.text) or {}
        except Exception as e:
            return ProcessResponse(summary=f"Error: {str(e)}")
        return ProcessResponse(
            summary=result.get("summary"),
            explanation=result.get("explanation"),
            candidates=result.get("candidates"),
            retrieved_ids=result.get("retrieved_ids"),
        )

    if req.task == "both":
        try:
            result = summarize_pipeline(req.text) or {}
            trans = translate_pipeline(
                result.get("summary", ""), src_lang="auto", tgt_lang=req.tgt_lang or "en"
            )
        except Exception as e:
            return ProcessResponse(summary="Error", translation=f"Error: {str(e)}")
        return ProcessResponse(
            summary=result.get("summary"),
            translation=trans,
            explanation=result.get("explanation"),
            candidates=result.get("candidates"),
            retrieved_ids=result.get("retrieved_ids"),
        )

    if req.task == "drift":
        try:
            drift = drift_pipeline(req.text, top_k=req.top_k or 4) or {}
        except Exception as e:
            return ProcessResponse(drift_percentage=0.0, new_concepts=[f"Error: {str(e)}"])
        return ProcessResponse(
            drift_percentage=drift.get("drift_percentage"),
            new_concepts=drift.get("new_concepts"),
        )
