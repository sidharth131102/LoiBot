from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline import HybridPipeline
from src.config import Config

app = FastAPI(title="Legal NLP API")

config = Config()
pipeline = HybridPipeline(config)

class Request(BaseModel):
    text: str
    task: str
    lang: str = "en"
    src_lang: str = "en_XX"
    tgt_lang: str = "fr_XX"

@app.post("/process/")
def process(req: Request):
    if req.task == "summarize":
        return {"summary": pipeline.summarize(req.text, lang=req.lang)}
    elif req.task == "translate":
        return {"translation": pipeline.translate(req.text, src_lang=req.src_lang, tgt_lang=req.tgt_lang)}
    elif req.task == "drift":
        return pipeline.drift(req.text)
    elif req.task == "explain":
        return str(pipeline.explain(req.text))  # simplified for now
    else:
        return {"error": "Invalid task"}
