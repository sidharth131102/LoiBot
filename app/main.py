from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.api.routes import router
from app.models.loaders import (
    get_legal_pegasus,
    get_lexlm_bart,
    get_mbart50,
    get_legalbert_embedder
)

# 🔄 Lifespan context (replaces on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Preloading models into memory...")
    get_legal_pegasus()
    get_lexlm_bart()
    get_mbart50()
    get_legalbert_embedder()
    print("✅ Models loaded successfully!")
    yield   # App runs here
    print("🛑 Shutting down...")

app = FastAPI(
    title="LoiBot - Legal NLP Pipeline",
    version="0.1.0",
    lifespan=lifespan
)

# Allow frontend to call backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backend routes
app.include_router(router, prefix="")

# Mount static folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve frontend HTML files
@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/demo")
async def serve_demo():
    return FileResponse("frontend/demo.html")

@app.get("/contact")
async def serve_contact():
    return FileResponse("frontend/contact.html")

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LoiBot"}
