import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.models.loaders import embed_texts   # ✅ mpnet or your embedder

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west-2")  # fallback region
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "loibot-refs")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists (optional, safe check)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,   # ✅ match your SentenceTransformer (mpnet = 768)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Get index handle
index = pc.Index(INDEX_NAME)

def retrieve_context(query: str, top_k: int = 5):
    # Get embeddings
    query_emb = embed_texts([query])[0].tolist()

    # Query Pinecone
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )

    references = []
    ids = []
    for match in results["matches"]:
        ids.append(match["id"])
        references.append({
            "reference": match["metadata"].get("reference", ""),
            "score": match["score"]
        })

    return {"ids": ids, "references": references}
