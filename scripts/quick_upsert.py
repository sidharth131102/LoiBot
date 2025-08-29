import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Load Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("loibot-refs")

# Embedding model
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Path to your dataset
DATA_PATH = "data/train.json"

# Chunking function
def chunk_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Load data
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

vectors = []
for i, item in enumerate(data):
    ref_text = item.get("reference", "")
    if not ref_text.strip():
        continue

    # Chunk the reference text
    for j, chunk in enumerate(chunk_text(ref_text, max_words=500)):
        emb = embedder.encode(chunk).tolist()
        vectors.append({
            "id": f"train-{i}-{j}",
            "values": emb,
            "metadata": {
                "celex_id": item.get("celex id", ""),
                "summary": item.get("summary", ""),
                "reference_chunk": chunk  # small chunk only
            }
        })

print(f"Prepared {len(vectors)} vectors after chunking.")

# Upsert in batches
BATCH_SIZE = 50  # keep batches moderate
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    print(f"Upserting batch {i//BATCH_SIZE+1} with {len(batch)} vectors...")
    index.upsert(vectors=batch)

print(f"✅ Successfully upserted {len(vectors)} vectors into Pinecone.")
