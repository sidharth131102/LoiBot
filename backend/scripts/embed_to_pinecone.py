import os, json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-references"

pc = Pinecone(api_key=API_KEY)

if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("backend/data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vectors = []
for i, item in enumerate(tqdm(data)):
    ref = item.get("reference", "").strip()
    summ = item.get("summary", "").strip()
    cid = str(item.get("celex id", i))

    if not ref:
        continue

    emb = model.encode(ref).tolist()
    vectors.append({
        "id": cid,
        "values": emb,
        "metadata": {
            "reference": ref,
            "gold_summary": summ
        }
    })

    if len(vectors) >= 100:
        index.upsert(vectors)
        vectors = []

if vectors:
    index.upsert(vectors)

print("✅ Uploaded all references to Pinecone")
