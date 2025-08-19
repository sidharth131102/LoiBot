from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

class PineconeRetriever:
    def __init__(self, index_name, api_key, embedder="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder)
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def retrieve(self, query, top_k=5):
        emb = self.embedder.encode(query).tolist()
        results = self.index.query(vector=emb, top_k=top_k, include_metadata=True)
        contexts = [m["metadata"]["reference"] for m in results["matches"]]
        return contexts, results["matches"]
