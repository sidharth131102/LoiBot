from sentence_transformers import SentenceTransformer, util

class ConceptDriftDetector:
    def __init__(self, threshold=0.7):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = threshold

    def detect(self, text, contexts):
        text_emb = self.embedder.encode(text, convert_to_tensor=True)
        sims = [util.cos_sim(text_emb, self.embedder.encode(c, convert_to_tensor=True)).item() for c in contexts]

        max_sim = max(sims) if sims else 0
        drift = max_sim < self.threshold
        return {"drift": drift, "similarity": max_sim, "new_concepts": [] if not drift else ["Potential new terms"]}
