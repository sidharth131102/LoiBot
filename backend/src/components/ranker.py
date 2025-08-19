from sentence_transformers import SentenceTransformer, util

class SummaryRanker:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def rank(self, source_text, summaries, contexts):
        scores = []
        src_emb = self.embedder.encode(source_text, convert_to_tensor=True)

        for summ in summaries:
            summ_emb = self.embedder.encode(summ, convert_to_tensor=True)
            sim_src = util.cos_sim(src_emb, summ_emb).item()

            sim_ctx = 0.0
            for ctx in contexts:
                ctx_emb = self.embedder.encode(ctx, convert_to_tensor=True)
                sim_ctx = max(sim_ctx, util.cos_sim(ctx_emb, summ_emb).item())

            score = 0.7 * sim_src + 0.3 * sim_ctx
            scores.append(score)

        best_idx = scores.index(max(scores))
        return summaries[best_idx], scores
