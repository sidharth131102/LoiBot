from src.config import Config
from src.components.keyword_extractor import LegalBERTKeywordExtractor
from src.components.rag_retriever import PineconeRetriever
from src.components.summarizers import LegalSummarizers
from src.components.mbart import MultilingualMBart
from src.components.ranker import SummaryRanker
from src.components.drift import ConceptDriftDetector
from src.components.xai_shap import XAIExplainer

class HybridPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.keyword_extractor = LegalBERTKeywordExtractor()
        self.retriever = PineconeRetriever(config.pinecone_index, config.pinecone_api_key)
        self.summarizers = LegalSummarizers()
        self.mbart = MultilingualMBart()
        self.ranker = SummaryRanker()
        self.drift_detector = ConceptDriftDetector(config.drift_threshold)
        self.explainer = XAIExplainer()

    def summarize(self, text, lang="en"):
        if lang == "en":
            keywords = self.keyword_extractor.extract_keywords(text)
            contexts, matches = self.retriever.retrieve(" ".join(keywords))
            pegasus_sum = self.summarizers.summarize_pegasus(text + " " + " ".join(contexts))
            lexlm_sum = self.summarizers.summarize_lexlm(text + " " + " ".join(contexts))
            best_summary, _ = self.ranker.rank(text, [pegasus_sum, lexlm_sum], contexts)
            return best_summary
        else:
            translated = self.mbart.translate(text, src_lang=f"{lang}_XX", tgt_lang="en_XX")
            return self.summarize(translated, lang="en")

    def translate(self, text, src_lang="en_XX", tgt_lang="fr_XX"):
        return self.mbart.translate(text, src_lang=src_lang, tgt_lang=tgt_lang)

    def drift(self, text):
        contexts, _ = self.retriever.retrieve(text)
        return self.drift_detector.detect(text, contexts)

    def explain(self, text):
        return self.explainer.explain(text)
