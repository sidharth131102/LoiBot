import os

class Config:
    def __init__(self):
        self.device = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"

        # Models
        self.models = {
            "legalbert": "nlpaueb/legal-bert-base-uncased",
            "pegasus": "nsi319/legal-pegasus",
            "lexlm": "MikaSie/LexLM_Longformer_BART_hybrid_V1",
            "mbart": "facebook/mbart-large-50-many-to-many-mmt"
        }

        # Languages
        self.languages = ["en", "fr", "de", "es", "it"]
        self.drift_threshold = 0.7

        # Pinecone
        self.pinecone_index = "legal-references"
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
