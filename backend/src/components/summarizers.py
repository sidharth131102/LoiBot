from transformers import pipeline

class LegalSummarizers:
    def __init__(self):
        self.pegasus = pipeline("summarization", model="nsi319/legal-pegasus")
        self.lexlm = pipeline("summarization", model="MikaSie/LexLM_Longformer_BART_hybrid_V1")

    def summarize_pegasus(self, text, max_length=150, min_length=40):
        return self.pegasus(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    def summarize_lexlm(self, text, max_length=150, min_length=40):
        return self.lexlm(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
