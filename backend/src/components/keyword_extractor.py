from transformers import AutoTokenizer, AutoModel
import torch

class LegalBERTKeywordExtractor:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_keywords(self, text, top_k=5):
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze()

        keywords = tokens[:top_k]
        return keywords
