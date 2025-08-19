import shap
from transformers import pipeline

class XAIExplainer:
    def __init__(self, model="nsi319/legal-pegasus"):
        self.pipe = pipeline("summarization", model=model)
        self.explainer = shap.Explainer(self.pipe)

    def explain(self, text):
        shap_values = self.explainer([text])
        return shap_values
