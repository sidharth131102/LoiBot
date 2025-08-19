from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class MultilingualMBart:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt"):
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    def translate(self, text, src_lang="fr_XX", tgt_lang="en_XX"):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang])
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
