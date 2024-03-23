import fasttext

from EWA import crawler
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


LANGUAGE_MODEL_MAPPING = {
    "tr": {
        "ner": "savasy/bert-base-turkish-ner-cased",
        "ke": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    },
    "en": {
        "ner": "dslim/bert-base-NER",
        "ke": "sentence-transformers/paraphrase-mpnet-base-v2"
    },
    "other": {
        "ner": "", # update with multilanguage model.
        "ke": ""
    },
}


class NER:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name):
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
        print(f"NER model is loaded: {self.model_name}")
        return ner_model
    
    def predict(self, text):
        return self.model(text, aggregation_strategy="simple")


class KeywordExtraction:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name):
        return True

class LanguageDetection:

    MODEL_PATH = "./EWA/lid.176.ftz"
    CUSTOM_LANGAUGES = ['tr', 'en']
    def __init__(self):
        self.model = self._load_model(self.MODEL_PATH)
    
    def _load_model(self, model_path):
        return fasttext.load_model(model_path)

    def predict(self, text):
        text = text.replace("\n", " ")
        result, _ = self.model.predict(text)
        language = result[0].split("__label__")[1]
        if language not in self.CUSTOM_LANGAUGES:
            return 'other'
        
        print(f"predicted language is: '{language}'")
        return language

class Analyzer:

    def __init__(self, url, ner_model, kw_model) -> None:
        self.url = url
        self.ner_model_name = ner_model
        self.kw_model_name = kw_model
        
        self.language = None # to be extracted later.
        self.language_detector = LanguageDetection()
        
        self.content = crawler.get_text(self.url)


    def setup_ner(self):
        """Load the required models and values according to the given input."""
        if self.ner_model_name:
            self.ner = NER(self.ner_model_name)
            return True
        
        # if ner model name is not given:
        if not self.language:
            self.language = self.language_detector.predict(self.content)

        # get default model name for language:
        self.ner_model_name = LANGUAGE_MODEL_MAPPING[self.language]['ner']
        self.ner = NER(self.ner_model_name)

        return True
    

    def setup_kw(self):
        """Load the required models and values according to the given input."""

        if self.kw_model_name:
            self.kw = KeywordExtraction(self.kw_model_name)
            return True
        
        if not self.language:
            self.language = self.language_detector.predict(self.content)

        self.kw_model_name = LANGUAGE_MODEL_MAPPING[self.language]['kw']
        self.kw = KeywordExtraction(self.kw_model_name)
        return True



if __name__ == "__main__":
    url = 'https://www.ntv.com.tr/galeri/dunya/moskovada-katliam-taniklarin-gozunden-adim-adim-yasananlar,Fyt-uIW2PEu0DWy-Y76KDg'
    analyzer= Analyzer(url=url, ner_model=None, kw_model=None)
    analyzer.setup_ner()

    result = analyzer.ner.predict(analyzer.content)
    print(result)

