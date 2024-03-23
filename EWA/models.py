import fasttext

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

CUSTOM_LANGAUGES = ['tr', 'en']

class NER:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _load_model(self, model_name):
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline('ner', model=model, tokenizer=tokenizer)
    
    def predict(self, text):
        return self.model(text)

class KeywordExtraction:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name):
        return True




lng_model = fasttext.load_model(("./EWA/lid.176.ftz"))

def detect_language(text):
    res = lng_model.predict(text)[0][0]
    language = res.split("__label__")[1]
    if language not in CUSTOM_LANGAUGES:
        return 'other'
    return language

def load_models(language):
    ner_model_name = LANGUAGE_MODEL_MAPPING[language]['ner']
    ke_model_name = LANGUAGE_MODEL_MAPPING[language]['ke']

    ner = NER(ner_model_name)
    keyword = KeywordExtraction(ke_model_name)

    return ner, keyword


if __name__ == "__main__":
    text = input("Input: ")

    lang = detect_language(text)
    print("Detected language is: ", lang)

    ner, keyword = load_models(lang)

    results = {'ner': ner.predict(text)}
    print(results)
