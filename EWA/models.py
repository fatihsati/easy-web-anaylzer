import fasttext

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

CUSTOM_LANGAUGES = ['tr', 'en']

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
        return pipeline('ner', model=model, tokenizer=tokenizer)
    
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
    def __init__(self):
        self.model = self._load_model(self.MODEL_PATH)
    
    def _load_model(self, model_path):
        return fasttext.load_model(model_path)

    def predict(self, text):
        result, score = self.model.predict(text)
        return result[0], score[0]

def detect_language(language_model, text):
    res, _ = language_model.predict(text)
    language = res.split("__label__")[1]
    if language not in CUSTOM_LANGAUGES:
        return 'other'
    return language

def load_models(language):
    ner_model_name = LANGUAGE_MODEL_MAPPING[language]['ner']
    ke_model_name = LANGUAGE_MODEL_MAPPING[language]['ke']
    print("NER model: ", ner_model_name)
    print("Keyword Extraction model: ", ke_model_name)

    ner = NER(ner_model_name)
    keyword = KeywordExtraction(ke_model_name)

    return ner, keyword




if __name__ == "__main__":
    lng_model = LanguageDetection()

    text = "testing NER model with a new user, Fatih Sati who works in a company called VeriUs Technology"
    lang = detect_language(lng_model, text)
    print("Detected language is: ", lang)

    ner, keyword = load_models(lang)

    results = {'ner': ner.predict(text)}
    print(results)
