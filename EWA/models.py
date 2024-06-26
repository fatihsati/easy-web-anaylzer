import fasttext

from EWA import crawler
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers import logging
from keybert import KeyBERT

logging.set_verbosity_error()


LANGUAGE_MODEL_MAPPING = {
    "tr": {
        "ner": "savasy/bert-base-turkish-ner-cased",
        "kw": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    },
    "en": {
        "ner": "dslim/bert-base-NER",
        "kw": "sentence-transformers/paraphrase-mpnet-base-v2"
    },
    "other": {
        "ner": "Babelscape/wikineural-multilingual-ner", # update with multilanguage model.
        "kw": "paraphrase-multilingual-MiniLM-L12-v2"
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
        output = self.model(text, aggregation_strategy="simple")
        for entity in output:
            entity['score'] = float(entity['score'])
        return output


class KeywordExtraction:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name):
        kw_model = KeyBERT(model_name)
        print(f"KeywordExtraction model is loaded: {self.model_name}")
        return kw_model
    
    def predict(self, text, language='en'):

        output = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=10, use_maxsum=True)
        # convert float32 to python float
        output = [
            {
                'keyword': word, 
                'score': float(score)
            } for word, score in output]
        
        return output
    

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

    def __init__(self, url, ner_model, kw_model, **kwargs) -> None:
        self.url = url
        self.ner_model_name = ner_model
        self.kw_model_name = kw_model
        
        self.language = None # to be extracted later.
        self.language_detector = LanguageDetection()
        
        self.content = crawler.get_text(self.url)


    def setup(self):
        """Loads models:
            1. ner
            2. kw
        """
        self.ner = self.load_ner()
        self.kw = self.load_kw()


    def load_ner(self):
        """Load the required models and values according to the given input."""
        if self.ner_model_name:
            return NER(self.ner_model_name)
        
        # if ner model name is not given:
        if not self.language:
            self.language = self.language_detector.predict(self.content)

        # get default model name for language:
        self.ner_model_name = LANGUAGE_MODEL_MAPPING[self.language]['ner']
        return NER(self.ner_model_name)
    

    def load_kw(self):
        """Load the required models and values according to the given input."""

        if self.kw_model_name:
            return KeywordExtraction(self.kw_model_name)
        
        if not self.language:
            self.language = self.language_detector.predict(self.content)

        self.kw_model_name = LANGUAGE_MODEL_MAPPING[self.language]['kw']
        return KeywordExtraction(self.kw_model_name)


    def extract_entities(self):
        
        results = {
            'ner': self.ner.predict(self.content),
            'kw': self.kw.predict(self.content, self.language)
        }        
        
        return results


if __name__ == "__main__":
    test_content = """Geçtiğimiz günlerde sosyal medyada bir mekânın işletmecisinin paylaştığı video gündem oldu. İftar için rezervasyon yaptıran 15 kişilik bir grubun haber vermeden mekâna gitmediğini anlatan işletmeci, “Müşteriler arıyor, masalar dolu yer yok diyoruz. Ancak 15 kişi yer ayırtmış, arıyoruz telefonlarını da açmıyorlar. Yemekleri pişti, masalarında salatalar, çorbalar var, hepsi ziyan oldu. Yarım saat önceden bile arayıp ‘gelemiyoruz’ deseler yine sorun olmayacak. Arayıp bilgi verseler masayı başka müşteriye devredebiliriz. Masayı da geçtim, boş kalsın. Peki  hazır pişmiş yemeklere yazık değil mi?” ifadelerine yer verdi.
    Rezervasyonu iptal etmeden mekâna gitmeyen müşteriler sadece ülkemizde değil, dünya genelinde işletmelerin çok sık karşılaştığı bir sorun. ABD’de bazı işletmeler bu sorunun önüne geçebilmek için ‘rezervasyon ücreti’ alıyor. Kişi başı 100 dolar olan rezervasyon ücreti, son dakikaya kadar bilgi vermeden restorana gitmeyen müşterileri caydırmak, işletmeleri korumak için uygulanıyor."""
    
    ner = NER(LANGUAGE_MODEL_MAPPING['tr']['ner'])
    kw = KeywordExtraction(LANGUAGE_MODEL_MAPPING['tr']['kw'])
    
    results = {
        'ner': ner.predict(test_content),
        'kw': kw.predict(test_content)
    }

    print(results)
