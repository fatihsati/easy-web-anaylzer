import gradio as gr

from EWA.models import Analyzer

def extract_entity(url, ner_model, kw_model):
    print(url, ner_model, kw_model)
    analyzer = Analyzer(url=url, ner_model=ner_model, kw_model=kw_model)
    analyzer.setup()

    results = analyzer.extract_entities()

    ner_results = "\n".join([str({'word': each['word'], 'entity': each['entity_group']}) for each in results['ner']])

    kw_results = "\n".join([str(each) for each in results['kw']])
    
    return kw_results, ner_results


ui = gr.Interface(
    fn=extract_entity,
    inputs=[
        gr.components.Text(None, label='URL', placeholder='Give a single URL.'),
        gr.components.Text(None, label='NER model name', placeholder='This will be selected based on the content language, if not given'),
        gr.components.Text(None, label='Keyword model name', placeholder='This will be selected based on the content language, if not given.'),
    ],
    outputs=[
        gr.components.Text(None, label='Keyword Output'),
        gr.components.Text(None, label='NER Output')
    ],
    description="For a given URL, PERSON, ORGANIZATION, LOCATION and KEYWORDS will be extracted. Language of the content will be detected and the default models will be used accordingly to the detected language. If a model name is given, then this model will be used. Input for the models should be the name of the model from Huggingface Hub.",
    title="Easy-Web-Analyzer",
)

ui.launch()