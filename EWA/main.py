import argparse
import json

from EWA.models import Analyzer


def get_arguments():

    parser = argparse.ArgumentParser(
        prog='easy-web-ananlyzer',
        description="Extracts PER-LOC-ORG entities and list of Keywords from URL"
    )

    parser.add_argument("--url", required=True, help="URL to extract entities and keywords")
    parser.add_argument("--ner-model", required=False, help="hf_hub model to be used as NER model.")
    parser.add_argument("--kw-model", required=False, help="hf_hub embedding model to be used for keyword extraction.")
    parser.add_argument("--save-output", help="Bool value, true for saving output as json.")

    return parser.parse_args()


def save_json(filename, data):

    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    
    args = get_arguments()

    analyzer = Analyzer(**args.__dict__)

    analyzer.setup()
    results = analyzer.extract_entities()

    print(results)

    if args.save_output:
        save_json('output.json', results)
    

if __name__ == "__main__":
    main()
