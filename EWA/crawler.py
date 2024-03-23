import requests
import re

from bs4 import BeautifulSoup

TAGS_TO_SELECT = ['h1', 'h2', 'h3', 'h4', 'p']

def preprocess_text(text):
    text = re.sub("\n+", "\n", text)
    text = re.sub("\t+", " ", text)
    text = re.sub(" +", " ", text)
    return text

def check_to_add(text):
    """decide to add the text chunk to the output. Chunks are being received by each tag. Return True or False."""
    if len(text) < 20: # len limit.
        return False
    elif not re.search(r"[a-zA-Z]", text): # if not letter, filter it.
        return False
    return True

def get_text(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise "An error occured while crawling the URL."
    soup = BeautifulSoup(response.text, features="html.parser")
    text = ""
    for tag in TAGS_TO_SELECT:
        for res in soup.find_all(tag):
            if not check_to_add(res.text): # pass if text does not meet the requirements to be added.
                continue
            text += "\n" + res.text.strip()

    processed_text = preprocess_text(text)
    return processed_text.strip()



