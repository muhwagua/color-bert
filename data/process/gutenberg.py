import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "white",
    "black",
    "pink",
    "lime",
    "gray",
    "violet",
    "cyan",
    "magenta",
    "khaki",
]


def has_color(sentence):
    for color in colors:
        if re.search(f"(\s|^){color}(\s|[.!?\\-])", sentence):
            return True
    return False


def get_color_sentences(url):
    color_sentences = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    p_tags = soup.find_all("p")
    for tag in p_tags:
        chunk = tag.text.strip().replace("\r\n", " ")
        sentences = chunk.split(".")
        for sentence in sentences:
            if has_color(sentence):
                sentence = " ".join(sentence.split())
                color_sentences.append(sentence + ".")
    return color_sentences


def get_book_urls():
    pass


def main():
    all_sentences = []
    urls = get_book_urls
    for url in urls:
        color_sentences = get_color_sentences(url)
        all_sentences.extend(color_sentences)
    df = pd.DataFrame(all_sentences)
    pd.save_csv(os.path.join(os.path.dirname(__file__), "raw", "gutenberg.csv"))


if __name__ == "__main__":
    main()
