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
    soup = BeautifulSoup(response.content, features="html.parser")
    p_tags = soup.find_all("p")
    for tag in p_tags:
        chunk = tag.text.replace("\r\n", " ")
        chunk = re.sub("[\[\d\]]", "", chunk)
        sentences = re.split("(?<=[.!?]) +", chunk)
        for sentence in sentences:
            if has_color(sentence):
                sentence = " ".join(sentence.split())
                color_sentences.append(sentence)
    return color_sentences


def get_book_urls():
    book_urls = []
    root = "https://www.gutenberg.org"
    popular_books_url = "https://www.gutenberg.org/browse/scores/top#books-last1"
    response = requests.get(popular_books_url)
    soup = BeautifulSoup(response.content, features="html.parser")
    book_list = soup.find("ol")
    urls = [f"{root}{a['href']}" for a in book_list.find_all("a")]
    for url in urls:
        book_response = requests.get(url)
        book_soup = BeautifulSoup(book_response.content, features="html.parser")
        try:
            a_tag = book_soup.find("a", text="Read this book online: HTML")
            book_urls.append(f"{root}{a_tag['href']}")
        except TypeError:
            continue
    return book_urls


def main():
    all_sentences = []
    urls = get_book_urls()
    for url in urls:
        color_sentences = get_color_sentences(url)
        all_sentences.extend(color_sentences)
    df = pd.DataFrame(all_sentences)
    df.to_csv(os.path.join(os.path.dirname(__file__), "raw", "gutenberg.csv"))


if __name__ == "__main__":
    main()
