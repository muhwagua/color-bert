import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect

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
        sentences = preprocess(tag.text)
        # list comprehension is faster than for loop
        [
            color_sentences.append(sentence)
            for sentence in sentences
            if has_color(sentence)
        ]
    return color_sentences


def preprocess(text):
    sentences = []
    text = text.replace("\r\n", " ")
    text = re.sub("[\[\d\]]", "", text)
    chunks = re.split("(?<!Mr|Ms|Dr)(?<!Mrs)\.", text)
    # list comprehension is faster than for loop
    [sentences.extend(re.split("(?<=[!?;:]) +", chunk)) for chunk in chunks]
    sentences = [" ".join(sentence.split()) for sentence in sentences]
    return sentences


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


def english_only(df):
    df["is_en"] = df[0].apply(detect)
    df = df[df["is_en"] == "en"]
    df_en = df.drop("is_en", axis=1)
    return df_en


def main():
    all_sentences = []
    urls = get_book_urls()
    for url in urls:
        color_sentences = get_color_sentences(url)
        all_sentences.extend(color_sentences)
    df = pd.DataFrame(all_sentences)
    df_en = english_only(df)
    df_en.to_csv(os.path.join(os.path.dirname(__file__), "raw", "gutenberg.csv"))


if __name__ == "__main__":
    main()
