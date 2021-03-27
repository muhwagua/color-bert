import os

import pandas as pd


def get_all_sentences():
    sentences = []
    for file in os.listdir():
        if os.path.isdir(file):
            continue
        file_name, extension = os.path.splitext(file)
        if extension == ".txt":
            with open(file) as f:
                sentences.extend(f.read().splitlines())
        elif extension == ".csv":
            df = pd.read_csv(file)
            sentences.extend(df.sentence.tolist())
    return sentences


def main():
    sentences = get_all_sentences()
    print(f"Total {len(sentences)} sentences")
    string_sentences = "\n".join(sentences)
    with open("all.txt", "w+") as f:
        f.write(string_sentences)


if __name__ == "__main__":
    main()
