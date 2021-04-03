import os
import random
import re

from torch.utils.data import DataLoader, Dataset


class ColorDataset(Dataset):
    def __init__(self, data_path="all.txt", color_ratio=0.5):
        with open(data_path) as f:
            self.data = f.read().splitlines()
        self.color_ratio = color_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        coin = random.random()
        if coin > self.color_ratio:
            masked = mask(sentence)
        else:
            masked = color_mask(sentence)
        return masked, sentence


def color_mask(sentence):
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
    matches = []
    for color in colors:
        match = re.search(f"(\s|^){color}(\s|[.!?\\-]|$)", sentence)
        if match:
            matches.append(match.span())
    (start, end) = random.choice(matches)
    # offset by 1, match includes whitespace to left & right
    return sentence[: start + 1] + "[MASK]" + sentence[end - 1 :]


def mask(sentence):
    words = sentence.split()
    mask_idx = random.choice(range(len(words)))
    words[mask_idx] = "[MASK]"
    return " ".join(words)
