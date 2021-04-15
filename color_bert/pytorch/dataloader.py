import random
import re
import urllib.request
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertTokenizer

txt_url = "https://raw.githubusercontent.com/muhwagua/color-bert/main/data/all.txt"
urllib.request.urlretrieve(txt_url, "train.txt")

args = Namespace()
args.train = "train.txt"
args.max_len = 128
args.model_name = "bert-base-uncased"
args.batch_size = 4
args.color_ratio = 0.5


tokenizer = BertTokenizer.from_pretrained(args.model_name)


class MaskedLMDataset(Dataset):
    def __init__(self, file, color_ratio, tokenizer, masking):
        self.tokenizer = tokenizer
        self.color_ratio = color_ratio
        self.masking = masking
        self.lines = self.load_lines(file)
        self.masked = self.all_mask(self.lines, self.color_ratio)
        self.ids = self.encode_lines(self.lines, self.masked, masking)

    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def color_mask(self, line, masking=True):
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
        for color in colors:
            match = re.search(f"(\s|^){color}(\s|[.!?\\-])", line)
            if match:
                global start, end
                (start, end) = random.choice([match.span()])
        return line[: start + 1] + "[MASK]" + line[end - 1 :]

    def random_mask(self, line, masking=True):
        words = line.split()
        mask_idx = random.choice(range(len(words)))
        words[mask_idx] = "[MASK]"
        return " ".join(words)

    def all_mask(self, lines, color_ratio, masking=True):
        masked = []
        for line in lines:
            coin = random.random()
            if coin > color_ratio:
                masked.append(self.random_mask(line))
            else:
                masked.append(self.color_mask(line))

        return masked

    def encode_lines(self, lines, masked, masking):
        if masking == True:
            batch_encoding = self.tokenizer(
                masked,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=args.max_len,
            )
            return batch_encoding["input_ids"]

        elif masking == False:
            batch_encoding = self.tokenizer(
                lines,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=args.max_len,
            )
            return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)
