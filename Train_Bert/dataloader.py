import random
import re
import urllib.request
from config import args


class MaskedLMDataset():
    def __init__(self, url, config, tokenizer):
        self.url = url
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.color_ratio = config.color_mask_ratio


    def __call__(self):
        lines = self.load_lines()
        non_masked = self.encode_lines(lines)

        masked_lines = self.all_mask(lines,self.color_ratio)
        masked = self.encode_lines(masked_lines)

        return {'non_masked':non_masked,'masked':masked}

    def load_lines(self):
        file = urllib.request.urlopen(self.url)
        lines = [sent.decode('utf-8').strip() for sent in file]
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
        return line[: start + 1] + "[MASK]" + line[end - 1:]


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


    def encode_lines(self, lines):
        batch_encoding = self.tokenizer.batch_encode_plus(lines,return_attention_mask=False,return_token_type_ids=False,padding=True, truncation=True,max_length=self.max_len)
        return batch_encoding["input_ids"]

#from transformers import BertTokenizer
#txt_url = "https://raw.githubusercontent.com/muhwagua/color-bert/main/data/all.txt"
#tokenizer = BertTokenizer.from_pretrained(args.vanilla_bert)

#Masker = MaskedLMDataset(txt_url,args,tokenizer)
#data = Masker()
#print(data.keys())
#print(data['non_masked'][:10])
#print(data['masked'][:10])