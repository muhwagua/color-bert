import random
import re
import urllib.request

from tensorflow.keras.utils import Sequence
import tensorflow as tf
from transformers import (
    BertConfig,
    BertTokenizer,
)
from argparse import Namespace


txt_url = "https://raw.githubusercontent.com/muhwagua/color-bert/main/data/all.txt"
urllib.request.urlretrieve(txt_url, "train.txt")

args = Namespace()
args.train = "train.txt"
args.max_len = 128
args.model_name = "bert-base-uncased"
args.batch_size = 4
args.color_ratio = 0.5


tokenizer = BertTokenizer.from_pretrained(args.model_name)


class MaskedLMDataset(Sequence):
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
            batch_encoding = self.tokenizer.batch_encode_plus(
                masked,
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            return batch_encoding["input_ids"]

        elif masking == False:
            batch_encoding = self.tokenizer.batch_encode_plus(
                lines,
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.ids[idx]


train_dataset = MaskedLMDataset(args.train, args.color_ratio, tokenizer, masking=True)
label_dataset = MaskedLMDataset(args.train, args.color_ratio, tokenizer, masking=False)


class Dataloader(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tpu, self.strategy, self.global_batch_size = self.connect_TPU(
            self.batch_size
        )
        self.dist_dataset = self.distributed_dataset(
            self.x, self.y, self.global_batch_size, shuffle
        )

    def connect_TPU(self, batch_size):
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        global_batch_size = batch_size * strategy.num_replicas_in_sync

        return tpu, strategy, global_batch_size

    def distributed_dataset(self, x, y, global_batch_size, shuffle):
        dataset_x = tf.data.Dataset.from_tensor_slices(self.x)
        dataset_y = tf.data.Dataset.from_tensor_slices(self.y)
        dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
        AUTO = tf.data.experimental.AUTOTUNE
        if shuffle == True:
            dataset = dataset.shuffle(len(self.x)).repeat()

        dataset = dataset.batch(global_batch_size).prefetch(AUTO)
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

        return dist_dataset


prepareTPU = Dataloader(train_dataset, label_dataset, batch_size=16, shuffle=True)
