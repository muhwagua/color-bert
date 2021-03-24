import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len',default=256)
parser.add_argument('--batch_size_TPU',default=16)
parser.add_argument('--batch_size',default=64)
parser.add_argument('--epochs',default=100)
parser.add_argument('--learning_rate',default=1e-10)
parser.add_argument('--evaluate_every',default=5)
parser.add_argument('--vanilla_bert',default='bert_base_uncased')

args=parser.parse_args()

