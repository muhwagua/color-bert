import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len',default=256)
parser.add_argument('--batch_size_TPU',default=32)
parser.add_argument('--batch_size',default=32)
parser.add_argument('--epochs',default=10)
parser.add_argument('--learning_rate',default=1e-5)
parser.add_argument('--evaluate_every',default=5)
parser.add_argument('--vanilla_bert',default='bert-base-uncased')

parser.add_argument('--color_mask_ratio',default= 0.8)

args=parser.parse_args()

