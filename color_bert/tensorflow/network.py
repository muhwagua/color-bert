from configuration import args
from transformers import BertTokenizer

from model import (
    TrainTpu,
    connect_TPU,
    create_distributed_dataset,
    get_model_and_optimizer,
)

txt_url = "https://raw.githubusercontent.com/muhwagua/color-bert/main/data/all.txt"
tokenizer = BertTokenizer.from_pretrained(args.vanilla_bert)

Masker = MaskedLMDataset(txt_url, args, tokenizer)
data = Masker()  # {'non_masked': , 'masked'}
TPU, Strategy, GlobalBatchSize = connect_TPU(args.batch_size_TPU)
# X : train_text, y : train_label
dist_dataset = create_distributed_dataset(
    Strategy, data["masked"], data["non_masked"], GlobalBatchSize
)
Model, Optimizer = get_model_and_optimizer(
    Strategy, args.vanilla_bert, args.learning_rate
)
train = TrainTpu(
    dist_dataset, Model, Optimizer, args, TPU, Strategy, GlobalBatchSize
)  # train_dist_dataset : dataset, model: bert-based-uncased
train()
