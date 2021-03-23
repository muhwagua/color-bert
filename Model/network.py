from model import get_config, connect_TPU

config = get_config('config.json')
TPU, Strategy, GlobalBatchSize = connect_TPU(config['BatchSizeTpu'])
train = TrainTpu(train_dist_dataset,model,config,TPU,Strategy,GlobalBatchSize) # train_dist_dataset : dataset, model: bert-based-uncased
train()