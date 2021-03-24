from model import connect_TPU, TrainTpu , get_model_and_optimizer
from configuration import args


TPU, Strategy, GlobalBatchSize = connect_TPU(args.batch_size_TPU)
Model , Optimizer = get_model_and_optimizer(Strategy,args.vanilla_bert,args.learning_rate)
train = TrainTpu(train_dist_dataset,Model,Optimizer,args,TPU,Strategy,GlobalBatchSize) # train_dist_dataset : dataset, model: bert-based-uncased
train()