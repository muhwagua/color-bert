from model import connect_TPU, TrainTpu,get_model_and_optimizer,create_distributed_dataset
from configuration import args


TPU, Strategy, GlobalBatchSize = connect_TPU(args.batch_size_TPU)

# X : train_text, y : train_label
dist_dataset = create_distributed_dataset(Strategy,X,y,GlobalBatchSize,training=True)

Model , Optimizer = get_model_and_optimizer(Strategy,args.vanilla_bert,args.learning_rate)
train = TrainTpu(dist_dataset,Model,Optimizer,args,TPU,Strategy,GlobalBatchSize) # train_dist_dataset : dataset, model: bert-based-uncased
train()