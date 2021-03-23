import json
from collections import OrderedDict

config = OrderedDict()

config['MaxLength'] = 256
config['BatchSizeTpu'] = 16  # per TPU core
config['BatchSize'] = 64
config['Epochs'] = 100
config['LearningRate'] = 1e-5
config['EvaluateEvery'] = 5


with open('config.json','w',encoding='utf-8') as fp:
    json.dump(config, fp, ensure_ascii=False, indent='\t')

