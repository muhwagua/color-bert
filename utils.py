import json
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.cuda.amp import autocast


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AttrDict(Namespace):
    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            value = AttrDict(value) if isinstance(value, dict) else value
            setattr(self, key, value)

    def __setattr__(self, key, value):
        value = AttrDict(value) if isinstance(value, dict) else value
        super().__setattr__(key, value)

def load_config(file_path: str) -> AttrDict:
    with open(file_path) as f:
        return AttrDict(json.load(f))

def load_trainer(checkpoint_path):
    # avoid circular import
    from trainer import Trainer

    root, _ = os.path.split(checkpoint_path)
    config_path = os.path.join(root, "config.json")
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    return trainer