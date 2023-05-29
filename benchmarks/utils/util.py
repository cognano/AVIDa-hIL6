import os
import json
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict


def fix_seed(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json(filename):
    filename = Path(filename)
    with filename.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def get_config(config_file="./benchmarks/config.yaml"):
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
