import os
import gc
import argparse
from distutils.util import strtobool
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split

from utils import fix_seed
from utils import get_config
from logger import setup_logging
from trainer import Trainer
from models import AbAgIntPre, PIPR, MLP
from data_loader import AAIdataset


def main(args, save_dirs, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or cfg["batch_size"]
    train_dataset = np.load(args.train_data)
    train_dataset = AAIdataset(
        train_dataset["ab"], train_dataset["ag"], train_dataset["label"]
    )
    train_idx, valid_idx = train_test_split(
        list(range(len(train_dataset))),
        test_size=args.valid_ratio,
        random_state=cfg["seed"],
        shuffle=True,
    )
    valid_dataset = Subset(train_dataset, valid_idx)
    train_dataset = Subset(train_dataset, train_idx)
    test_dataset = np.load(args.test_data)
    test_dataset = AAIdataset(test_dataset["ab"], test_dataset["ag"], test_dataset["label"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    del train_dataset, valid_dataset, test_dataset
    gc.collect()

    if args.model_name == "AbAgIntPre":
        model = AbAgIntPre().to(device)
    elif args.model_name == "PIPR":
        model = PIPR().to(device)
    elif args.model_name == "MLP":
        model = MLP().to(device)
    else:
        raise ValueError("The model-name argument must be one of ['AbAgIntPre', 'PIPR', 'MLP'].")

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_epochs=args.epochs,
        save_dirs=save_dirs,
        amp=strtobool(args.amp),
        model_name=args.model_name,
    )
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    cfg = get_config()
    fix_seed(cfg["seed"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=None, type=str, help="Path of train data ")
    parser.add_argument("--test-data", default=None, type=str, help="Path of test data")
    parser.add_argument(
        "--valid-ratio",
        default=0.1,
        type=float,
        help="Ratio used for validation data from training data",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        type=str,
        required=True,
        help="Model name must be one of ['AbAgIntPre', 'PIPR', 'MLP'].",
    )
    parser.add_argument(
        "--save-dir", default="./saved", type=str, help="Save directory path (default: ./saved)"
    )
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs (default: 20)")
    parser.add_argument(
        "--batch-size", default=256, type=int, help="The size of batch (default in config.yaml)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        help="Model path used for retraining (default: None)",
    )
    parser.add_argument(
        "--amp",
        default="False",
        type=str,
        help="Use Automatic Mixed Precision (default: False)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        type=str,
        help="Run ID used for the directory name for saving the results (default: '')",
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    run_id = args.run_id or datetime.now().strftime(r"%Y%m%d_%H%M%S")
    save_dirs = dict()
    save_dirs["model"] = os.path.join(save_dir, run_id, "models")
    save_dirs["log"] = os.path.join(save_dir, run_id, "logs")
    for dir in save_dirs.values():
        os.makedirs(dir, exist_ok=True)
    setup_logging(save_dirs["log"])

    main(args, save_dirs, cfg)
