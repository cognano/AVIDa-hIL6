import os
import gc
import csv
import pickle
import argparse
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
)
from utils import fix_seed
from utils import get_config
from logger import setup_logging


def main(args, save_dirs, cfg):
    train_data = np.load(args.train_data)
    train_y = train_data["label"]
    train_x = np.concatenate(
        [train_data["ab"].reshape((-1, 3600)), train_data["ag"].reshape((-1, 4400))], axis=1
    )
    del train_data
    gc.collect()
    train_idx, _ = train_test_split(
        list(range(train_x.shape[0])), test_size=0.1, random_state=cfg["seed"], shuffle=True
    )
    train_y = train_y[train_idx]
    train_x = train_x[train_idx]

    model = LogisticRegression(
        max_iter=1000,
        random_state=cfg["seed"],
        verbose=1,
    )
    model.fit(train_x, train_y)
    del train_x, train_y
    gc.collect()
    with open(os.path.join(save_dirs["model"], "model.pkl"), mode="wb") as f:
        pickle.dump(model, f)

    test_data = np.load(args.test_data)
    test_y = test_data["label"]
    test_x = np.concatenate(
        [test_data["ab"].reshape((-1, 3600)), test_data["ag"].reshape((-1, 4400))], axis=1
    )
    preds = model.predict(test_x)
    outputs = model.predict_proba(test_x)[:, 1]
    test_accuracy = accuracy_score(test_y, preds)
    test_precision = precision_score(test_y, preds)
    test_recall = recall_score(test_y, preds)
    test_f1 = f1_score(test_y, preds)
    test_mcc = matthews_corrcoef(test_y, preds)
    precisions, recalls, _ = precision_recall_curve(test_y, outputs)
    test_auroc = roc_auc_score(test_y, outputs)
    test_auprc = auc(recalls, precisions)
    print("Accuracy: {}".format(test_accuracy))
    print("Precision: {}".format(test_precision))
    print("Recall: {}".format(test_recall))
    print("F1: {}".format(test_f1))
    print("AUPRC: {}".format(test_auprc))
    with open(os.path.join(save_dirs["log"], "test_outputs.csv"), mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["outputs", "preds", "targets"])
        for i in range(len(outputs)):
            writer.writerow([outputs[i], preds[i], int(test_y[i])])

    with open(os.path.join(save_dirs["log"], "test_summary.csv"), mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "precision", "recall", "f1", "MCC", "AUROC", "AUPRC"])
        writer.writerow(
            [
                test_accuracy,
                test_precision,
                test_recall,
                test_f1,
                test_mcc,
                test_auroc,
                test_auprc,
            ]
        )


if __name__ == "__main__":
    cfg = get_config()
    fix_seed(cfg["seed"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=None, type=str, help="Train data path")
    parser.add_argument("--test-data", default=None, type=str, help="Test data path")
    parser.add_argument(
        "--save-dir", default="./saved", type=str, help="Save directory path (default: ./saved)"
    )
    parser.add_argument(
        "--run-id",
        default="",
        type=str,
        help="Run ID for saving results (default: '')",
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