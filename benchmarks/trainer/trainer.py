import os
import csv
import shutil
import torch
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
from tqdm import tqdm
from logging import getLogger
from models import AbAgIntPre, PIPR, MLP


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        train_loader,
        valid_loader=None,
        test_loader=None,
        n_epochs=None,
        save_dirs=None,
        amp=None,
        model_name=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.save_dirs = save_dirs
        self.logger = getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        if amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.model_name = model_name

    def train(self):
        best_acc = 0
        with open(os.path.join(self.save_dirs["log"], "info.csv"), mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_acc", "train_recall", "valid_acc", "valid_recall"])
        for epoch in range(1, self.n_epochs + 1):
            (
                train_loss,
                train_auroc,
                train_auprc,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
                train_mcc,
            ) = self._train_epoch()
            (
                valid_loss,
                valid_auroc,
                valid_auprc,
                valid_accuracy,
                valid_precision,
                valid_recall,
                valid_f1,
                valid_mcc,
            ) = self._valid_epoch()
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train: loss {train_loss:.4f}, AUROC {train_auroc:.4f}, AUPRC {train_auprc:.4f}, accuracy {train_accuracy:.4f}, "
                f"precision {train_precision:.4f}, recall {train_recall:.4f}, f1 {train_f1:.4f}, MCC {train_mcc:.4f}"
            )
            self.logger.info(
                f"Epoch {epoch}: "
                f"Valid: loss {valid_loss:.4f}, AUROC {valid_auroc:.4f}, AUPRC {valid_auprc:.4f}, accuracy {valid_accuracy:.4f}, "
                f"precision {valid_precision:.4f}, recall {valid_recall:.4f}, f1 {valid_f1:.4f}, MCC {valid_mcc:.4f}"
            )
            with open(os.path.join(self.save_dirs["log"], "info.csv"), mode="a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch, train_accuracy, train_recall, valid_accuracy, valid_recall]
                )

            save_info = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "loss": valid_loss,
                "accuracy": valid_accuracy,
                "racall": valid_recall,
            }
            torch.save(
                save_info,
                os.path.join(self.save_dirs["model"], "model_latest.pt"),
            )
            if valid_accuracy > best_acc:
                torch.save(
                    save_info,
                    os.path.join(self.save_dirs["model"], "model_best.pt"),
                )
                best_acc = valid_accuracy
            if epoch % 100 == 0:
                shutil.copyfile(
                    os.path.join(self.save_dirs["model"], "model_best.pt"),
                    os.path.join(self.save_dirs["model"], "model_best_epoch{}.pt".format(epoch)),
                )

    def test(self):
        if self.model_name == "AbAgIntPre":
            model = AbAgIntPre().to(self.device)
        elif self.model_name == "PIPR":
            model = PIPR().to(self.device)
        elif self.model_name == "MLP":
            model = MLP().to(self.device)
        else:
            raise ValueError(
                "The model-name argument must be one of ['AbAgIntPre', 'PIPR', 'MLP']."
            )
        checkpoint = torch.load(os.path.join(self.save_dirs["model"], "model_best.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        test_loss = 0
        outputs = []
        preds = []
        targets = []
        with torch.no_grad():
            for batch in self.test_loader:
                x_ab, x_ag, target = batch
                targets.extend(target.tolist())
                x_ab, x_ag, target = self._to_device(x_ab, x_ag, target)
                output = model(x_ab, x_ag)
                output = torch.squeeze(output)
                output_sig = torch.sigmoid(output)
                outputs.extend(output_sig.tolist())
                loss = self.criterion(output, target)
                test_loss += loss.item()
                pred = torch.round(output_sig)
                preds.extend(pred.tolist())
        test_loss = test_loss / len(self.test_loader)
        test_auroc = roc_auc_score(targets, outputs)
        test_accuracy = accuracy_score(targets, preds)
        test_precision = precision_score(targets, preds)
        test_recall = recall_score(targets, preds)
        test_f1 = f1_score(targets, preds)
        test_mcc = matthews_corrcoef(targets, preds)
        precisions, recalls, _ = precision_recall_curve(targets, outputs)
        test_auprc = auc(recalls, precisions)
        self.logger.info(
            f"Test: loss {test_loss:.4f}, accuracy {test_accuracy:.4f}, AUROC {test_auroc:.4f}, AUPRC {test_auprc:.4f}, "
            f"precision {test_precision:.4f}, recall {test_recall:.4f}, f1 {test_f1:.4f}, MCC {test_mcc:.4f}"
        )
        with open(os.path.join(self.save_dirs["log"], "test_summary.csv"), mode="w") as f:
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
        with open(os.path.join(self.save_dirs["log"], "test_outputs.csv"), mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["outputs", "preds", "targets"])
            for i in range(len(outputs)):
                writer.writerow([outputs[i], preds[i], int(targets[i])])

    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        outputs = []
        preds = []
        targets = []
        for batch in tqdm(self.train_loader):
            x_ab, x_ag, target = batch
            targets.extend(target.tolist())
            x_ab, x_ag, target = self._to_device(x_ab, x_ag, target)
            self.optimizer.zero_grad()
            if self.amp:
                with torch.cuda.amp.autocast():
                    output = self.model(x_ab, x_ag)
                    output = torch.squeeze(output)
                    output_sig = torch.sigmoid(output)
                    outputs.extend(output_sig.tolist())
                    loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x_ab, x_ag)
                output = torch.squeeze(output)
                output_sig = torch.sigmoid(output)
                outputs.extend(output_sig.tolist())
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            train_loss += loss.item()
            pred = torch.round(output_sig)
            preds.extend(pred.tolist())
        train_loss = train_loss / len(self.train_loader)
        train_auroc = roc_auc_score(targets, outputs)
        train_accuracy = accuracy_score(targets, preds)
        train_precision = precision_score(targets, preds)
        train_recall = recall_score(targets, preds)
        train_f1 = f1_score(targets, preds)
        train_mcc = matthews_corrcoef(targets, preds)
        precisions, recalls, _ = precision_recall_curve(targets, outputs)
        train_auprc = auc(recalls, precisions)
        return (
            train_loss,
            train_auroc,
            train_auprc,
            train_accuracy,
            train_precision,
            train_recall,
            train_f1,
            train_mcc,
        )

    def _valid_epoch(self):
        self.model.eval()
        valid_loss = 0
        outputs = []
        preds = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                x_ab, x_ag, target = batch
                targets.extend(target.tolist())
                x_ab, x_ag, target = self._to_device(x_ab, x_ag, target)
                output = self.model(x_ab, x_ag)
                output = torch.squeeze(output)
                output_sig = torch.sigmoid(output)
                outputs.extend(output_sig.tolist())
                loss = self.criterion(output, target)
                valid_loss += loss.item()
                pred = torch.round(output_sig)
                preds.extend(pred.tolist())
        valid_loss = valid_loss / len(self.test_loader)
        valid_auroc = roc_auc_score(targets, outputs)
        valid_accuracy = accuracy_score(targets, preds)
        valid_precision = precision_score(targets, preds)
        valid_recall = recall_score(targets, preds)
        valid_f1 = f1_score(targets, preds)
        valid_mcc = matthews_corrcoef(targets, preds)
        precisions, recalls, _ = precision_recall_curve(targets, outputs)
        valid_auprc = auc(recalls, precisions)
        return (
            valid_loss,
            valid_auroc,
            valid_auprc,
            valid_accuracy,
            valid_precision,
            valid_recall,
            valid_f1,
            valid_mcc,
        )

    def _to_device(self, x_ab, x_ag, target):
        x_ab, x_ag, target = (
            x_ab.to(self.device),
            x_ag.to(self.device),
            target.to(self.device),
        )
        return (x_ab, x_ag, target)
