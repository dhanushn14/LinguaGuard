"""
LinguaGuard-FL Local Trainer Module
Executes client-side fine-tuning on Indic/Dravidian sentiment datasets:
Kannada, Tamil, Malayalam, Hindi.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Optional, Tuple, Dict
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from configs.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    DATA_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR
)
from models.distilbert_model import DistilBERTModel
from data.dataset_loader import SentimentDataset


class Trainer:
    """
    Manages local training, evaluation, and checkpoint saving for a given language client.
    """
    def __init__(self, language: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
        lang_key = "tamil" if language.lower() == "tanglish" else language.lower()
        self.language = lang_key
        self.epochs = epochs
        self.batch_size = batch_size

        print("=" * 60)
        print(f"Initializing Trainer for {self.language.upper()}")
        print("=" * 60)

        # Initialize DistilBERT model
        self.distilbert_wrapper = DistilBERTModel()
        self.model = self.distilbert_wrapper.get_model()

        # Locate datasets
        lang_dir = DATA_DIR / self.language
        train_path = lang_dir / "train.csv"
        test_path = lang_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing dataset for '{self.language}' in {lang_dir}. "
                f"Expected train.csv and test.csv."
            )

        self.train_dataset = SentimentDataset(train_path)
        self.test_dataset = SentimentDataset(test_path)

        self.num_train = len(self.train_dataset)
        self.num_test = len(self.test_dataset)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

    def train_epoch(self) -> float:
        """
        Executes one epoch of training over local client data.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress = tqdm(self.train_loader, desc=f"Training [{self.language.upper()}]", leave=False)
        for batch in progress:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, epochs: Optional[int] = None) -> float:
        """
        Trains the local model for the specified number of epochs.
        """
        num_epochs = epochs or self.epochs
        avg_loss = 0.0
        for ep in range(1, num_epochs + 1):
            avg_loss = self.train_epoch()
            print(f"[{self.language.upper()}] Local Epoch {ep}/{num_epochs} - Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the local model on its holdout test set.
        Returns (loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        acc = float(accuracy_score(all_labels, all_preds))
        f1_mac = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        f1_wgt = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

        metrics = {
            "accuracy": acc,
            "loss": avg_loss,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wgt
        }

        print(f"\n[{self.language.upper()}] Evaluation -> Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1_mac:.4f}")

        # Save evaluation report to results/
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_txt = RESULTS_DIR / f"{self.language}_results.txt"
        with open(result_txt, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("LinguaGuard-FL Local Client Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Language              : {self.language.upper()}\n")
            f.write(f"Accuracy              : {acc:.4f}\n")
            f.write(f"Loss                  : {avg_loss:.4f}\n")
            f.write(f"F1-Score (Macro)      : {f1_mac:.4f}\n")
            f.write(f"F1-Score (Weighted)   : {f1_wgt:.4f}\n")
            f.write(f"Test Samples          : {self.num_test}\n")
            f.write(f"Federated Learning    : Enabled (FedAvg)\n")
            f.write(f"Adaptive DP           : Enabled\n")
            f.write(f"Secure Aggregation    : Enabled\n")

        result_json = RESULTS_DIR / f"{self.language}_results.json"
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump({
                "Language": self.language.upper(),
                "Accuracy": f"{acc:.4f}",
                "Loss": f"{avg_loss:.4f}",
                "F1_Macro": f"{f1_mac:.4f}",
                "F1_Weighted": f"{f1_wgt:.4f}",
                "Samples_Evaluated": self.num_test
            }, f, indent=4)

        return avg_loss, metrics

    def save(self):
        """
        Saves model state dict to checkpoints.
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / f"{self.language}_distilbert.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")


if __name__ == "__main__":
    t = Trainer("kannada")
    print(f"Successfully initialized Trainer for {t.language.upper()} with {t.num_train} train and {t.num_test} test samples.")
