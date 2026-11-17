"""
LinguaGuard-FL Dataset Loader Module
Loads and tokenizes regional Indic/Dravidian language datasets:
Kannada, Tamil, Malayalam, Hindi.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from configs.config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    DATA_DIR
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SentimentDataset(Dataset):
    """
    PyTorch Dataset wrapper for multilingual sentiment reviews.
    Reads preprocessed CSV files with ['text', 'label'] columns.
    """
    def __init__(self, csv_file, max_length: int = MAX_LENGTH):
        self.df = pd.read_csv(csv_file)
        # Ensure text is clean string and labels are valid integers
        self.df["text"] = self.df["text"].fillna("").astype(str)
        self.df["label"] = self.df["label"].fillna(1).astype(int).clip(0, 2)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = int(self.df.iloc[idx]["label"])

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(language: str, batch_size: int = BATCH_SIZE):
    """
    Returns train and test DataLoaders for the specified language.
    Supports: 'kannada', 'tamil', 'malayalam', 'hindi' (and 'tanglish' -> 'tamil').
    """
    lang_key = "tamil" if language.lower() == "tanglish" else language.lower()
    lang_dir = DATA_DIR / lang_key

    train_path = lang_dir / "train.csv"
    test_path = lang_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing dataset files for {language} in {lang_dir}")

    train_ds = SentimentDataset(train_path)
    test_ds = SentimentDataset(test_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_ds), len(test_ds)


if __name__ == "__main__":
    for lang in ["kannada", "tamil", "malayalam", "hindi"]:
        tr_ld, te_ld, n_tr, n_te = get_dataloaders(lang)
        print(f"[{lang.upper()}] Train: {n_tr} samples, Test: {n_te} samples")
