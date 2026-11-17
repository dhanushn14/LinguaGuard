"""
LinguaGuard-FL Dataset Cleaning & Preprocessing Script
Cleans and normalizes sentiment review datasets for all 4 Indic languages:
Kannada, Tamil, Malayalam, Hindi.
Standardizes schema to ['text', 'label'] where 0=Negative, 1=Neutral/Mixed, 2=Positive.
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import DATA_DIR, RANDOM_SEED

# Universal label normalization rules
LABEL_MAPPING = {
    "negative": 0, "neg": 0, "0": 0, "-1": 0, "bad": 0,
    "neutral": 1, "mixed": 1, "mixed_feelings": 1, "mixed feelings": 1, "unknown": 1, "1": 1,
    "positive": 2, "pos": 2, "2": 2, "good": 2
}


def clean_dataframe(df: pd.DataFrame, text_col: str = None, label_col: str = None) -> pd.DataFrame:
    """
    Cleans raw dataframe: drops nulls, deduplicates, strips whitespace, maps labels.
    """
    # Detect text column if not provided
    if text_col is None:
        candidates = ["text", "review", "comment", "cleaned_text", "sentence", "body"]
        for c in candidates:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            text_col = df.columns[0]

    # Detect label column if not provided
    if label_col is None:
        candidates = ["label", "sentiment", "category", "score", "class", "target"]
        for c in candidates:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = df.columns[1]

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Drop nulls and duplicates
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])

    # Normalize labels
    def map_val(val):
        s = str(val).lower().strip()
        if s in LABEL_MAPPING:
            return LABEL_MAPPING[s]
        try:
            num = int(float(s))
            if num in [0, 1, 2]:
                return num
            elif num > 2:
                return 2
            elif num < 0:
                return 0
        except ValueError:
            pass
        return None

    df["label"] = df["label"].apply(map_val)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df


def process_language(language: str, raw_file_path: str, test_size: float = 0.2):
    """
    Cleans raw data file and creates train.csv and test.csv in data/{language}/
    """
    print("=" * 60)
    print(f"Cleaning Dataset for: {language.upper()}")
    print(f"Source file: {raw_file_path}")
    print("=" * 60)

    if raw_file_path.endswith(".tsv"):
        raw_df = pd.read_csv(raw_file_path, sep="\t")
    else:
        raw_df = pd.read_csv(raw_file_path)

    cleaned = clean_dataframe(raw_df)
    print(f"Retained {len(cleaned)} clean rows.")
    print("Class distribution:", cleaned["label"].value_counts().to_dict())

    # Split train and test
    train_df, test_df = train_test_split(
        cleaned,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=cleaned["label"] if cleaned["label"].nunique() > 1 else None
    )

    out_dir = DATA_DIR / language.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and prepare language datasets")
    parser.add_argument("--language", type=str, choices=["kannada", "tamil", "malayalam", "hindi"], help="Language code")
    parser.add_argument("--file", type=str, help="Path to raw csv/tsv file")
    args = parser.parse_args()

    if args.language and args.file:
        process_language(args.language, args.file)
    else:
        print("Usage example:")
        print("  python scripts/clean_dataset.py --language malayalam --file path/to/malayalam_raw.csv")
