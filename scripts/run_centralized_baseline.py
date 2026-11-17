"""
Centralized Baseline Training Script
Trains a centralized multilingual DistilBERT model on pooled multi-language data
(Kannada, Tamil/Tanglish, Hindi, Malayalam) without federated partitioning and without differential privacy.
Used to calculate the utility benchmark and substantiate the exact privacy-utility tradeoff for the research thesis.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import LANGUAGES, PROCESSED_DATA_DIR, RESULTS_DIR, RANDOM_SEED, LOCAL_EPOCHS
from models.distilbert_model import load_model_and_tokenizer
from data.dataset_loader import load_language_dataloaders
from training.trainer import Trainer
from utils.metrics import save_results


def run_centralized_baseline(epochs: int = LOCAL_EPOCHS):
    print("=" * 70)
    print("  Running Centralized Baseline Training (All 4 Languages Combined)")
    print("  Used as Benchmark for Federated Privacy-Utility Tradeoff Analysis")
    print("=" * 70)

    # 1. Pool all available cleaned language datasets
    pooled_dfs = []
    for lang in LANGUAGES:
        p = PROCESSED_DATA_DIR / f"{lang}_cleaned.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["language"] = lang
            pooled_dfs.append(df)
            print(f"Loaded {lang.upper()}: {len(df)} samples")

    if not pooled_dfs:
        raise FileNotFoundError(f"No cleaned datasets found in {PROCESSED_DATA_DIR}. Run scripts/clean_dataset.py first.")

    pooled_df = pd.concat(pooled_dfs, ignore_index=True)
    pooled_df = shuffle(pooled_df, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\nTotal Pooled Dataset Size: {len(pooled_df)} samples across {len(LANGUAGES)} languages.")

    # 2. Build model, tokenizer, and loaders
    model, tokenizer = load_model_and_tokenizer()
    train_loader, test_loader, n_train, n_test = load_language_dataloaders(
        language="pooled",
        tokenizer=tokenizer,
        custom_df=pooled_df
    )

    # 3. Train
    trainer = Trainer(
        language="centralized_multilingual",
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs
    )

    print(f"\nStarting Centralized Fine-Tuning for {epochs} epochs...")
    final_train_loss = trainer.train()

    # 4. Evaluate
    print("\nEvaluating Centralized Model on Multi-Language Test Set...")
    eval_loss, metrics = trainer.evaluate()

    print("\n" + "=" * 70)
    print(f"  Centralized Baseline Results:")
    print(f"  Test Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Test Loss     : {metrics['loss']:.4f}")
    print(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
    print("=" * 70)

    save_results(
        {
            "Experiment": "Centralized_Non_Private_Baseline",
            "Languages": ", ".join(LANGUAGES),
            "Total_Samples": len(pooled_df),
            "Train_Samples": n_train,
            "Test_Samples": n_test,
            "Epochs": epochs,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Loss": f"{metrics['loss']:.4f}",
            "F1_Macro": f"{metrics['f1_macro']:.4f}",
            "F1_Weighted": f"{metrics['f1_weighted']:.4f}"
        },
        RESULTS_DIR / "centralized_baseline_results"
    )

    # Save checkpoint
    trainer.save_checkpoint()


if __name__ == "__main__":
    run_centralized_baseline()
