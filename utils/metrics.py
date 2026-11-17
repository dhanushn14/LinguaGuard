"""
Metrics & Evaluation Utilities
Calculates accuracy, precision, recall, F1-score, and persists results to formatted summary files.
"""

from typing import Dict, List, Tuple
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes standard multi-class classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted)
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 3) -> np.ndarray:
    """
    Generates confusion matrix for the predictions.
    """
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def save_results(results_dict: dict, file_path: Path) -> None:
    """
    Saves results dictionary to JSON and human-readable text file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    # Also save as structured text report matching project report format
    with open(file_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  LinguaGuard-FL Evaluation Results\n")
        f.write("=" * 60 + "\n")
        for k, v in results_dict.items():
            f.write(f"{k.ljust(25)}: {v}\n")
        f.write("=" * 60 + "\n")
