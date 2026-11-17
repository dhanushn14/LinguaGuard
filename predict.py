"""
LinguaGuard-FL Inference Script
Enables real-time sentiment prediction on multilingual and code-switched text
(Malayalam, Hindi, Tanglish, Romanized Kannada) using the trained federated model.
"""

import sys
import os
import zipfile
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from configs.config import DEVICE, LABEL_MAPPING, MAX_SEQ_LENGTH, CHECKPOINT_DIR, MODEL_NAME


def resolve_model_directory() -> Path:
    """
    Locates the trained model directory.
    If a downloaded 'final_linguaguard_model.zip' is found in checkpoints/ or root,
    it automatically extracts it.
    """
    candidate_paths = [
        CHECKPOINT_DIR / "final_linguaguard_model",
        BASE_DIR / "final_linguaguard_model",
        CHECKPOINT_DIR / "kannada_model"
    ]

    for p in candidate_paths:
        if p.exists() and (p / "config.json").exists():
            return p

    # Check for zip file downloaded from Colab or Kaggle
    zip_candidates = [
        CHECKPOINT_DIR / "final_linguaguard_model.zip",
        BASE_DIR / "final_linguaguard_model.zip",
        Path.home() / "Downloads" / "final_linguaguard_model.zip"
    ]

    for z in zip_candidates:
        if z.exists():
            print(f"--> Found downloaded model archive at {z}. Extracting to {CHECKPOINT_DIR / 'final_linguaguard_model'}...")
            extract_target = CHECKPOINT_DIR / "final_linguaguard_model"
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(extract_target)
            return extract_target

    print(f"[NOTE] No custom trained checkpoint found in {CHECKPOINT_DIR}.")
    print(f"Using default base model '{MODEL_NAME}' for inference demonstration.")
    print("To use your fine-tuned weights, place 'final_linguaguard_model.zip' into 'checkpoints/'.\n")
    return None


class SentimentPredictor:
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = resolve_model_directory()

        if model_path and model_path.exists():
            print(f"Loading trained LinguaGuard-FL model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_class = int(probs.argmax())

        return {
            "text": text,
            "label_id": pred_class,
            "sentiment": LABEL_MAPPING.get(pred_class, "Unknown"),
            "confidence": float(probs[pred_class]),
            "probabilities": {
                LABEL_MAPPING[0]: float(probs[0]),
                LABEL_MAPPING[1]: float(probs[1]),
                LABEL_MAPPING[2]: float(probs[2])
            }
        }


def run_interactive():
    print("=" * 65)
    print("   LinguaGuard-FL: Multilingual Sentiment Predictor")
    print("   Supported: Malayalam, Hindi, Tanglish, Romanized Kannada")
    print("   Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    predictor = SentimentPredictor()

    while True:
        try:
            user_input = input("\nEnter text / review: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting.")
                break

            result = predictor.predict(user_input)
            print("-" * 50)
            print(f"Predicted Sentiment : {result['sentiment'].upper()} (Confidence: {result['confidence']*100:.2f}%)")
            print("Class Breakdown:")
            for k, v in result["probabilities"].items():
                bar = "=" * int(v * 20)
                print(f"  {k.ljust(15)} : {v*100:5.2f}% [{bar.ljust(20)}]")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinguaGuard-FL Inference Engine")
    parser.add_argument("--text", type=str, default=None, help="Input text review to classify")
    parser.add_argument("--language", "--lang", type=str, default=None, help="Optional language tag (kannada, tamil, malayalam, hindi)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model directory")
    args = parser.parse_args()

    if args.text:
        predictor = SentimentPredictor(Path(args.model_path) if args.model_path else None)
        res = predictor.predict(args.text)
        print(f"\nText       : {res['text']}")
        if args.language:
            print(f"Language   : {args.language.upper()}")
        print(f"Sentiment  : {res['sentiment']}")
        print(f"Confidence : {res['confidence']*100:.2f}%\n")
    else:
        run_interactive()
