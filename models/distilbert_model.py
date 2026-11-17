"""
Multilingual DistilBERT Model Wrapper Module
Manages model initialization, tokenizer loading, parameter extraction, and weight updates.
Compatible with both class-based interface and functional interface.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import MODEL_NAME, NUM_LABELS, DEVICE


class DistilBERTModel:
    """
    Object-oriented wrapper for multilingual DistilBERT model & tokenizer.
    """
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(DEVICE)

    def get_model(self) -> nn.Module:
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer


def load_model_and_tokenizer(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Loads pretrained multilingual DistilBERT model and its corresponding tokenizer.
    """
    wrapper = DistilBERTModel(model_name=model_name, num_labels=num_labels)
    return wrapper.get_model(), wrapper.get_tokenizer()


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extracts all model state dict parameters as a list of NumPy arrays (for Flower / FL communication).
    """
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Sets the model state dict from a list of NumPy arrays received from the server/aggregator.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, device=DEVICE) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
