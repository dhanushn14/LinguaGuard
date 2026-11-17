"""
LinguaGuard-FL Flower Client Module
Connects local language client to the central federated server over gRPC.
Applies Adaptive Differential Privacy and Secure Aggregation before transmitting weights.
Supports 4 Languages: Kannada, Tamil, Malayalam, Hindi.
"""

import sys
import argparse
import gc
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Tuple
import numpy as np
import torch
import flwr as fl

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import DEVICE, DP_ENABLED, SECURE_AGG_ENABLED, LANGUAGES
from training.trainer import Trainer
from privacy.adaptive_dp import AdaptiveDifferentialPrivacy
from privacy.secure_aggregation import SecureAggregation
from models.distilbert_model import get_model_parameters, set_model_parameters


class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing a regional Indic/Dravidian language node.
    """
    def __init__(self, language: str):
        lang_key = "tamil" if language.lower() == "tanglish" else language.lower()
        self.language = lang_key

        print("=" * 60)
        print(f"Starting Flower Client: {self.language.upper()}")
        print("=" * 60)

        # Initialize local language trainer
        self.trainer = Trainer(self.language)

        # Initialize privacy defenses
        self.dp = AdaptiveDifferentialPrivacy() if DP_ENABLED else None
        self.secure_agg = SecureAggregation() if SECURE_AGG_ENABLED else None

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Returns current local model parameters.
        """
        return get_model_parameters(self.trainer.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates local model with parameters received from the server.
        """
        set_model_parameters(self.trainer.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receives global model, trains locally, applies Adaptive DP + Secure Aggregation,
        and returns masked parameters.
        """
        print(f"\n---> [Client {self.language.upper()}] Received Global Model. Starting local training...")

        # 1. Update local model with global parameters
        self.set_parameters(parameters)

        # 2. Extract configuration
        epochs = int(config.get("local_epochs", self.trainer.epochs))

        # 3. Local training on language dataset
        train_loss = self.trainer.train(epochs=epochs)

        # 4. Extract updated parameters
        updated_parameters = self.get_parameters(config={})

        # 5. Apply Adaptive Differential Privacy (clipping + Gaussian noise)
        if self.dp is not None:
            print("Applying Adaptive Differential Privacy...")
            updated_parameters = self.dp.privatize(updated_parameters)
            print("Differential Privacy Applied!")

        # 6. Apply Secure Aggregation (zero-sum Gaussian masking simulation)
        if self.secure_agg is not None:
            print("Applying Secure Aggregation...")
            updated_parameters = self.secure_agg.mask_parameters(updated_parameters)
            print("Secure Aggregation Applied!")

        print(f"Sending Updated Model Parameters from [{self.language.upper()}] to Federated Server.\n")
        # Free memory and temporary activations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return updated_parameters, self.trainer.num_train, {"train_loss": float(train_loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """
        Evaluates received parameters on client's local holdout test set.
        """
        self.set_parameters(parameters)
        loss, metrics = self.trainer.evaluate()
        return (
            float(loss),
            self.trainer.num_test,
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics.get("f1_macro", 0.0))
            }
        )


def start_client(language: str, server_address: str = "127.0.0.1:8080"):
    """
    Spawns the Flower client process with configured 1GB gRPC buffer.
    """
    client = FlowerClient(language=language)
    grpc_max_message_length = 1024 * 1024 * 1024  # 1 GB
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client(),
            grpc_max_message_length=grpc_max_message_length
        )
    except (AttributeError, TypeError):
        try:
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client,
                grpc_max_message_length=grpc_max_message_length
            )
        except TypeError:
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinguaGuard-FL Client Node")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["kannada", "tamil", "malayalam", "hindi", "tanglish"],
        help="Target language (kannada, tamil, malayalam, hindi, tanglish)"
    )
    parser.add_argument(
        "--server",
        "--server_address",
        dest="server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Central server gRPC address:port (default: 127.0.0.1:8080)"
    )
    args = parser.parse_args()

    start_client(args.language, args.server_address)
