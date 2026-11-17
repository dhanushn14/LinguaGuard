"""
Federated Simulation Runner
Executes an end-to-end multi-client Federated Learning simulation across all 4 languages:
Malayalam, Hindi, Tanglish, and Romanized Kannada, applying Adaptive DP and Secure Aggregation.
Runs completely within a single process/script using Flower simulation.
"""

import sys
from pathlib import Path
import flwr as fl

# Ensure root directory is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    LANGUAGES,
    NUM_ROUNDS,
    FRACTION_FIT,
    FRACTION_EVAL,
    DP_ENABLED,
    DP_CLIP_NORM,
    DP_NOISE_MULTIPLIER,
    SECURE_AGG_ENABLED,
    SECURE_AGG_MASK_SCALE,
    RESULTS_DIR
)
from training.trainer import Trainer
from clients.client import FlowerClient
from server.server import weighted_average, get_fit_config_fn
from utils.metrics import save_results


def client_fn(cid: str) -> fl.client.Client:
    """
    Client factory function for Flower simulation.
    Maps client ID ("0", "1", "2", "3") to language client instance.
    """
    client_idx = int(cid) % len(LANGUAGES)
    language = LANGUAGES[client_idx]
    
    trainer = Trainer(language=language)
    client = FlowerClient(language=language, trainer=trainer)
    return client.to_client()


def run_simulation(num_rounds: int = NUM_ROUNDS):
    """
    Runs the multi-client FL simulation.
    """
    print("=" * 70)
    print("  Starting LinguaGuard-FL Federated Multi-Language Simulation")
    print(f"  Languages: {', '.join([l.upper() for l in LANGUAGES])}")
    print(f"  Number of Rounds: {num_rounds}")
    print(f"  Adaptive Differential Privacy: {'ENABLED' if DP_ENABLED else 'DISABLED'} (Clip: {DP_CLIP_NORM}, Noise: {DP_NOISE_MULTIPLIER})")
    print(f"  Secure Aggregation Simulation: {'ENABLED' if SECURE_AGG_ENABLED else 'DISABLED'} (Scale: {SECURE_AGG_MASK_SCALE})")
    print("=" * 70)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=len(LANGUAGES),
        min_evaluate_clients=len(LANGUAGES),
        min_available_clients=len(LANGUAGES),
        on_fit_config_fn=get_fit_config_fn(),
        evaluate_metrics_aggregation_fn=weighted_average
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(LANGUAGES),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25 if False else 0}
    )

    # Summarize final results
    final_loss = history.losses_distributed[-1][1] if history.losses_distributed else 0.0
    final_acc = 0.0
    if history.metrics_distributed and "accuracy" in history.metrics_distributed:
        final_acc = history.metrics_distributed["accuracy"][-1][1]

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETED SUCCESSFULLY")
    print(f"  Final Distributed Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  Final Distributed Loss    : {final_loss:.4f}")
    print("=" * 70)

    save_results(
        {
            "Total_Clients": len(LANGUAGES),
            "Languages": ", ".join(LANGUAGES),
            "Rounds": num_rounds,
            "Algorithm": "FedAvg",
            "Adaptive_DP": "Enabled" if DP_ENABLED else "Disabled",
            "Secure_Aggregation": "Enabled" if SECURE_AGG_ENABLED else "Disabled",
            "Final_Distributed_Accuracy": f"{final_acc:.4f}",
            "Final_Distributed_Loss": f"{final_loss:.4f}"
        },
        RESULTS_DIR / "federated_simulation_results"
    )

    return history


if __name__ == "__main__":
    run_simulation()
