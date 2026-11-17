"""
LinguaGuard-FL Central Server Module
Coordinates Federated Averaging (FedAvg) across 4 regional Indic language clients:
Kannada, Tamil, Malayalam, Hindi.
Collects masked model updates, aggregates parameters, and logs global telemetry.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json
import flwr as fl
from flwr.common import Metrics

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    NUM_ROUNDS,
    MIN_CLIENTS,
    FRACTION_FIT,
    FRACTION_EVAL,
    LANGUAGES,
    DP_ENABLED,
    SECURE_AGG_ENABLED,
    RESULTS_DIR
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Computes sample-size weighted average of evaluation metrics across clients.
    """
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}

    weighted_acc = sum([num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m])
    result = {"accuracy": weighted_acc / total_examples}

    if any("f1_macro" in m for _, m in metrics):
        weighted_f1 = sum([num_examples * m["f1_macro"] for num_examples, m in metrics if "f1_macro" in m])
        result["f1_macro"] = weighted_f1 / total_examples

    return result


def get_fit_config_fn():
    """
    Returns config dictionary sent to clients on each round.
    """
    def fit_config(server_round: int) -> Dict[str, str]:
        return {
            "server_round": str(server_round),
            "local_epochs": "2"
        }
    return fit_config


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = NUM_ROUNDS,
    min_clients: int = MIN_CLIENTS
):
    """
    Launches central Flower server for 4-client federated training.
    """
    print("=" * 65)
    print("   Starting LinguaGuard-FL Central Server (FedAvg)")
    print(f"   Target Languages: {', '.join([l.upper() for l in LANGUAGES])}")
    print(f"   Rounds: {num_rounds} | Required Clients: {min_clients}")
    print(f"   Adaptive Differential Privacy: {'ENABLED' if DP_ENABLED else 'DISABLED'}")
    print(f"   Secure Aggregation Simulation: {'ENABLED' if SECURE_AGG_ENABLED else 'DISABLED'}")
    print(f"   Listening on: {server_address}")
    print("=" * 65)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=get_fit_config_fn(),
        evaluate_metrics_aggregation_fn=weighted_average
    )

    grpc_max_message_length = 1024 * 1024 * 1024  # 1 GB
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=grpc_max_message_length
    )

    # Summarize final distributed results
    final_loss = 0.0
    final_acc = 0.0

    if history.losses_distributed:
        final_loss = history.losses_distributed[-1][1]
    if history.metrics_distributed and "accuracy" in history.metrics_distributed:
        final_acc = history.metrics_distributed["accuracy"][-1][1]

    print("\n" + "=" * 65)
    print("   LinguaGuard-FL Federated Training Complete")
    print(f"   Final Distributed Loss    : {final_loss:.4f}")
    print(f"   Final Distributed Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print("=" * 65)

    # Save to results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = RESULTS_DIR / "federated_results.txt"
    with open(res_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("LinguaGuard-FL Federated Multi-Language Evaluation\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Active Clients          : {min_clients}\n")
        f.write(f"Participating Languages : {', '.join([l.upper() for l in LANGUAGES])}\n")
        f.write(f"Federated Rounds        : {num_rounds}\n")
        f.write(f"Aggregation Algorithm   : FedAvg\n")
        f.write(f"Adaptive DP             : {'Enabled' if DP_ENABLED else 'Disabled'}\n")
        f.write(f"Secure Aggregation      : {'Enabled' if SECURE_AGG_ENABLED else 'Disabled'}\n")
        f.write(f"Final Distributed Loss  : {final_loss:.4f}\n")
        f.write(f"Final Distributed Acc   : {final_acc:.4f}\n")

    res_json = RESULTS_DIR / "federated_results.json"
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump({
            "Clients": min_clients,
            "Languages": ", ".join([l.upper() for l in LANGUAGES]),
            "Rounds": num_rounds,
            "Algorithm": "FedAvg",
            "Adaptive_DP": "Enabled" if DP_ENABLED else "Disabled",
            "Secure_Aggregation": "Enabled" if SECURE_AGG_ENABLED else "Disabled",
            "Final_Distributed_Loss": f"{final_loss:.4f}",
            "Final_Distributed_Accuracy": f"{final_acc:.4f}"
        }, f, indent=4)

    print(f"Results saved to: {res_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinguaGuard-FL Server")
    parser.add_argument(
        "--server_address",
        "--address",
        dest="server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Address and port to bind (default: 0.0.0.0:8080)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of federated training rounds (default: {NUM_ROUNDS})"
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=MIN_CLIENTS,
        help=f"Minimum required clients (default: {MIN_CLIENTS})"
    )
    args = parser.parse_args()

    start_server(args.server_address, args.rounds, args.min_clients)
