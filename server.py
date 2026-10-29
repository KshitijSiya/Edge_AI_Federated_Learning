"""
Federated Learning Server: Edge AI in the Real World (Module 1)
---------------------------------------------------------------
Project: Privacy-Preserving AI for Smart Gym Wearables
Dataset: UCI Human Activity Recognition (HAR)
Framework: Flower (flwr) & Scikit-Learn

Concept:
In traditional Centralized Cloud AI, edge wearables stream raw, high-frequency
biometric and motion sensor streams (accelerometer, gyroscope) to a central cloud.
This introduces serious privacy risks, network latency, and high radio battery drain.

In this Edge AI / Federated Learning architecture:
1. The server acts as a coordinator / aggregator.
2. Raw biometric motion data NEVER leaves the edge wearable.
3. The server broadcasts the global model parameters to participating edge clients.
4. Edge clients perform local training on their private sensor slice.
5. Clients send back ONLY model weight updates (coefficients and biases).
6. The server applies Federated Averaging (FedAvg; McMahan et al., 2017) to aggregate updates:
       w_{t+1} = sum( (n_k / n) * w_{t+1}^k )
7. The updated global model is broadcast back to edge wearables for local inference.
"""

import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import flwr as fl
from flwr.common import Metrics, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays


# ==============================================================================
# 1. METRICS AGGREGATION
# ==============================================================================
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Computes weighted average of evaluation metrics across all participating edge clients.
    
    Args:
        metrics: List of tuples containing:
                 - number of local evaluation examples (n_k)
                 - dictionary of metrics reported by the client (e.g., {"accuracy": 0.85})
                 
    Returns:
        Aggregated dictionary containing the global weighted accuracy.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]

    if not examples:
        return {}

    aggregated_accuracy = sum(accuracies) / sum(examples)
    print(f"\n[Server Aggregator] -> Aggregated Global Round Accuracy: {aggregated_accuracy * 100:.2f}% (over {sum(examples)} samples)")
    return {"accuracy": aggregated_accuracy}


# ==============================================================================
# 2. GLOBAL INITIAL PARAMETERS
# ==============================================================================
def get_initial_parameters() -> Parameters:
    """
    Initializes global model parameters for 6 HAR activity classes and 561 feature inputs.
    
    Classes in UCI HAR:
        1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS,
        4: SITTING, 5: STANDING, 6: LAYING
        
    Dimensions:
        - Weights (W): shape (6, 561) - 6 classes, 561 sensor features
        - Intercepts (b): shape (6,)  - bias term for each class
        
    Returns:
        flwr.common.Parameters: Initialized zero weights serialized for gRPC distribution.
    """
    num_classes = 6
    num_features = 561
    
    initial_weights = np.zeros((num_classes, num_features), dtype=np.float32)
    initial_intercepts = np.zeros(num_classes, dtype=np.float32)
    
    # Pack into Flower Parameters
    return ndarrays_to_parameters([initial_weights, initial_intercepts])


# ==============================================================================
# 3. FEDAVG STRATEGY WITH LOGGING & PERSISTENCE
# ==============================================================================
class EdgeSaveModelFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that logs convergence metrics and saves
    the final aggregated model parameters to disk for edge deployment.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"\n[Round {server_round}] -> Successfully aggregated updates from {len(results)} edge client(s).")
            # Convert Flower Parameters back to NumPy arrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            weights, intercepts = ndarrays[0], ndarrays[1]
            
            # Save the latest global model for edge inference verification
            np.savez("global_model.npz", weights=weights, intercepts=intercepts)
            print(f"[Round {server_round}] -> Checkpoint saved to 'global_model.npz' (Weights: {weights.shape}, Biases: {intercepts.shape})")

        return aggregated_parameters, aggregated_metrics


# ==============================================================================
# 4. MAIN SERVER ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Central Aggregator for Smart Gym Wearables")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated training rounds (default: 3)")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of edge clients required to train (default: 2)")
    parser.add_argument("--port", type=str, default="8080", help="Port to listen for client connections (default: 8080)")
    args = parser.parse_args()

    server_address = f"127.0.0.1:{args.port}"

    print("=" * 75)
    print("  FEDERATED LEARNING CLOUD AGGREGATOR - EDGE AI SIMULATION")
    print("=" * 75)
    print(f"Server Address:        {server_address}")
    print(f"Federated Rounds:      {args.rounds}")
    print(f"Min Participating Dev: {args.min_clients}")
    print("Privacy Guarantee:     Zero raw sensor data collected. Only model weights aggregated.")
    print("=" * 75)

    # Define the Federated Averaging strategy
    strategy = EdgeSaveModelFedAvg(
        fraction_fit=1.0,                       # Sample 100% of available edge clients
        fraction_evaluate=1.0,                  # Sample 100% of available edge clients for evaluation
        min_fit_clients=args.min_clients,       # Never start a round with fewer than min_clients
        min_evaluate_clients=args.min_clients,  # Min clients required for federated evaluation
        min_available_clients=args.min_clients, # Wait until at least this many devices connect
        evaluate_metrics_aggregation_fn=weighted_average, # Aggregate accuracy across clients
        initial_parameters=get_initial_parameters(),       # Baseline model parameters
    )

    # Launch gRPC Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
