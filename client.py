"""
Federated Learning Client: Edge AI in the Real World (Module 1)
---------------------------------------------------------------
Project: Privacy-Preserving AI for Smart Gym Wearables
Dataset: UCI Human Activity Recognition (HAR)
Framework: Flower (flwr) & Scikit-Learn

Simulated Edge Device:
Each instance of this script simulates a smart gym wearable (e.g., smartwatch,
fitness tracker, or chest strap) worn by a specific gym user (Subject ID).

Edge AI vs. Cloud AI Core Principles Demonstrated:
1. PRIVACY:
   - Raw triaxial accelerometer/gyroscope readings (561 features) stay strictly
     in local memory on the wearable.
   - Zero raw user motion data is transmitted to the server.
2. LATENCY:
   - Inference and local training are executed directly on the edge CPU.
   - Local training takes mere milliseconds without round-trip network hops.
3. POWER EFFICIENCY:
   - Wireless radio transmission consumes massive power compared to local compute.
   - Instead of streaming continuous 50Hz sensor data 24/7, transmission happens
     only during scheduled federated model synchronization rounds.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import flwr as fl


# ==============================================================================
# 1. LOCAL DATASET PARTITIONING (NON-IID EDGE SLICE)
# ==============================================================================
def load_subject_data(
    data_dir: str,
    subject_id: int,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and partitions the UCI HAR dataset for a single specific volunteer (Subject).
    
    In the real world, this simulates the local sensor partition collected
    by a smart wearable on one gym member's body.

    Args:
        data_dir: Directory containing 'train' and 'test' folders.
        subject_id: Volunteer ID (1 to 30) representing the edge user.
        test_size: Fraction of local data reserved for on-device testing.
        random_state: Seed for reproducible local split.

    Returns:
        X_train, X_test, y_train, y_test: Partitioned feature matrices and activity labels.
    """
    print(f"\n[Edge Device] Loading local sensor partition for Subject #{subject_id}...")

    # 1. Determine whether the subject resides in 'train' or 'test' partition
    train_sub_path = os.path.join(data_dir, "train", "subject_train.txt")
    train_subs = pd.read_csv(train_sub_path, header=None)[0]

    if subject_id in set(train_subs):
        split_name = "train"
        y_path = os.path.join(data_dir, "train", "y_train.txt")
        x_path = os.path.join(data_dir, "train", "X_train.txt")
        subjects = train_subs
    else:
        test_sub_path = os.path.join(data_dir, "test", "subject_test.txt")
        test_subs = pd.read_csv(test_sub_path, header=None)[0]
        if subject_id in set(test_subs):
            split_name = "test"
            y_path = os.path.join(data_dir, "test", "y_test.txt")
            x_path = os.path.join(data_dir, "test", "X_test.txt")
            subjects = test_subs
        else:
            raise ValueError(
                f"Subject #{subject_id} not found in UCI HAR dataset. "
                f"Valid subjects are integers 1 through 30."
            )

    labels = pd.read_csv(y_path, header=None)[0]

    # Find the row indices belonging to the specified subject
    subject_indices = subjects[subjects == subject_id].index.tolist()

    # 2. Extract only rows belonging to this subject
    # (Skip rows not matching subject indices for fast I/O)
    index_set = set(subject_indices)
    features_df = pd.read_csv(
        x_path,
        sep=r"\s+",
        header=None,
        skiprows=lambda i: i not in index_set,
        engine="c",
    )

    X = features_df.values.astype(np.float32)
    y = labels.iloc[subject_indices].values.astype(np.int32)

    # 3. Local Train/Test split for on-device validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. FLOWER NUMPY CLIENT (EDGE WEARABLE)
# ==============================================================================
class WearableClient(fl.client.NumPyClient):
    """
    Simulates a smart wearable edge device participating in Federated Learning.
    
    Communicates with the central aggregator using Flower's NumPyClient interface.
    Model weights and biases are synchronized without exposing raw user sensor data.
    """
    def __init__(
        self,
        subject_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        local_epochs: int = 3,
    ):
        self.subject_id = subject_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.local_epochs = local_epochs

        # 6 HAR Activity classes: 1 to 6
        self.classes = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)

        # SGDClassifier with log_loss provides multinomial logistic regression
        # trained via mini-batch stochastic gradient descent on the edge CPU
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            warm_start=True,
            random_state=42,
        )

        # Initialize local weights and intercepts to zero
        self.model.coef_ = np.zeros((len(self.classes), X_train.shape[1]), dtype=np.float32)
        self.model.intercept_ = np.zeros(len(self.classes), dtype=np.float32)
        self.model.classes_ = self.classes

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Returns the current local model parameters (Weights and Intercepts).
        Called by the Flower framework to inspect model state.
        """
        return [self.model.coef_, self.model.intercept_]

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Executes local training on the wearable edge device.
        
        Steps:
        1. Overwrite local model parameters with the latest global parameters from the server.
        2. Perform local Stochastic Gradient Descent (SGD) for multiple epochs.
        3. Return updated parameters and sample count (n_k) to the central aggregator.
        """
        print(f"\n[Wearable #{self.subject_id}] ---> Starting Federated Fit Round...")

        # Step 1: Ingest global model parameters from central server
        self.model.coef_ = np.array(parameters[0], dtype=np.float32, copy=True)
        self.model.intercept_ = np.array(parameters[1], dtype=np.float32, copy=True)

        # Step 2: Measure local edge training latency (compute benchmark)
        start_time = time.perf_counter()

        # Run local SGD passes over the private edge dataset
        for epoch in range(self.local_epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=self.classes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        print(
            f"[Wearable #{self.subject_id}] Local training complete: "
            f"{self.local_epochs} epoch(s) on {len(self.X_train)} samples "
            f"in {elapsed_ms:.2f} ms (Edge Compute Latency)"
        )
        print(
            f"[Wearable #{self.subject_id}] PRIVACY LOG: "
            f"0 bytes raw sensor data transmitted. Sending only model weights ({self.model.coef_.nbytes / 1024:.2f} KB)."
        )

        # Step 3: Return updated local parameters and sample count (n_k)
        return [self.model.coef_, self.model.intercept_], len(self.X_train), {"latency_ms": elapsed_ms}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluates the global model parameters locally on the wearable's private holdout set.
        
        Returns:
            loss: Cross-entropy loss on local holdout.
            num_examples: Count of local test samples.
            metrics: Dictionary containing local classification accuracy.
        """
        print(f"\n[Wearable #{self.subject_id}] ---> Evaluating Global Model on Local Holdout...")

        # Update local model with global parameters to evaluate
        self.model.coef_ = np.array(parameters[0], dtype=np.float32, copy=True)
        self.model.intercept_ = np.array(parameters[1], dtype=np.float32, copy=True)

        # Predict classes and probabilities
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)

        # Calculate accuracy and cross-entropy loss
        accuracy = float(accuracy_score(self.y_test, y_pred))
        loss = float(log_loss(self.y_test, y_proba, labels=self.classes))

        print(
            f"[Wearable #{self.subject_id}] Local Evaluation Result -> "
            f"Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f} "
            f"(Tested on {len(self.X_test)} samples)"
        )

        return loss, len(self.X_test), {"accuracy": accuracy}


# ==============================================================================
# 3. MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulated Gym Wearable Edge Client (UCI HAR)")
    parser.add_argument("--subject", type=int, default=1, help="Volunteer Subject ID (e.g. 1, 3, 5, 6, 7...)")
    parser.add_argument("--epochs", type=int, default=3, help="Local training epochs per round (default: 3)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Aggregator server address (default: 127.0.0.1:8080)")
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data_dir", type=str, default=default_dir, help=f"Root directory containing HAR data (default: {default_dir})")
    args = parser.parse_args()

    # Load local sensor slice for the specified subject
    X_train, X_test, y_train, y_test = load_subject_data(
        data_dir=args.data_dir,
        subject_id=args.subject,
    )

    # Telemetry and Edge vs. Cloud Metrics Summary
    raw_data_bytes = (X_train.nbytes + X_test.nbytes)
    model_param_bytes = (6 * 561 * 4) + (6 * 4)  # 6 classes * 561 float32 weights + 6 biases

    print("=" * 75)
    print(f"  EDGE WEARABLE SIMULATION: GYM MEMBER #{args.subject}")
    print("=" * 75)
    print(f"Local Training Samples:    {len(X_train)} time windows")
    print(f"Local Testing Samples:     {len(X_test)} time windows")
    print(f"Sensor Feature Dimension:  561 features (Acc / Gyro)")
    print(f"Raw Sensor Memory:         {raw_data_bytes / 1024:.2f} KB (STAYS ON DEVICE)")
    print(f"Model Parameter Size:      {model_param_bytes / 1024:.2f} KB (Transmitted to Aggregator)")
    print(f"Bandwidth / Privacy Gain:  {(1.0 - (model_param_bytes / raw_data_bytes)) * 100:.1f}% reduction vs raw streaming")
    print(f"Connecting to Aggregator:  {args.server}")
    print("=" * 75)

    # Initialize edge client
    client = WearableClient(
        subject_id=args.subject,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        local_epochs=args.epochs,
    )

    # Connect to Flower central server
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
