# Privacy-Preserving Federated Learning for Smart Gym Wearables
**Edge AI in the Real World — Module 1 Assignment**

---

## 📌 Project Overview
This repository implements a local **Federated Learning (FL)** simulation designed to prove why **Edge AI** is superior to traditional **Centralized Cloud AI** for smart gym wearable devices.

Using the **UCI Human Activity Recognition (HAR)** dataset and the **Flower (`flwr`)** framework, this project simulates multiple wearable edge devices collaborating to train a global activity recognition model without sharing sensitive personal sensor data with the cloud.

---

## ⚖️ Edge AI vs. Centralized Cloud AI Trade-offs

| Metric | Centralized Cloud AI | Edge AI + Federated Learning (FL) |
| :--- | :--- | :--- |
| **Privacy & Security** | Continuous raw sensor streaming (accelerometer/gyroscope) exposes user biometrics to cloud interception, data breaches, and unauthorized surveillance. | **Zero raw data leaves the edge device.** Only mathematical model parameter updates (weights and biases) are transmitted to the aggregator. |
| **Latency & Responsiveness** | Inference depends on continuous network connectivity, round-trip cloud ping, and server queue latency. | **Zero-latency on-device inference.** Wearable classifies gym movements in real time even when offline. |
| **Power Consumption** | High radio duty cycle: transmitting continuous 50 Hz triaxial sensor streams quickly depletes wearable battery life. | **Low power profile.** On-chip edge compute consumes milliwatts; wireless transmission only occurs periodically during model synchronization. |

---

## 📊 Dataset: UCI Human Activity Recognition (HAR)
The dataset comprises sensor signals gathered from 30 volunteers wearing a smartphone on their waist performing six activities:
1. `WALKING`
2. `WALKING_UPSTAIRS`
3. `WALKING_DOWNSTAIRS`
4. `SITTING`
5. `STANDING`
6. `LAYING`

Each sample contains a **561-feature vector** with time and frequency domain variables. In our simulation, edge clients are partitioned by **Subject ID**, directly modeling realistic **heterogeneous (Non-IID)** real-world edge data.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/KshitijSiya/Edge-AI-Federated-Learning.git
cd Edge-AI-Federated-Learning

# Create and activate Python virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Central Aggregator Server
```bash
python server.py --rounds 3 --min_clients 2
```

### 3. Launch Edge Clients (Wearables)
Open separate terminals for each simulated gym member wearable:

```bash
# Terminal 2: Edge Wearable 1 (Subject 1)
python client.py --subject 1

# Terminal 3: Edge Wearable 2 (Subject 2)
python client.py --subject 2
```

### 4. Interactive Web Dashboard (Streamlit Frontend)
You can also run the graphical mission control to simulate fleets, view interactive telemetry charts, and test real-time on-device inference:
```bash
streamlit run app.py
```

---

## 🔬 Mathematical Aggregation (FedAvg)
The central server coordinates rounds using the **Federated Averaging** algorithm (McMahan et al., 2017):

$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k$$

Where:
- $K$ is the number of active edge devices in the round.
- $n_k$ is the number of training samples on edge device $k$.
- $n = \sum n_k$ is the total number of samples across participating devices.
- $w_{t+1}^k$ represents the locally trained model parameters submitted by client $k$.

---

## 🏷️ Releases & Tags
- `v0.0`: Initial release featuring project architecture, dataset partitioning specification, and `server.py` implementation with custom FedAvg checkpointing.
