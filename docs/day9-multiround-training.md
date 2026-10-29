
# Day 9 — Multi-Round Federated Learning Simulation

## Objective

Implement and verify multi-round federated learning
using five clients, IID data partitioning, local CNN
training, FedAvg aggregation, and global validation.

## Implementation

### Federated Coordinator

File: `federated/server/coordinator.py`

The FederatedCoordinator:

1. Initializes a global CNN using ExperimentConfig.
2. Partitions training data across clients using IID.
3. Distributes the current global model to clients.
4. Executes independent local training.
5. Aggregates client updates using FedAvg.
6. Loads aggregated weights into the global model.
7. Evaluates the model on the validation dataset.
8. Stores metrics in RoundResult objects.
9. Repeats the process for all configured rounds.

### Experiment Runner

File: `federated/server/run_federated.py`

The experiment runner supports configurable:

- Dataset
- Number of clients
- Federated rounds
- Local epochs
- Batch size
- Learning rate
- Random seed
- Device

It saves JSON experiment results and a final
PyTorch global model checkpoint.

## Experiment Configuration

| Parameter | Value |
|---|---|
| Clients | 5 |
| Federated rounds | 3 |
| Local epochs | 1 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Random seed | 42 |
| Partitioning | IID |
| Aggregation | FedAvg |
| Device | CPU |

## Fashion-MNIST Results

Training samples: 54,000
Validation samples: 6,000

| Round | Validation Accuracy | Loss | Macro-F1 | Time (s) |
|---|---|---|---|---|
| 1 | 82.55% | 0.5211 | 0.8250 | 13.68 |
| 2 | 85.95% | 0.4074 | 0.8599 | 13.85 |
| 3 | 87.03% | 0.3607 | 0.8694 | 13.39 |

Final validation accuracy: 87.03%
Final validation Macro-F1: 0.8694

## PneumoniaMNIST Results

Training samples: 4,708
Validation samples: 524

| Round | Validation Accuracy | Loss | Macro-F1 | Time (s) |
|---|---|---|---|---|
| 1 | 74.24% | 0.4822 | 0.4261 | 1.24 |
| 2 | 89.69% | 0.2714 | 0.8457 | 1.14 |
| 3 | 93.70% | 0.1759 | 0.9145 | 1.14 |

Final validation accuracy: 93.70%
Final validation Macro-F1: 0.9145

## Testing

Test file: `tests/test_federated_coordinator.py`

Seven automated tests verify:

- Multiple federated rounds complete.
- Global model weights change after training.
- Validation metrics are valid.
- Client participation and aggregation weights.
- Unsupported aggregation methods are rejected.
- Unsupported partition types are rejected.
- Deterministic results with a fixed seed.

Full project test suite:

93 passed, 1 warning.

## Generated Artifacts

Fashion-MNIST:

- results/federated/fashion_mnist_fedavg.json
- checkpoints/fashion_mnist_fedavg_final.pt

PneumoniaMNIST:

- results/federated/pneumonia_mnist_fedavg.json
- checkpoints/pneumonia_mnist_fedavg_final.pt

## Findings

Both datasets demonstrated improving validation
accuracy and decreasing validation loss over three
federated rounds.

Fashion-MNIST validation accuracy increased from
82.55% to 87.03%.

PneumoniaMNIST validation accuracy increased from
74.24% to 93.70%.

The experiments demonstrate successful multi-round
federated training using sample-weighted FedAvg.

## Limitations

- Clients currently run sequentially on one machine.
- Only IID partitioning and FedAvg are supported.
- These results use validation data, not test data.
- Results do not yet establish performance relative
  to centralized training.

## Next Step

Day 10: Integrate the federated experiment runner
with the FastAPI backend.
