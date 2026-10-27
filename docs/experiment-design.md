
# FedLearn — Experimental Design

## Primary Dataset
Fashion-MNIST

- Training samples: 60,000
- Test samples: 10,000
- Image dimensions: 1 x 28 x 28
- Number of classes: 10

Training data will be divided into client-specific
partitions. Test data will remain separate.

## Model Architecture
Small Convolutional Neural Network

1. Conv2D(1, 16, kernel_size=3, padding=1)
2. ReLU
3. MaxPool2D(2)
4. Conv2D(16, 32, kernel_size=3, padding=1)
5. ReLU
6. MaxPool2D(2)
7. Flatten
8. Linear(1568, 128)
9. ReLU
10. Linear(128, 10)

## Initial Hyperparameters

- Clients: 5
- Communication rounds: 10
- Local epochs: 1
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Seed: 42
- Device: CPU

## Data Partitioning

### IID
Training samples are randomly distributed
across clients with approximately equal sizes.

### Non-IID
Dirichlet label-skew partitioning will be used.

Planned alpha values:
- 0.5: Moderate label heterogeneity
- 0.1: Strong label heterogeneity

## Aggregation Methods

1. Sample-weighted FedAvg
2. Reliability-aware adaptive aggregation

The adaptive aggregation formula will be
finalized on Day 12.

## Evaluation Metrics

- Global accuracy
- Macro F1 score
- Global loss
- Convergence across communication rounds
- Client participation
- Training duration
- Communication cost

## Reproducibility

- Fixed random seeds
- Consistent model architecture
- Identical data partitions for paired comparisons
- Consistent hyperparameters
- Separate test evaluation
- Multiple-seed experiments during final evaluation

## Research Safeguards

The test set must not be used to compute
adaptive client aggregation weights.

The initial implementation uses simulated
clients. Independent network clients will
be implemented during Days 16-20.
