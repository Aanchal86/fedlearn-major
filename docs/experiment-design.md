
# FedLearn — Experimental Design

## 1. Research Objective

The objective of this study is to evaluate federated learning under different client data distributions and compare Federated Averaging (FedAvg) with sample-size- and training-accuracy-weighted adaptive aggregation.

The evaluation focuses on model accuracy, Macro F1, validation loss, convergence, and variability across random seeds.

## 2. Primary Dataset

Fashion-MNIST

- Training samples: 60,000
- Test samples: 10,000
- Image dimensions: 1 x 28 x 28
- Number of classes: 10

Training data is partitioned among federated clients. Evaluation data is kept separate from client training partitions.

PneumoniaMNIST is supported by the platform's dataset pipeline but is not included in the reported 18-run Fashion-MNIST multi-seed comparison.

## 3. Model Architecture

The experimental design specifies a small Convolutional Neural Network:

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

## 4. Training Configuration

| Parameter | Configuration |
|---|---|
| Clients | 5 |
| Communication rounds | 10 |
| Local epochs | 1 |
| Batch size | 64 (initial design) |
| Learning rate | 0.001 (initial design) |
| Optimizer | Adam (initial design) |
| Loss function | CrossEntropyLoss (initial design) |
| Execution device | CPU |
| Random seeds | 42, 123, 2026 |

The batch size, optimizer, and learning rate are documented initial settings and should be checked against the experiment runner before claiming they were used unchanged in every completed run.

## 5. Data Partitioning

### IID

Training samples are distributed approximately uniformly among clients.

### Non-IID

Dirichlet label-skew partitioning creates heterogeneous client label distributions.

Two concentration parameters are evaluated:

- Alpha = 0.5: Moderate label heterogeneity
- Alpha = 0.1: Strong label heterogeneity

Smaller alpha values generally produce more uneven class distributions among clients.

## 6. Aggregation Methods

### Federated Averaging

FedAvg weights client models according to their training sample counts.

For client k:

\[
w_k = \frac{n_k}{\sum_j n_j}
\]

### Adaptive Aggregation

FedLearn implements sample-size- and training-accuracy-weighted aggregation.

First, an unnormalized weight is calculated:

\[
s_k = n_k \max(a_k, \epsilon)
\]

Then:

\[
w_k = \frac{s_k}{\sum_j s_j}
\]

Where:

- n_k = client training sample count
- a_k = local training accuracy
- epsilon = 0.000001

The global model is formed by weighted aggregation of compatible floating-point model parameters.

The adaptive method uses local training accuracy rather than validation accuracy or network reliability.

## 7. Experimental Matrix

| Distribution | FedAvg | Adaptive | Seeds |
|---|---|---|---|
| IID | Yes | Yes | 42, 123, 2026 |
| Dirichlet alpha=0.5 | Yes | Yes | 42, 123, 2026 |
| Dirichlet alpha=0.1 | Yes | Yes | 42, 123, 2026 |

Total experimental configurations:

3 distributions × 2 aggregation methods × 3 random seeds = 18 experiments.

Each experiment uses five clients and ten communication rounds.

## 8. Evaluation Metrics

The completed Fashion-MNIST evaluation includes:

- Global classification accuracy
- Macro F1 score
- Validation loss
- Communication-round convergence
- Mean and sample standard deviation across seeds
- Paired accuracy and Macro F1 differences

The platform also records round execution information.

Communication cost is a research consideration, but a comprehensive communication-cost comparison is not established by the reported 18-run results.

## 9. Reproducibility

The experimental workflow uses:

- Explicit random seeds
- Consistent experimental configurations
- Matching seeds for paired aggregation comparisons
- Separate evaluation data
- Saved experiment summaries
- Saved round-level metrics
- Statistical analysis scripts

The analysis pipeline produces summary statistics and paired aggregation comparisons.

## 10. Completed Evaluation

The 18 Fashion-MNIST multi-seed experiments were completed using CPU execution.

### Mean Accuracy

| Distribution | FedAvg | Adaptive |
|---|---:|---:|
| IID | 90.84% | 90.91% |
| Dirichlet alpha=0.5 | 88.66% | 88.65% |
| Dirichlet alpha=0.1 | 76.08% | 75.99% |

The results demonstrate lower mean accuracy and greater seed-to-seed variability under strong non-IID partitioning.

Adaptive aggregation did not consistently outperform FedAvg.

Detailed statistical results, limitations, and generated visualizations are documented in `docs/final-evaluation-report.md`.

## 11. Research Safeguards

- Raw client training datasets are not transmitted as part of normal model-update exchange.
- Evaluation data must remain separate from local client training.
- Adaptive aggregation weights are computed from local training accuracy and sample counts, not test-set accuracy.
- Federated learning alone does not guarantee differential privacy.
- Experimental conclusions must reflect variability across seeds.
- Results must not be generalized beyond the evaluated configurations without additional experiments.

## 12. Execution Modes

FedLearn includes:

1. Simulated federated learning with multiple logical clients.
2. Network-connected federated learning with independent client processes.

Distributed orchestration, fault tolerance, and recovery mechanisms were developed during Days 16–20.

The primary 18-run Fashion-MNIST statistical evaluation was conducted using CPU-based experiments.

## 13. Limitations and Future Evaluation

Future experiments may investigate:

- More random seeds
- Larger client populations
- Additional communication rounds
- More model architectures and datasets
- Communication overhead and network latency
- Different client dropout rates
- Differential privacy and secure aggregation
- Alternative aggregation strategies
