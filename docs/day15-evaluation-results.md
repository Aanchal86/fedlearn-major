# FedLearn — Evaluation and Experimental Results

## 1. Experimental Objective

The objective of this evaluation was to compare standard Federated Averaging (FedAvg) against a sample-size and training-accuracy-based Adaptive Aggregation method under different client data distributions.

Experiments examined convergence behavior, data heterogeneity, noisy-client robustness, and sensitivity to random seeds.

## 2. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Datasets | Fashion-MNIST, PneumoniaMNIST |
| Model | Small Convolutional Neural Network |
| Framework | PyTorch |
| Federated clients | 5 simulated clients |
| Communication rounds | 10 |
| Local epochs per round | 1 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Data partitions | IID and Dirichlet label-skew |
| Dirichlet α | 0.5 and 0.1 |
| Aggregation methods | FedAvg and Adaptive |
| Random seeds | 42, 123, 2026 |
| Evaluation | Held-out validation datasets |

The total evaluation consisted of 36 federated experiments across two datasets, six experimental configurations, and three random seeds.

All experiments used the same configuration within each paired FedAvg–Adaptive comparison.

## 3. Aggregation Methods

### 3.1 Federated Averaging

FedAvg aggregates client parameters in proportion to each client's training dataset size.

\[
w_i = \frac{n_i}{\sum_j n_j}
\]

Here, \(n_i\) represents the number of local training samples on client \(i\).

### 3.2 Adaptive Aggregation

The implemented adaptive method additionally incorporates the locally reported training accuracy:

\[
w_i = \frac{n_i \max(a_i,\epsilon)}
{\sum_j n_j \max(a_j,\epsilon)}
\]

Here, \(a_i\) is the local training accuracy and \(\epsilon=10^{-6}\) prevents zero raw weights.

This approach was designed to dynamically adjust client contributions, but local accuracy is not necessarily a reliable measure of global generalization under heterogeneous datasets.

## 4. Multi-Seed Experimental Results

Final validation accuracy is reported as mean ± sample standard deviation across three random seeds.

| Dataset | Distribution | FedAvg accuracy | Adaptive accuracy |
|---|---|---|---|
| Fashion-MNIST | IID | 90.93 ± 0.23% | 90.76 ± 0.14% |
| Fashion-MNIST | Dirichlet α = 0.5 | 88.83 ± 0.50% | 88.62 ± 0.55% |
| Fashion-MNIST | Dirichlet α = 0.1 | 76.06 ± 8.49% | 76.42 ± 8.44% |
| PneumoniaMNIST | IID | 95.10 ± 0.40% | 95.23 ± 0.33% |
| PneumoniaMNIST | Dirichlet α = 0.5 | 95.23 ± 0.57% | 95.17 ± 0.58% |
| PneumoniaMNIST | Dirichlet α = 0.1 | 74.24 ± 0.00% | 74.24 ± 0.00% |

## 5. Robustness Experiments

A controlled experiment simulated one unreliable Fashion-MNIST client by corrupting either 30% or 60% of its training labels.

The corrupted client was Client 3, and all other clients retained clean training data.

| Noise rate | FedAvg accuracy | Adaptive accuracy |
|---|---|---|
| 0% | 88.32% | 88.08% |
| 30% | 87.58% | 87.28% |
| 60% | 87.45% | 87.75% |

At 60% label noise, adaptive aggregation reduced Client 3's final contribution from 19.95% to 8.59%.

This demonstrates sensitivity to local training performance, although the resulting global accuracy improvement was small and observed for one random seed.

## 6. Error Analysis

Under the Dirichlet α = 0.1 condition, PneumoniaMNIST exhibited severe performance degradation.

The inspected Adaptive model produced the validation confusion matrix:

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 0 | 135 |
| Class 1 | 0 | 389 |

All validation samples were predicted as Class 1.

Although the model achieved approximately 74.24% validation accuracy, its Macro F1 was only 0.4261 and Class 0 recall was zero.

This demonstrates the importance of using class-sensitive metrics rather than accuracy alone.

## 7. Main Findings

1. Both aggregation methods performed similarly under IID data.
2. Moderate Non-IID data caused limited performance degradation.
3. Severe Non-IID data significantly reduced performance and increased seed sensitivity.
4. Adaptive aggregation did not consistently outperform FedAvg.
5. Accuracy-based weighting reduced the influence of a noisy client but did not guarantee better global generalization.
6. PneumoniaMNIST under extreme label skew demonstrated a majority-class prediction failure.

## 8. Limitations and Future Work

The experiments used simulated federated clients on a single machine rather than physically distributed devices.

Only three random seeds were evaluated, limiting statistical power.

The adaptive method relied on local training accuracy, which can be misleading for highly imbalanced clients.

Model-update privacy was not guaranteed by the federated architecture alone. Differential privacy, secure aggregation, client-level validation, class-aware weighting, larger seed sweeps, and stronger robustness strategies remain future improvements.

The experiments reported validation performance. Final held-out test evaluation should be conducted only after model selection and experimental settings are finalized.

## 9. Conclusion

FedLearn successfully implemented federated training with IID and Non-IID partitioning, standard FedAvg, an adaptive aggregation baseline, and controlled noisy-client experiments.

The evaluation demonstrates that client heterogeneity and class imbalance are critical challenges in federated learning. Adaptive aggregation can modify client influence based on locally reported performance, but this mechanism does not consistently improve global accuracy.

The project provides a reproducible experimental framework for further investigation of reliable and privacy-preserving federated learning.