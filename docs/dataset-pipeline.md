
# FedLearn Dataset Pipeline

## Overview

FedLearn supports two image classification datasets:

1. Fashion-MNIST — general-purpose benchmark.
2. PneumoniaMNIST — healthcare image classification.

Both datasets use a common PyTorch data pipeline.

## Dataset Specifications

| Property | Fashion-MNIST | PneumoniaMNIST |
|---|---|---|
| Image size | 28x28 | 28x28 |
| Channels | 1 | 1 |
| Classes | 10 | 2 |
| Training | 54,000 | 4,708 |
| Validation | 6,000 | 524 |
| Testing | 10,000 | 624 |

Fashion-MNIST uses a seeded 90/10 split of its
official training data.

PneumoniaMNIST preserves its official train,
validation, and test splits.

## Preprocessing

Both datasets use:

- Conversion to float32 PyTorch tensors
- Pixel scaling to [0, 1]
- Dataset-specific normalization
- Standardized integer classification labels

Normalization settings:

- Fashion-MNIST: mean=0.286, std=0.353
- PneumoniaMNIST: mean=0.5, std=0.5

PneumoniaMNIST uses fixed scaling constants,
not estimated dataset statistics.

## DataLoaders

- Default batch size: 64
- Random seed: 42
- Training shuffle: enabled
- Validation/test shuffle: disabled
- DataLoader workers: 0

Output batch shapes:

- Images: [batch_size, 1, 28, 28]
- Labels: [batch_size]

## Reproducibility

Fashion-MNIST splitting uses a seeded PyTorch
generator.

Training DataLoader shuffling uses a separate
seeded generator.

## Data Leakage Prevention

- Official test datasets remain separate.
- Validation data is excluded from local training.
- Test results must not influence adaptive
  aggregation weights or hyperparameter selection.
- Healthcare evaluation must account for the
  dataset's class imbalance and any limitations
  in patient-level independence.

## Federated Learning Integration

The current pipeline produces centralized
training, validation, and test datasets.

On Day 6, the training dataset will be divided
into client-specific partitions.

Both IID and non-IID partitioning will be
supported.

Validation and test data will remain outside
the client training partitions.
