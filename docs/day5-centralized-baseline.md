
# Day 5 — CNN and Centralized Training Baselines

## Objective

Implement and evaluate a centralized CNN classification
baseline for Fashion-MNIST and PneumoniaMNIST.

These baselines will serve as reference points for
future federated learning experiments.

## CNN Architecture

Input: 1 x 28 x 28 grayscale image

1. Conv2D: 1 -> 16 channels, kernel 3, padding 1
2. ReLU
3. MaxPool2D: kernel 2
4. Conv2D: 16 -> 32 channels, kernel 3, padding 1
5. ReLU
6. MaxPool2D: kernel 2
7. Flatten: 1568 features
8. Fully connected: 1568 -> 128
9. ReLU
10. Output layer:
    - Fashion-MNIST: 10 classes
    - PneumoniaMNIST: 2 classes

Model parameters:
- Fashion-MNIST: 206,922
- PneumoniaMNIST: 205,890

## Training Configuration

- Framework: PyTorch
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Random seed: 42
- Default device: CPU
- Best checkpoint selection: Validation macro-F1

## Dataset Splits

Fashion-MNIST:
- Training: 54,000
- Validation: 6,000
- Test: 10,000

PneumoniaMNIST:
- Training: 4,708
- Validation: 524
- Test: 624

The test datasets are reserved for final evaluation.

## Evaluation Metrics

- Cross-entropy loss
- Classification accuracy
- Macro precision
- Macro recall
- Macro-F1
- Weighted-F1
- Per-class precision, recall and F1
- Confusion matrix

## Run Experiments

Fashion-MNIST:

python -m federated.training.run_baseline \
  --dataset fashion_mnist --epochs 5

PneumoniaMNIST:

python -m federated.training.run_baseline \
  --dataset pneumonia_mnist --epochs 5

Verify saved checkpoints:

python -m federated.training.verify_baselines

## Generated Artifacts

Checkpoints:
- checkpoints/fashion_mnist_centralized_best.pt
- checkpoints/pneumonia_mnist_centralized_best.pt

Metrics:
- results/baselines/fashion_mnist_centralized.json
- results/baselines/pneumonia_mnist_centralized.json
- results/baselines/centralized_comparison.csv

Generated artifacts are excluded from Git.

## Testing

Run:

python -m pytest tests/ -q

Tests cover:
- CNN output shapes
- Model parameter validation
- Gradient propagation
- Training loop correctness
- Parameter updates
- Evaluation metrics
- Confusion matrix structure
- Empty dataset handling
- Checkpoint verification through the CLI

## Future Integration

The CNN and training functions will be reused for:

1. Client-side local training
2. FedAvg aggregation
3. Adaptive aggregation
4. IID and non-IID experiments
5. Centralized versus federated comparisons

## Limitations

- Training is currently centralized.
- Federated client training is not yet implemented.
- The CNN is a lightweight reference architecture.
- The current results are validation results, not final
  test-set results.
- Model checkpoints and experiment results are
  stored locally.


## Experimental Results

The centralized SmallCNN model was trained on Fashion-MNIST and PneumoniaMNIST using the Adam optimizer with a learning rate of 0.001, batch size of 64, and 5 training epochs.

The best checkpoint for each dataset was selected based on validation macro-F1.

### Centralized Baseline Performance

| Dataset | Best Epoch | Validation Accuracy | Validation Macro-F1 |
|---|---|---|---|
| Fashion-MNIST | 5 | 91.23% | 91.24% |
| PneumoniaMNIST | 5 | 95.61% | 94.22% |

### Observations

- **Fashion-MNIST:** The model achieved 91.23% validation accuracy and 91.24% macro-F1, indicating relatively balanced classification performance across the ten clothing categories.
- **PneumoniaMNIST:** The model achieved 95.61% validation accuracy and 94.22% macro-F1. The difference between accuracy and macro-F1 suggests that class-wise performance should also be examined.
- Both models achieved their highest recorded validation macro-F1 at epoch 5.
- These results establish centralized baselines for subsequent comparisons with FedAvg and adaptive federated aggregation.

### Conclusion

The centralized CNN successfully learned classification patterns from both datasets. The saved checkpoints and evaluation metrics provide reference results for evaluating the effectiveness of federated learning under IID and non-IID client data distributions.

These results represent validation performance, not final test-set performance.
