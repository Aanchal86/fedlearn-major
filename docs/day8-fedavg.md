
# Day 8 — Federated Averaging (FedAvg)

## Objective

Implement sample-weighted Federated Averaging to combine
independently trained client CNN models into an updated
global model.

## Implementation

Main module: `federated/aggregation/fedavg.py`

Function: `federated_averaging(updates)`

The aggregator:

1. Receives a list of ClientUpdate objects.
2. Validates client metadata and model tensors.
3. Rejects duplicate clients and mismatched rounds.
4. Checks parameter names, shapes, and data types.
5. Calculates sample-weighted averages.
6. Returns a new aggregated model state dictionary.

## FedAvg Formula

W_global = sum((n_k / N) * W_k)

Where:

- n_k = number of samples belonging to client k
- N = total samples across participating clients
- W_k = trained model weights of client k

Clients with more training samples receive proportionally
greater aggregation weights.

## Numerical Precision

Floating-point tensors are accumulated using float64
precision and converted back to their original dtype.

Non-floating tensors must match across clients.

The aggregation process does not modify the original
ClientUpdate objects.

## Automated Testing

Test file: `tests/test_fedavg.py`

Six tests cover:

- Correct sample-weighted averaging
- Equal-size client averaging
- Preservation of original client updates
- Invalid and duplicate client updates
- Incompatible model parameters
- Loading aggregated weights into the CNN

Results:

- 86 project tests passed
- 1 non-blocking warning

## Integration Verification

Script: `federated/aggregation/verify_fedavg.py`

The script integrates:

- Day 6 IID partitioning
- Day 7 ClientTrainer
- Day 8 FedAvg aggregation

Configuration:

- 5 clients per dataset
- 64 smoke-test samples per client
- Batch size: 16
- Local epochs: 1
- Learning rate: 0.001
- CPU training

Supported datasets:

- Fashion-MNIST
- PneumoniaMNIST

Verification confirms:

- Clients train independently.
- The initial global model remains unchanged during
  local client training.
- Five client updates are successfully aggregated.
- Sample-weighted averaging produces the expected values.
- Aggregated weights load into the global CNN.

## Limitations

This is a single-round integration test using small
training subsets.

The test verifies aggregation correctness but does not
measure global model accuracy improvements.

## Next Step

Day 9: Implement multi-round federated learning,
evaluate the global model after each round, and track
accuracy and loss across rounds.
