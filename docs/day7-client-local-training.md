
# Day 7 — Federated Client Local Training

## Objective

Implement independent local CNN training for federated
learning clients using their assigned IID training data.

## Implementation

Main module: `federated/client/trainer.py`

The ClientTrainer:

1. Receives a global CNN model.
2. Creates an independent copy of the model.
3. Loads the client's assigned training dataset.
4. Trains using Adam and cross-entropy loss.
5. Extracts updated model weights.
6. Returns a validated ClientUpdate object.

## ClientUpdate Contract

Each client returns:

- Client ID
- Federated round number
- Number of local training samples
- Updated model state dictionary
- Final-epoch training loss
- Final-epoch training accuracy
- Local training duration

## Model Isolation

Each client uses `copy.deepcopy(global_model)`.

This ensures local optimization does not modify the
original global model or another client's model.

## Verification

Both mandatory datasets were tested with five clients.

Configuration:

- Clients: 5
- Smoke-test samples per client: 64
- Batch size: 16
- Local epochs: 1
- Learning rate: 0.001
- Device: CPU

### Fashion-MNIST

| Client | Samples | Loss | Accuracy |
|--------|---------|------|----------|
| 1 | 64 | 2.2895 | 15.62% |
| 2 | 64 | 2.2897 | 15.62% |
| 3 | 64 | 2.2880 | 14.06% |
| 4 | 64 | 2.3170 | 14.06% |
| 5 | 64 | 2.2324 | 18.75% |

### PneumoniaMNIST

| Client | Samples | Loss | Accuracy |
|--------|---------|------|----------|
| 1 | 64 | 0.6065 | 67.19% |
| 2 | 64 | 0.6519 | 56.25% |
| 3 | 64 | 0.6533 | 59.38% |
| 4 | 64 | 0.6762 | 54.69% |
| 5 | 64 | 0.6716 | 54.69% |

## Testing

Commands:

    python -m pytest tests/test_client_trainer.py -v
    python -m pytest tests/ -q
    python -m federated.client.verify_training

Results:

- 6 client-training tests passed.
- 80 total project tests passed.
- 1 non-blocking warning.
- All 10 real-data client training runs succeeded.
- Global model weights remained unchanged.

## Limitations

The verification uses 64 samples per client for speed.
It is not a full-dataset accuracy benchmark.

Client training currently runs sequentially in a local
simulation. Network-connected clients will be implemented
in a later phase.

## Next Step

Day 8: Implement sample-weighted Federated Averaging
(FedAvg) to aggregate multiple ClientUpdate objects
into an updated global model.
