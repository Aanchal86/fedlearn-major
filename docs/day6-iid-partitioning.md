
# Day 6 — IID Client Dataset Partitioning

## Objective

Implement deterministic IID partitioning of training datasets
across five simulated federated learning clients.

## Datasets

| Dataset | Training | Validation | Test |
|---------|----------|------------|------|
| Fashion-MNIST | 54,000 | 6,000 | 10,000 |
| PneumoniaMNIST | 4,708 | 524 | 624 |

Only training data is partitioned. Validation and test data
remain centralized.

## Implementation

- Module: `federated/partitioning/iid.py`
- Algorithm: seeded random permutation followed by equal-sized splitting
- Number of clients: 5
- Random seed: 42
- Client datasets represented using PyTorch `Subset`
- Class distributions computed using `get_label_distribution()`

## Client Sample Counts

| Client | Fashion-MNIST | PneumoniaMNIST |
|--------|---------------|---------------|
| 1 | 10,800 | 942 |
| 2 | 10,800 | 942 |
| 3 | 10,800 | 942 |
| 4 | 10,800 | 941 |
| 5 | 10,800 | 941 |

## Validation

Verified:

1. Every training sample is assigned exactly once.
2. No duplicate training indices exist across clients.
3. All training indices are covered.
4. Partitioning is reproducible using the same seed.
5. Different seeds produce different assignments.
6. Invalid client counts are rejected.
7. Client label distributions are available.
8. Only training datasets are partitioned.

## Testing

Run:

```bash
python -m pytest tests/test_iid_partitioning.py -v
python -m pytest tests/ -q
python -m federated.partitioning.verify_iid
```

Results:

- IID partitioning tests: 6 passed
- Complete test suite: 74 passed, 1 warning
- Real-dataset verification: passed for both datasets

## Next Step

Day 7 will implement client-side local training using the
partitioned datasets, with local model updates returned to
the federated server for aggregation.
