
# FedLearn

**Privacy-Preserving Distributed Federated Learning Platform**

FedLearn is a final-year B.Tech Information Technology major project focused on collaborative machine learning without centrally collecting raw client training data.

The platform combines a custom PyTorch federated learning engine, FastAPI backend, React dashboard, and reproducible experimental evaluation.

## Key Features

- Custom federated learning implementation using PyTorch
- Federated Averaging (FedAvg) and adaptive aggregation
- IID and Dirichlet-based non-IID data partitioning
- Multi-round federated training and evaluation
- Independent network-connected training clients
- Distributed aggregation and fault-tolerance mechanisms
- Experiment monitoring through a React dashboard
- Multi-seed statistical analysis and comparison visualizations
- Automated backend and frontend tests

## Technology Stack

| Component | Technology |
|---|---|
| Machine Learning | Python 3.11, PyTorch |
| Backend | FastAPI, SQLAlchemy |
| Frontend | React, Vite |
| Data Analysis | Pandas, Matplotlib |
| Testing | Pytest, Vitest |
| Containerization | Docker Compose |

## Project Structure

```text
FedLearn/
├── backend/          # FastAPI application
├── frontend/         # React dashboard
├── federated/        # Federated learning engine
├── experiments/      # Experiment runners and analysis
├── tests/            # Automated backend tests
├── docs/             # Technical documentation
├── results/          # Generated experimental results
└── README.md
```

## Supported Datasets

### Fashion-MNIST

- 10-class clothing image classification
- Primary dataset for federated learning experiments
- Evaluated using IID and non-IID partitions

### PneumoniaMNIST

- Binary chest X-ray image classification
- Supported by the dataset preprocessing pipeline
- Included as a secondary dataset in the experimental design

Both datasets use shared preprocessing and DataLoader interfaces.

## Federated Learning Workflow

1. The coordinator initializes the global machine learning model.
2. Training data is partitioned among independent clients.
3. Clients receive the current global model.
4. Each client performs local training using its own data.
5. Clients return model updates instead of raw training samples.
6. The coordinator aggregates updates using FedAvg or adaptive aggregation.
7. The global model is evaluated before the next communication round.

FedLearn supports local simulations and network-connected client execution.

**Privacy note:** Federated learning avoids centrally collecting raw client training data, but does not automatically guarantee differential privacy or protection against information leakage from model updates.

## Local Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Git
- Docker (optional)

### Backend Setup (macOS/Linux)

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

For Windows PowerShell, activate the virtual environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

### Application URLs

| Service | URL |
|---|---|
| Backend | http://localhost:8000 |
| Frontend | http://localhost:5173 |
| API Documentation | http://localhost:8000/docs |

## Docker Setup

From the project root:

```bash
docker compose up --build
```

## Experimental Design

The primary Fashion-MNIST evaluation uses:

| Parameter | Configuration |
|---|---|
| Number of clients | 5 |
| Communication rounds | 10 |
| Local epochs | 1 |
| Aggregation methods | FedAvg, Adaptive |
| Data distributions | IID, Dirichlet alpha=0.5, alpha=0.1 |
| Random seeds | 42, 123, 2026 |
| Total multi-seed experiments | 18 |

### Evaluation Metrics

- Classification accuracy
- Macro F1 score
- Validation loss
- Communication-round convergence
- Mean and standard deviation across random seeds
- Paired aggregation-method differences

## Experimental Results

### Fashion-MNIST Multi-Seed Evaluation

Results represent the mean and sample standard deviation across three random seeds.

| Distribution | Aggregation | Accuracy (%) | Macro F1 |
|---|---|---:|---:|
| IID | FedAvg | 90.84 ± 0.20 | 0.9085 |
| IID | Adaptive | 90.91 ± 0.13 | 0.9092 |
| Dirichlet alpha=0.5 | FedAvg | 88.66 ± 0.67 | 0.8870 |
| Dirichlet alpha=0.5 | Adaptive | 88.65 ± 0.43 | 0.8870 |
| Dirichlet alpha=0.1 | FedAvg | 76.08 ± 8.28 | 0.7195 |
| Dirichlet alpha=0.1 | Adaptive | 75.99 ± 8.43 | 0.7178 |

### Key Research Findings

1. Both aggregation methods achieved approximately 91% mean accuracy under IID partitioning.
2. Moderate non-IID partitioning reduced model accuracy compared with IID training.
3. Strong non-IID partitioning produced lower accuracy and greater variation across random seeds.
4. Adaptive aggregation did not consistently outperform FedAvg.
5. The results are specific to the tested datasets, configurations, and random seeds.

## Reproducibility

The repository includes experiment runners and analysis scripts in the `experiments/` directory.

To regenerate analysis outputs from existing experimental results:

```bash
python -m experiments.analyze_multiseed
python -m experiments.plot_comparisons
python -m experiments.plot_convergence
```

The scripts require their corresponding experimental result CSV files.

### Generated Results

```text
results/
├── comparisons/
│   ├── fashion_mnist_comparison_summary.csv
│   └── figures/
└── multiseed/
    ├── multiseed_summary.csv
    ├── statistical_summary.csv
    ├── paired_differences.csv
    ├── paired_summary.csv
    └── figures/
```

Generated visualizations include final-metric comparisons, convergence curves, and multi-seed statistical comparisons.

## Testing

### Backend Tests

```bash
python -m pytest tests/ -q
```

### Frontend Tests

```bash
cd frontend
npm test -- --run
npm run build
```

## Documentation

- [System Architecture](docs/architecture.md)
- [Dataset Pipeline](docs/dataset-pipeline.md)
- [Experimental Design](docs/experiment-design.md)
- [Earlier Evaluation Results](docs/day15-evaluation-results.md)

## Limitations

- Experiments use a limited number of clients, communication rounds, and random seeds.
- The reported multi-seed evaluation focuses on Fashion-MNIST.
- Federated learning alone does not provide formal differential privacy or secure aggregation guarantees.
- Network conditions and client availability may influence distributed execution.

## Project Status

The core federated learning engine, distributed execution workflow, dashboard, and Fashion-MNIST experimental evaluation have been implemented.

Final documentation, integration verification, and release preparation are in progress.
