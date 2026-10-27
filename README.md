
# FedLearn

Privacy-Preserving Distributed Federated Learning Platform

## Overview

FedLearn is a final-year B.Tech IT major project focused on
collaborative machine learning without centrally collecting
raw client training data.

## Technology Stack

- Python 3.11
- PyTorch
- FastAPI
- React + Vite
- SQLAlchemy
- Docker Compose

## Project Structure

- backend/ - FastAPI application
- frontend/ - React dashboard
- federated/ - Federated learning engine
- tests/ - Automated tests
- docs/ - Architecture and documentation

## Local Setup

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Docker Setup

From the project root:

```powershell
docker compose up --build
```

Backend: http://localhost:8000

Frontend: http://localhost:5173

API Docs: http://localhost:8000/docs

## Planned Features

- Custom PyTorch federated learning
- Federated Averaging (FedAvg)
- Adaptive aggregation
- IID and non-IID experiments
- Independent network-connected clients
- React experiment monitoring
- Reproducible model evaluation

## Supported Datasets

FedLearn currently supports:

- Fashion-MNIST: 10-class clothing classification.
- PneumoniaMNIST: binary chest X-ray classification.

Both datasets use a shared preprocessing and
DataLoader interface.

### Inspect Datasets

```powershell
python -m federated.data.inspect_dataset
python -m federated.data.inspect_medmnist
```

### Run Dataset Tests

```powershell
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_splits.py -v
python -m pytest tests/test_dataloaders.py -v
```

See `docs/dataset-pipeline.md` for dataset details.
