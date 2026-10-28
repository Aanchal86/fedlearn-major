
# Day 4 — Backend API and Experiment Dashboard

## Objective

Develop REST APIs and a React dashboard for configuring,
saving, and viewing federated learning experiments.

## Backend Implementation

Framework: FastAPI
Database: SQLite
ORM: SQLAlchemy
Validation: Pydantic

### Dataset Endpoints

- GET /api/datasets
- GET /api/datasets/{dataset_name}

These endpoints provide metadata for Fashion-MNIST
and PneumoniaMNIST.

### Experiment Endpoints

- POST /api/experiments
- GET /api/experiments
- GET /api/experiments/{experiment_id}

Experiment configurations are validated and
stored in SQLite.

## Frontend Implementation

Framework: React + Vite

Implemented features:

- Dataset selection
- Experiment name
- Number of clients
- Training rounds
- Local epochs
- Batch size
- Learning rate
- FedAvg / Adaptive aggregation selection
- IID / Non-IID partition selection
- Dirichlet alpha configuration
- Saved experiment history
- API error and success messages

## Integration

The frontend communicates with FastAPI using
a reusable JavaScript API service.

CORS permits requests from the local Vite
development server.

## Testing

Backend:
- Dataset API tests
- Experiment schema tests
- Experiment API tests
- Existing database and dataset pipeline tests

Frontend:
- Six API service tests using Vitest
- Production build verification

## Current Limitations

Creating an experiment saves its configuration
but does not start federated training.

FedAvg and adaptive aggregation will be
implemented in later development phases.

## Outcome

FedLearn now provides an integrated interface
for configuring and managing experiments.
