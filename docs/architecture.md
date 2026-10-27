
# FedLearn — System Architecture

## 1. Project Overview

FedLearn is a federated learning research platform
developed using PyTorch, FastAPI, React, and Docker.

Its objective is to support collaborative model
training without centrally collecting raw client
training datasets.

The platform supports simulated federated learning
and will later support independent network-connected
clients.

## 2. Main Components

### React Dashboard

The dashboard allows users to:

- Configure federated learning experiments
- Start and monitor training
- View global accuracy, loss, and macro F1
- Compare FedAvg and adaptive aggregation
- Review experiment history

### FastAPI Backend

The backend is responsible for:

- Receiving experiment configurations
- Validating API requests
- Managing experiment lifecycle
- Providing experiment results
- Communicating with the database

### Federated Coordinator

The coordinator manages:

- Global model initialization
- Client selection
- Model distribution
- Local training execution
- Client update collection
- Global model aggregation
- Round evaluation

### Federated Clients

Each client:

- Maintains its own local training dataset
- Receives the current global model
- Trains using PyTorch
- Returns model parameters and sample count
- Does not upload raw training examples

### Aggregation Engine

FedLearn will implement:

1. Federated Averaging (FedAvg)
2. Reliability-aware adaptive aggregation

FedAvg uses sample-count-weighted aggregation.

Adaptive aggregation will incorporate client
reliability information using validation metrics.

The adaptive formula will be finalized on Day 12.

## 3. Training Workflow

1. User creates an experiment in React.
2. FastAPI validates the configuration.
3. Coordinator initializes the global model.
4. Participating clients receive model parameters.
5. Each client trains on local data.
6. Clients return updated model parameters.
7. The aggregation engine computes global weights.
8. The global model is evaluated.
9. Round metrics are saved.
10. The next training round begins.

## 4. Development Stages

### Stage 1: Simulated Federated Learning

Days 6-15

- Five simulated clients
- Single-machine execution
- IID and non-IID partitions
- FedAvg and adaptive aggregation
- Reproducible experiments

### Stage 2: Network-Connected Clients

Days 16-20

- Independent client processes
- HTTP-based model exchange
- Client registration
- Round orchestration
- Timeouts and dropout handling
- Two-laptop demonstration

## 5. Technology Stack

- Python 3.11
- PyTorch
- FastAPI
- React + Vite
- SQLAlchemy
- Docker Compose
- SQLite for initial development

## 6. Research Evaluation

Planned evaluation metrics:

- Accuracy
- Macro F1
- Training loss
- Convergence across rounds
- Client participation
- Training time
- Communication cost

Experiments will compare:

- Centralized training
- Standard FedAvg
- Adaptive aggregation
- IID and non-IID data distributions

## 7. Privacy and Limitations

Raw client training datasets remain local.

Federated learning does not automatically guarantee
privacy, because model updates can leak information.

Differential privacy and secure aggregation are
outside the initial project scope.

The test dataset must not influence adaptive
aggregation weights.


## 8. Database Architecture

FedLearn initially uses SQLite with SQLAlchemy ORM.

### Experiments Table

Stores:
- Experiment identifier
- Experiment name
- Training configuration
- Experiment status
- Creation timestamp

### Training Rounds Table

Stores:
- Experiment reference
- Communication round number
- Global accuracy
- Global loss
- Global macro F1
- Round execution time

### Client Participation Table

Stores:
- Training round reference
- Client identifier
- Local training sample count
- Aggregation weight
- Local training loss
- Local training accuracy

### Relationships

One experiment can contain multiple training rounds.

Each training round can contain multiple
client participation records.

Foreign keys maintain relationships between
the tables.

### Future Integration

The database will be connected to experiment
management APIs and the federated coordinator
during later development stages.
