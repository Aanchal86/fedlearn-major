
from fastapi import FastAPI

app = FastAPI(
    title="FedLearn API",
    description="Federated Learning Research Platform",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to FedLearn",
        "version": "0.1.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "fedlearn-backend"
    }
