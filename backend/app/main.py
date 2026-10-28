
from fastapi import FastAPI
from backend.app.api.datasets import router as datasets_router
from backend.app.api.experiments import router as experiments_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="FedLearn API",
    description="Federated Learning Research Platform",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(datasets_router)
app.include_router(experiments_router)


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

