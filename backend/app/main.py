
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.datasets import router as datasets_router
from backend.app.api.experiments import router as experiments_router
from backend.app.db.database import Base, engine
from backend.app.db import models


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="FedLearn API",
    description="Federated Learning Research Platform",
    version="0.1.0",
    lifespan=lifespan,
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
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "fedlearn-backend",
    }
