from contextlib import asynccontextmanager
from fastapi import FastAPI
#from app.core.loader import load_trained
from app.core.registry import ModelRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting...")
    #load_trained()
    ModelRegistry.load()
    yield
    print("App is shutting down...")