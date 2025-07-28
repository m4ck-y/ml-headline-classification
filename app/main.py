from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.lifecycle import lifespan
from app.api.routes import router



app = FastAPI(title="Clasificador de Headlines", description="API para clasificar headlines en categorías", lifespan=lifespan)

app.include_router(router)