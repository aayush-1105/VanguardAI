from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(title="VanguardAI Gateway", lifespan=lifespan)
app.include_router(router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VanguardAI Gateway"}
