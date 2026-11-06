from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="VanguardAI Gateway")
app.include_router(router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VanguardAI Gateway"}
