from fastapi import FastAPI

app = FastAPI(title="VanguardAI Gateway")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VanguardAI Gateway"}
