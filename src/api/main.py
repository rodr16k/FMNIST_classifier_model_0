from fastapi import FastAPI
from src.api.routes import predict

app = FastAPI(title='FashionMNIST CNN API')
app.include_router(predict.router)


@app.get("/health")
def health():
    return {"status": "ok"}
