from fastapi import FastAPI
from app.routers import ask

app = FastAPI()

app.include_router(ask.router)

@app.get("/")
def index():
    return {"message": "Bienvenido al sistema RAG con FastAPI"}