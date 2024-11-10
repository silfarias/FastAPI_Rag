from fastapi import FastAPI
from app.routers import ask

# instanciamos la app con FastAPI
app = FastAPI()

# incluimos los rutas
app.include_router(ask.router)

# ruta de inicio
@app.get("/")
def index():
    return {"message": "Bienvenido al sistema RAG con FastAPI"}