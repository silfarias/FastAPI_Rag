from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import RAGService

# creamos instancia de enrutador
router = APIRouter()

# instanciamos el servicio RAG
rag_service = RAGService()

# definimos la estructura de la solicitud que la API espera recibir
class QueryRequest(BaseModel):
    question: str # pregunta del usuario de tipo string

# ruta post donde enviamos la pregunta y esperamos recibir la respuesta de la IA
@router.post("/api/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = rag_service.ask_question(request.question) # llamamos al metodo del servicio RAG
        return {"response": answer} # retornamos un json con la respuesta
    except Exception as e: # si algo falla levantamos un error de servidor
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta: {str(e)}")