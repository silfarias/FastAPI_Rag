from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import RAGService

router = APIRouter()

# instanciamos el servicio de RAG
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str

@router.post("/api/ask")
async def ask_question(request: QueryRequest):
    try:
        # obtiene la respuesta de la IA
        answer = rag_service.ask_question(request.question)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta: {str(e)}")