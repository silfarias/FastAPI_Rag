import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from app.config import PDF_PATH, CHROMA_DB_DIR

class RAGService:
    
    def __init__(self):
        # verifica si el archivo existe y tiene contenido
        if not os.path.isfile(PDF_PATH):
            raise FileNotFoundError(f"El archivo PDF no se encontró en la ruta especificada: {PDF_PATH}")
        if os.path.getsize(PDF_PATH) == 0:
            raise ValueError("El archivo PDF está vacío.")
        
        # inicializa el modelo y carga el documento una única vez
        self.llm = ChatOllama(model="llama3")
        
        # intenta cargar el PDF con PyMuPDFLoader
        try:
            loader = PyMuPDFLoader(PDF_PATH)
            documents = loader.load()
        except Exception as e:
            raise ValueError(f"Error al cargar el archivo PDF: {e}")
        
        if not documents:
            raise ValueError("No se encontraron documentos en el archivo especificado.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            raise ValueError("Error en la división de los documentos. Verifica el contenido de PDF_PATH.")
        
        # embeddings y vector store
        embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=CHROMA_DB_DIR,
            collection_name="learning_tensorflow_data"
        )
        
        # configuración del prompt
        custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español.
        Respuesta útil:
        """
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
        
        # configuración de QA
        retriever = self.vs.as_retriever(search_kwargs={'k': 5})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask_question(self, question: str) -> str:
        # procesa la pregunta y devuelve la respuesta
        response = self.qa({"query": question})
        return response["result"]