import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from app.config import PDF_PATH, CHROMA_DB_DIR

class RAGService: # servicio principal para manejar RAG
    
    def __init__(self): # constructor de la clase
        
        
        # verificamos si el archivo especificado en PDF_PATH existe y tiene contenido
        if not os.path.isfile(PDF_PATH):
            raise FileNotFoundError(f"El archivo PDF no se encontró en la ruta especificada: {PDF_PATH}")
        if os.path.getsize(PDF_PATH) == 0:
            raise ValueError("El archivo PDF está vacío.")
        
        # inicializamos el modelo de lenguaje de Ollama
        self.llm = ChatOllama(model="llama3")
        
        # intentamos cargar el docuemento PDF con PyMuPDFLoader
        try:
            loader = PyMuPDFLoader(PDF_PATH)
            documents = loader.load()
        except Exception as e:
            raise ValueError(f"Error al cargar el archivo PDF: {e}")
        
        # verificamos si se cargaron documentos validos
        if not documents:
            raise ValueError("No se encontraron documentos en el archivo especificado.")
        
        # dividimos los documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        docs = text_splitter.split_documents(documents)
        
        # si no se pudo dividir los documentos, levantamos un error
        if not docs:
            raise ValueError("Error en la división de los documentos. Verifica el contenido de PDF_PATH.")
        
        # embeddings y vector store
        embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # para generar embeddings de texto usamos el modelo all-MiniLM-L6-v2
        
        # creamos una base de datos de vectores con Chroma donde guardaremos los embeddings
        self.vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=CHROMA_DB_DIR, # ubicación de la base de datos
            collection_name="learning_tensorflow_data"
        )
        
        # configuración la instrucción que se enviará al modelo de lenguaje para generar una respuesta
        custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
        Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

        Contexto: {context}
        Pregunta: {question}

        Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español.
        Respuesta útil:
        """
        
        # 
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
        
        # configuramos el recuperador de documentos a partir de la base de datos Chroma, 
        # limitando los resultados a los 5 documentos más relevantes.
        retriever = self.vs.as_retriever(search_kwargs={'k': 5})
        
        # creamos un objeto RetrievalQA con el modelo de lenguaje, el retriever y la instrucción 
        # (cadena de preguntas y respuestas)
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    # método para obtener una respuesta a una pregunta
    def ask_question(self, question: str) -> str:
        # procesa la pregunta y devuelve la respuesta
        response = self.qa({"query": question})
        return response["result"]