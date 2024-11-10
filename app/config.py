import os

# configuramos rutas y otros parámetros

# obtenemos la ruta absoluta del archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# juntamos el directorio base con la ruta relativa
PDF_PATH = os.path.normpath(os.path.join(BASE_DIR, r"../data/Learning-TensorFlow.pdf")) # usamos r para evitar secuencias de escape

#print("LA URL DEL ARCHIVO ES: ", PDF_PATH)

# generamos la ruta del vector store
CHROMA_DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "../chroma_db_dir"))