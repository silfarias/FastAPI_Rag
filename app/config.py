import os

# configuramos rutas y otros parámetros
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Opción 1: Usar r para evitar secuencias de escape
PDF_PATH = os.path.normpath(os.path.join(BASE_DIR, r"../data/Learning-TensorFlow.pdf"))
print("LA URL DEL ARCHIVO ES: ", PDF_PATH)

CHROMA_DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "../chroma_db_dir"))