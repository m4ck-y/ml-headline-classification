import os


MODEL_DIR = "models"       # Sin barra final, recomendado
os.makedirs(MODEL_DIR, exist_ok=True)  # Crea el directorio si no existe

PATH_MODEL = os.path.join(MODEL_DIR, "model.pkl")  # -> "models/model.pkl" o "models\model.pkl"
PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")
PATH_LABEL_ENCODER = os.path.join(MODEL_DIR, "label_encoder.pkl")
