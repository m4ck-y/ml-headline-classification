from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from app.preprocessing.process_text import tokenize_text
from app.preprocessing.process import process_headline
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = FastAPI(title="Clasificador de Headlines", description="API para clasificar headlines en categorías")

# Modelos globales
model = None
vectorizer = None
label_encoder = None

class HeadlineRequest(BaseModel):
    headline: str

class PredictionResponse(BaseModel):
    headline: str
    predicted_category: str
    confidence: float
    all_probabilities: dict

def load_models():
    """Carga los modelos entrenados"""
    global model, vectorizer, label_encoder
    
    try:
        # Cargar el modelo Random Forest
        if os.path.exists("models/random_forest.pkl"):
            model = joblib.load("models/random_forest.pkl")
            print("✅ Modelo Random Forest cargado exitosamente")
        else:
            raise FileNotFoundError("No se encontró el modelo random_forest.pkl")
        
        # Cargar el vectorizador
        if os.path.exists("models/vectorizer.pkl"):
            vectorizer = joblib.load("models/vectorizer.pkl")
            print("✅ Vectorizador cargado exitosamente")
        else:
            raise FileNotFoundError("No se encontró el vectorizer.pkl")
        
        # Crear el label encoder basado en las clases del modelo
        # Las clases están en el orden que el modelo las aprendió
        label_encoder = LabelEncoder()
        # Necesitamos recrear las clases originales
        # Por ahora usaremos las clases del modelo directamente
        classes = model.classes_
        label_encoder.classes_ = classes
        print(f"✅ Label encoder configurado con {len(classes)} clases")
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Carga los modelos al iniciar la aplicación"""
    load_models()

@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "API de Clasificación de Headlines", 
        "status": "activo",
        "modelo_cargado": model is not None,
        "vectorizador_cargado": vectorizer is not None
    }

@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": model is not None,
            "vectorizer": vectorizer is not None,
            "label_encoder": label_encoder is not None
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_category(request: HeadlineRequest):
    """Predice la categoría de un headline"""
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Modelos no cargados correctamente")
    
    try:
        # Procesar el headline igual que en el entrenamiento
        processed_headline = process_headline(request.headline)
        tokenized_headline = tokenize_text(processed_headline)
        
        # Vectorizar el texto
        headline_vector = vectorizer.transform([tokenized_headline])
        
        # Hacer la predicción
        prediction = model.predict(headline_vector)[0]
        
        # Obtener las probabilidades para todas las clases
        probabilities = model.predict_proba(headline_vector)[0]
        
        # Crear diccionario con todas las probabilidades
        all_probs = {}
        for i, prob in enumerate(probabilities):
            class_name = str(model.classes_[i])  # Convertir a string por si acaso
            all_probs[class_name] = float(prob)
        
        # Obtener la confianza de la predicción
        confidence = float(max(probabilities))
        
        # Convertir la predicción a string
        predicted_category = str(prediction)
        
        return PredictionResponse(
            headline=request.headline,
            predicted_category=predicted_category,
            confidence=confidence,
            all_probabilities=all_probs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Obtiene todas las categorías disponibles"""
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    return {
        "categories": [str(cls) for cls in model.classes_],
        "total_categories": len(model.classes_)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)