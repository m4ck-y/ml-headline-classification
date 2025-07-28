from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config.env import PATH_LABEL_ENCODER, PATH_MODEL, PATH_VECTORIZER
from app.models.utils import apply_oversampling, split_data
from app.preprocessing.process import load_data,process_df, process_headline
from app.preprocessing.process_dataframe import minimize_df
from app.preprocessing.process_text import tokenize_text, vectorize_texts
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# globales
model = None
vectorizer = None
label_encoder = None

def load_trained():
    global model, vectorizer, label_encoder

    # Verificar que todos los archivos existan
    if all(os.path.isfile(p) for p in [PATH_MODEL, PATH_VECTORIZER, PATH_LABEL_ENCODER]):
        print("Cargando modelo, vectorizador y codificador...")
        model = joblib.load(PATH_MODEL)
        vectorizer = joblib.load(PATH_VECTORIZER)
        label_encoder = joblib.load(PATH_LABEL_ENCODER)
    else:
        print("Alguno de los archivos no existe. Creando el modelo...")
        create_model()



def create_model():
    global vectorizer, label_encoder, model

     # 1. Cargar y preprocesar datos
    df = load_data("data/data.json")
    df = minimize_df(df) # Comentar esta lineas para probar todos los datos
    df = process_df(df)
    df.to_csv("data/data_processed.csv", index=False)
    df["headline"] = df["headline"].apply(tokenize_text)

    # 2. Vectorización y codificación
    X, vectorizer = vectorize_texts(df['headline'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['category'])
    joblib.dump(label_encoder, PATH_LABEL_ENCODER)

    # 3. Oversampling y división
    X_resampled, y_resampled = apply_oversampling(X, y_encoded)

    # 4. Entrenamiento y evaluación
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = split_data(X_resampled, y_resampled)
    y_test_labels_resampled = label_encoder.inverse_transform(y_test_resampled)

    # Entrenamiento de modelos
    from app.models.models import (
        train_random_forest_with_resampled,
        evaluate_model,
        save_model,
    )
    model = train_random_forest_with_resampled(X_train_resampled, y_train_resampled)
    evaluate_model(model, X_test_resampled, y_test_resampled, model_name="Random Forest con Oversampling")

    # 5. Guardar modelo y vectorizador
    save_model(model, vectorizer)

app = FastAPI(title="Clasificador de Headlines", description="API para clasificar headlines en categorías")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting...")
    load_trained()
    yield
    print("App is shutting down...")

app = FastAPI(lifespan=lifespan)

class HeadlineRequest(BaseModel):
    headline: str

class PredictionResponse(BaseModel):
    headline: str
    predicted_category: str
    confidence: float
    all_probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict_category(request: HeadlineRequest):
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