from fastapi import APIRouter, HTTPException
from app.schemas import HeadlineRequest, PredictionResponse
#from app.core.loader import model, vectorizer, label_encoder
from app.preprocessing.process import process_headline
from app.preprocessing.process_text import tokenize_text
from app.core.registry import ModelRegistry

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_category(request: HeadlineRequest):

    model = ModelRegistry.get_model()
    vectorizer = ModelRegistry.get_vectorizer()
    label_encoder = ModelRegistry.get_label_encoder()

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
    
@router.get("/categories")
async def get_available_categories():
    label_encoder = ModelRegistry.get_label_encoder()

    if label_encoder is None:
        raise HTTPException(status_code=500, detail="LabelEncoder no cargado")

    categories = label_encoder.classes_
    encoded_values = label_encoder.transform(categories)

    category_map = {
        category: int(value)  # aseguramos que sea un int
        for category, value in zip(categories, encoded_values)
    }

    return {"categories": category_map}