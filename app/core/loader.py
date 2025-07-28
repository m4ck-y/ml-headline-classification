from app.config.env import PATH_LABEL_ENCODER, PATH_MODEL, PATH_VECTORIZER
from app.models.utils import apply_oversampling, split_data
from app.preprocessing.process import load_data,process_df
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
    df = minimize_df(df, 30) # Comentar esta lineas para probar todos los datos
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