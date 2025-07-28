from app.models.utils import apply_oversampling, split_data
from app.preprocessing.process import load_data,process_df
from app.preprocessing.process_dataframe import minimize_df
from app.preprocessing.process_text import tokenize_text, vectorize_texts
from sklearn.preprocessing import LabelEncoder
import os
import joblib

model = None
modelo_path = "models/model.pkl"

def load_model():
    if os.path.exists(modelo_path):
        with open(modelo_path, 'rb') as f:
            print("Cargando modelo...")
            global model
            model = joblib.load(modelo_path)
    else:
        print("Creando el modelo...")
        create_model()



def create_model():

    # Proceso de datos

    df = load_data("data/data.json")

    #df = minimize_df(df) # Comentar esta lineas para probar todos los datos

    df = process_df(df)
    df.to_csv("data/data_processed.csv", index=False)

    df["headline"] = df["headline"].apply(tokenize_text)

    X, vectorizer = vectorize_texts(df['headline'])

    le = LabelEncoder()
    y_encoded = le.fit_transform(df['category'])

    X_resampled, y_resampled = apply_oversampling(X, y_encoded)

    # Dividir en entrenamiento y prueba con oversampling
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = split_data(X_resampled, y_resampled)
    y_test_labels_resampled = le.inverse_transform(y_test_resampled)

    # Entrenamiento de modelos

    from app.models.models import (
        train_random_forest_with_resampled,
        evaluate_model,
        save_model,
    )

    model_rf = train_random_forest_with_resampled(X_train_resampled, y_train_resampled)

    evaluate_model(model_rf, X_test_resampled, y_test_resampled, model_name="Random Forest con Oversampling")

    save_model(model_rf, vectorizer)
    global model
    model = model_rf

load_model()