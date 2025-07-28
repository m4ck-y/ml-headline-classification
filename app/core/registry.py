import joblib
import os
from app.config.env import PATH_MODEL, PATH_VECTORIZER, PATH_LABEL_ENCODER
from app.models.utils import apply_oversampling, split_data
from app.preprocessing.process import load_data, process_df
from app.preprocessing.process_dataframe import minimize_df
from app.preprocessing.process_text import tokenize_text, vectorize_texts
from sklearn.preprocessing import LabelEncoder


class ModelRegistry:
    model = None
    vectorizer = None
    label_encoder = None

    @classmethod
    def load(cls):
        if all(os.path.isfile(p) for p in [PATH_MODEL, PATH_VECTORIZER, PATH_LABEL_ENCODER]):
            print("Cargando modelo, vectorizador y codificador...")
            cls.model = joblib.load(PATH_MODEL)
            cls.vectorizer = joblib.load(PATH_VECTORIZER)
            cls.label_encoder = joblib.load(PATH_LABEL_ENCODER)
        else:
            print("Archivos no encontrados. Entrenando modelo desde cero...")
            cls.train_and_save()

    @classmethod
    def train_and_save(cls):
        df = load_data("data/data.json")
        df = minimize_df(df, n=30)  # Para entrenamiento rápido
        df = process_df(df)
        df.to_csv("data/data_processed.csv", index=False)
        df["headline"] = df["headline"].apply(tokenize_text)

        X, cls.vectorizer = vectorize_texts(df['headline'])

        cls.label_encoder = LabelEncoder()
        y_encoded = cls.label_encoder.fit_transform(df['category'])
        joblib.dump(cls.label_encoder, PATH_LABEL_ENCODER)

        X_resampled, y_resampled = apply_oversampling(X, y_encoded)
        X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

        from app.models.models import train_random_forest_with_resampled, evaluate_model, save_model
        cls.model = train_random_forest_with_resampled(X_train, y_train)
        evaluate_model(cls.model, X_test, y_test, model_name="Random Forest con Oversampling")

        save_model(cls.model, cls.vectorizer)

    @classmethod
    def get_model(cls):
        return cls.model

    @classmethod
    def get_vectorizer(cls):
        return cls.vectorizer

    @classmethod
    def get_label_encoder(cls):
        return cls.label_encoder
