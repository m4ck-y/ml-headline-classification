import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import time
import joblib

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\nEntrenando el modelo... sklearn RandomForestClassifier")
    start = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    end = time.time()
    print(f"Tiempo de entrenamiento: {end - start:.2f} segundos")
    return model


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("\nEntrenando el modelo... sklearn LogisticRegression")
    start = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    end = time.time()
    print(f"Tiempo de entrenamiento: {end - start:.2f} segundos")
    return model


def evaluate_model(model, X_test, y_test, model_name="Modelo"):

    print(f"\nEvaluación del Modelo: {model_name}")
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(f"Tiempo de evaluación: {end - start:.2f} segundos")


def save_model(model, vectorizer, model_name="Modelo", models_path="models/"):
    vec_path = models_path
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(model, f"{models_path}{model_name}.pkl")
    joblib.dump(vectorizer, f"{vec_path}/vectorizer.pkl")