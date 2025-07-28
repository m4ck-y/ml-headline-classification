import time
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
#typing
from scipy.sparse import spmatrix
from typing import Tuple

def apply_oversampling(X: spmatrix, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Balancea las clases utilizando oversampling

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(X, y, test_size=0.2):
    #Divide los datos en conjuntos de entrenamiento y prueba.

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def log_training(name):
    def decorator(func):
        def wrapper(X_train, y_train):
            print(f"\n>>> Entrenando: {name}")
            start = time.time()
            model = func(X_train, y_train)  # Llama a la función original
            end = time.time()

            print(f"\n-- Tiempo de entrenamiento '{name}': {end - start:.2f} segundos")
            print(f"Dimensión de X_train: {X_train.shape}")
            print(f"Dimensión de y_train: {y_train.shape}")
            return model
        return wrapper
    return decorator

def log_evaluation():
    def decorator(func):
        def wrapper(model, X_test, y_test, model_name):
            print(f"\n>>> Evaluando modelo: {model_name}")
            print(f"Dimensión de X_test: {X_test.shape}")
            print(f"Dimensión de y_test: {y_test.shape}")

            start = time.time()
            result = func(model, X_test, y_test, model_name)
            end = time.time()

            print(f"\n-- Tiempo de evaluación '{model_name}': {end - start:.2f} segundos\n")
            return result
        return wrapper
    return decorator