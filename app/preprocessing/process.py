import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#typing
from scipy.sparse import spmatrix
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo JSON.
    """
    df = pd.read_json(file_path)
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame para eliminar filas con valores nulos en 'text' o 'category'.
    """
    # Eliminar filas con valores nulos en 'text' o 'category'
    df = df.dropna(subset=['text', 'category'])
    
    # Limpiar el texto
    df['headline'] = df['headline'].apply(process_text)

    return df


def process_text(txt: str) -> str:
    "Procesa el texto para eliminar caracteres especiales y convertir a minúsculas"
    txt = txt.lower()
    txt = txt.replace("\n", " ").replace("\r", " ")
    txt = "".join(char for char in txt if char.isalnum() or char.isspace())
    return txt.strip()

def vectorize_texts(texts: pd.Series) -> Tuple[spmatrix, TfidfVectorizer]:
    """
    Vectoriza los textos utilizando TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def apply_oversampling(X: spmatrix, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    
    "Balancea las clases utilizando oversampling"

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(X, y, test_size=0.3):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)