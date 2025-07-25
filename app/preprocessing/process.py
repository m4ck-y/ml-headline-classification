import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab') #palabras comunes como "the", "is", "and"
nltk.download('stopwords')

#typing
from scipy.sparse import spmatrix
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo JSON.
    """
    df = pd.read_json(file_path)
    return df

def minimize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el DataFrame a las 10 categorías menos frecuentes 
    """

    # Contar cuántas veces aparece cada categoría
    category_counts = df['category'].value_counts()

    # Seleccionar las 10 categorías con menor cantidad de ejemplos
    rarest_categories = category_counts.nsmallest(10).index.tolist()

    # Filtrar el DataFrame para conservar solo esas categorías
    minimized_df = df[df['category'].isin(rarest_categories)].reset_index(drop=True)
    return minimized_df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame para eliminar filas con valores nulos en 'text' o 'category'.
    """
    # Eliminar filas con valores nulos en 'headline' o 'category'
    df = df.dropna(subset=['headline', 'category'])

    # Limpiar el texto
    df['headline'] = df['headline'].apply(process_text)

    return df


def process_text(txt: str) -> str:
    "Procesa el texto para eliminar caracteres especiales y convertir a minúsculas"
    txt = txt.lower()
    txt = txt.replace("\n", " ").replace("\r", " ")
    txt = "".join(char for char in txt if char.isalnum() or char.isspace())

    # tokenizar en palabras
    tokens = word_tokenize(txt)

    # solo tokens alfabéticos (por si quedó algo raro)
    tokens = [token for token in tokens if token.isalpha()]

    # eliminar stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)

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