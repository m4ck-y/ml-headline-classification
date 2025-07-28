from typing import Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import spmatrix
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

def clean_text(text: str) -> str:
    # Reemplaza saltos de línea y retornos de carro
    txt = text.replace("\n", " ").replace("\r", " ")
    # Elimina caracteres que no sean alfanuméricos, espacios o '&'
    txt = "".join(char for char in txt if char.isalnum() or char.isspace() or char == '&')
    # Elimina espacios en blanco sobrantes
    return txt.strip()


# Regex para detectar fechas en varios formatos comunes (mes+días, años, días con sufijos)
date_pattern = re.compile(
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
    r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
    r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:[-–]\d{1,2})?\b'
    r'|\b(19|20)\d{2}\b'
    r'|\b\d{1,2}(st|nd|rd|th)\b',
    flags=re.IGNORECASE
)

# Función para detectar si hay fechas en un texto (devuelve True o False)
def has_date(text) -> bool:
    return bool(date_pattern.search(text))

# Función para eliminar fechas del texto
def remove_dates(text) -> str:
    text = date_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text)  # limpiar espacios extras
    return text.strip()


def tokenize_text(txt) -> str:
    # tokenizar en palabras
    tokens = word_tokenize(txt)

    # solo tokens alfabéticos
    tokens = [token for token in tokens if token.isalpha()]

    # eliminar stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens) # porque TfidfVectorizer entiende str no list

def vectorize_texts(texts: pd.Series) -> Tuple[spmatrix, TfidfVectorizer]:
    #Vectoriza los textos utilizando TF-IDF.
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer