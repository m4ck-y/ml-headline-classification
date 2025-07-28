import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from app.preprocessing.process_dataframe import normalize_category, remove_duplicated, remove_empty_values
from app.preprocessing.process_text import clean_text, remove_dates

nltk.download('punkt_tab')
nltk.download('stopwords')

#typing
from scipy.sparse import spmatrix
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_json(file_path)
    return df

def process_df(df: pd.DataFrame) -> pd.DataFrame:

    df_cleaned = df.copy()

    df_cleaned["headline"] = df_cleaned["headline"].apply(process_headline)
    df_cleaned["category"] = df_cleaned["category"].apply(process_category)
    
    df_cleaned = remove_empty_values(df_cleaned)
    df_cleaned = remove_duplicated(df_cleaned)
    df_cleaned = normalize_category(df_cleaned)


    return df_cleaned

def process_category(txt: str) -> str:
    txt = txt.upper()
    txt = clean_text(txt)
    return txt


def process_headline(txt: str) -> str:
    txt = txt.lower()
    txt = remove_dates(txt)
    txt = clean_text(txt)
    return txt
    

def process_text(txt: str) -> str:
    
    txt = clean_text(txt)

    # tokenizar en palabras
    tokens = word_tokenize(txt)

    # solo tokens alfabéticos (por si quedó algo raro)
    tokens = [token for token in tokens if token.isalpha()]

    # eliminar stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)
