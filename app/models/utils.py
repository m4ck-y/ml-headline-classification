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