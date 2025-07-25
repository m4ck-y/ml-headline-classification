import pandas as pd
from app.preprocessing.process import (
    load_data,
    process_df,
    clean_df,
    vectorize_texts,
    apply_oversampling,
    split_data,
)

df = load_data("data/data.json")

df = process_df(df)

X, vectorizer = vectorize_texts(df['headline'])

y = df['category']

X_resampled, y_resampled = apply_oversampling(X, y)

X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)