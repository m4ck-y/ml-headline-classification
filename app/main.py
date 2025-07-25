from app.preprocessing.process import (
    load_data,
    minimize_df,
    process_df,
    vectorize_texts,
    apply_oversampling,
    split_data,
)

# Proceso de datos

df = load_data("data/data.json")

df = minimize_df(df)

df.to_csv("data/tmp_minimized.csv", index=False)

df = process_df(df)

X, vectorizer = vectorize_texts(df['headline'])

y = df['category']

X_resampled, y_resampled = apply_oversampling(X, y)

X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

# Entrenamiento de modelos

from app.models.models import (
    train_random_forest,
    train_logistic_regression,
    evaluate_model,
    save_model,
)

model_rf = train_random_forest(X_train, y_train)
model_lr = train_logistic_regression(X_train, y_train)

evaluate_model(model_rf, X_test, y_test, model_name="Random Forest")
evaluate_model(model_lr, X_test, y_test, model_name="Logistic Regression")


save_model(model_rf, vectorizer, model_name="random_forest")