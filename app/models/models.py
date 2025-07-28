import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import time
import joblib

from app.config.env import PATH_MODEL, PATH_VECTORIZER
from app.models.utils import log_evaluation, log_training

VERBOSE_RANDOM_FOREST_CLASSIFIER = 2 # Muestra progreso
VERBOSE_LOGISTIC_REGRESION = 1 # Muestra progreso

@log_training("RandomForestClassifier (balanced)")
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=VERBOSE_RANDOM_FOREST_CLASSIFIER
    ).fit(X_train, y_train)

@log_training("RandomForestClassifier (resampled)")
def train_random_forest_with_resampled(X_train, y_train) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=VERBOSE_RANDOM_FOREST_CLASSIFIER
    ).fit(X_train, y_train)

@log_training("LogisticRegression")
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=VERBOSE_LOGISTIC_REGRESION
    ).fit(X_train, y_train)


@log_evaluation()
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print("\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def save_model(model, vectorizer):
    joblib.dump(model, PATH_MODEL)
    joblib.dump(vectorizer, PATH_VECTORIZER)