"""
train.py

Trains and compares three models (Logistic Regression, Multinomial Naive
Bayes, Linear SVM) on TF-IDF features from the cleaned IMDb reviews, then
saves the best-performing model (by F1-score) and the fitted TF-IDF
vectorizer to models/.

Run directly:

    python src/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from evaluate import evaluate_model, plot_confusion_matrix, plot_model_comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_FILENAME = "sentiment_model.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
COMPARISON_FILENAME = "model_comparison.csv"
BEST_MODEL_INFO_FILENAME = "best_model_info.json"


def load_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found. Run src/data_preprocessing.py first."
        )
    return pd.read_csv(PROCESSED_DATA_PATH)


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF with unigrams + bigrams. min_df/max_df filter out extremely
    rare and extremely common terms to reduce noise and dimensionality.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
    )


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, C=1.0, max_iter=5000),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned dataset...")
    df = load_data()
    df = df.dropna(subset=["cleaned_review"])
    X_text = df["cleaned_review"].astype(str)
    y = df["label"]

    print(f"Dataset size: {len(df)} reviews "
          f"({(y == 1).sum()} positive, {(y == 0).sum()} negative)")

    # Stratified train/test split, fixed random state for reproducibility
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train_text)} | Test size: {len(X_test_text)}")

    # Fit TF-IDF ONLY on training data to prevent data leakage
    print("Fitting TF-IDF vectorizer on training data...")
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # Train and evaluate each model
    results = []
    trained_models = {}

    for name, model in get_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)
        print(
            f"{name} -> Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1-score: {metrics['f1_score']:.4f}"
        )

    # Build comparison table
    comparison_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    comparison_path = REPORTS_DIR / COMPARISON_FILENAME
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_path}")
    print(comparison_df.to_string(index=False))

    # Select best model based primarily on F1-score
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model_name"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected (by F1-score): {best_model_name}")

    # Plot comparison chart and confusion matrix for the best model
    plot_model_comparison(comparison_df, FIGURES_DIR / "model_comparison.png")
    plot_confusion_matrix(
        best_model, X_test, y_test, best_model_name,
        FIGURES_DIR / "confusion_matrix.png"
    )

    # Persist best model + vectorizer (never retrain inside the Streamlit app)
    model_path = MODELS_DIR / MODEL_FILENAME
    vectorizer_path = MODELS_DIR / VECTORIZER_FILENAME
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved best model to: {model_path}")
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")

    # Save metadata about the best model for the Streamlit app to display
    best_model_info = {
        "model_name": best_model_name,
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1_score": float(best_row["f1_score"]),
        "train_size": len(X_train_text),
        "test_size": len(X_test_text),
    }
    with open(MODELS_DIR / BEST_MODEL_INFO_FILENAME, "w") as f:
        json.dump(best_model_info, f, indent=2)
    print(f"Saved best model info to: {MODELS_DIR / BEST_MODEL_INFO_FILENAME}")


if __name__ == "__main__":
    main()
