"""
predict.py

Loads the saved TF-IDF vectorizer and best trained model, and exposes
functions to predict sentiment for a single review or a batch of reviews.
Used by app.py — training is never re-run inside the Streamlit app.
"""

from pathlib import Path

import joblib
import pandas as pd

from data_preprocessing import clean_text, ensure_nltk_resources, get_stopwords

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

_model = None
_vectorizer = None
_stop_words = None


def load_artifacts():
    """Load the trained model and vectorizer once, caching them in memory."""
    global _model, _vectorizer, _stop_words

    if _model is None or _vectorizer is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Model or vectorizer not found in models/. "
                "Run src/train.py first to train and save them."
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

    if _stop_words is None:
        ensure_nltk_resources()
        _stop_words = get_stopwords()

    return _model, _vectorizer, _stop_words


def predict_sentiment(review_text: str) -> dict:
    """
    Predict sentiment for a single raw review string.

    Returns a dict with:
        sentiment: "Positive" or "Negative"
        confidence: float in [0, 1], or None if the model doesn't support
                    probability/decision-based confidence
        word_count, char_count: basic stats on the raw input text
    """
    model, vectorizer, stop_words = load_artifacts()

    cleaned = clean_text(review_text, stop_words)
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[prediction])
    elif hasattr(model, "decision_function"):
        # LinearSVC has no predict_proba; convert the decision score to a
        # 0-1 pseudo-confidence via a sigmoid for display purposes.
        import numpy as np

        score = model.decision_function(X)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "word_count": len(review_text.split()),
        "char_count": len(review_text),
    }


def predict_batch(reviews: pd.Series) -> pd.DataFrame:
    """
    Predict sentiment for a batch (pandas Series) of raw review strings.
    Returns a DataFrame with review, predicted_sentiment, and confidence.
    """
    model, vectorizer, stop_words = load_artifacts()

    cleaned = reviews.astype(str).apply(lambda t: clean_text(t, stop_words))
    X = vectorizer.transform(cleaned)

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidences = [proba[i][pred] for i, pred in enumerate(predictions)]
    elif hasattr(model, "decision_function"):
        import numpy as np

        scores = model.decision_function(X)
        confidences = [1 / (1 + np.exp(-abs(s))) for s in scores]
    else:
        confidences = [None] * len(predictions)

    return pd.DataFrame({
        "review": reviews.values,
        "predicted_sentiment": ["Positive" if p == 1 else "Negative" for p in predictions],
        "confidence": confidences,
    })
