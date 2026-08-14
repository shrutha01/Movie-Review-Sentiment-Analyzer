"""
data_preprocessing.py

Handles loading the raw IMDb Movie Reviews dataset and applying NLP
preprocessing: HTML removal, lowercasing, punctuation cleanup, stopword
handling (with sentiment-bearing negations preserved), tokenization, and
whitespace normalization.

Run directly to produce data/processed/cleaned_reviews.csv from
data/raw/IMDB Dataset.csv:

    python src/data_preprocessing.py
"""

import re
import sys
from pathlib import Path

import nltk
import pandas as pd

# --------------------------------------------------------------------------
# Paths (relative to project root, resolved with pathlib so this works
# regardless of the machine or working directory it's run from)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def get_stopwords() -> set:
    """
    Return an English stopword set with sentiment-bearing negation words
    removed, so phrases like "not good" or "wasn't great" keep their
    polarity-flipping words instead of being stripped to "good"/"great".
    """
    from nltk.corpus import stopwords

    base_stopwords = set(stopwords.words("english"))
    negation_words = {
        "no", "not", "nor", "don", "don't", "didn", "didn't", "doesn",
        "doesn't", "isn", "isn't", "wasn", "wasn't", "aren", "aren't",
        "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
        "couldn", "couldn't", "shouldn", "shouldn't", "hasn", "hasn't",
        "haven", "haven't", "hadn", "hadn't", "can't", "cannot",
        "mustn", "mustn't", "needn", "needn't", "ain", "very", "against",
    }
    return base_stopwords - negation_words


def remove_html_tags(text: str) -> str:
    """Strip HTML tags (e.g. <br /> which is common in IMDb reviews)."""
    return re.sub(r"<.*?>", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove punctuation/special characters while keeping letters, digits,
    and single spaces. Contractions are expanded to "not" beforehand by
    the caller so meaning-bearing negations survive this step.
    """
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def expand_negation_contractions(text: str) -> str:
    """Normalize common negation contractions before punctuation removal."""
    contractions = {
        r"\bwon't\b": "will not",
        r"\bcan't\b": "can not",
        r"\bn't\b": " not",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_text(text: str, stop_words: set) -> str:
    """
    Apply the full cleaning pipeline to a single review:
    HTML removal -> lowercase -> negation expansion -> punctuation removal
    -> tokenize -> stopword filtering (negations kept) -> rejoin.
    """
    from nltk.tokenize import word_tokenize

    if not isinstance(text, str):
        return ""

    text = remove_html_tags(text)
    text = text.lower()
    text = expand_negation_contractions(text)
    text = remove_special_characters(text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]

    return " ".join(tokens)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw IMDb dataset CSV. Expects 'review' and 'sentiment' columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download the IMDb Dataset of 50K Movie Reviews from Kaggle "
            "(lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place "
            "'IMDB Dataset.csv' inside data/raw/."
        )
    df = pd.read_csv(path)

    expected_cols = {"review", "sentiment"}
    if not expected_cols.issubset(set(df.columns.str.lower())):
        raise ValueError(
            f"Expected columns {expected_cols} in dataset, found {list(df.columns)}"
        )
    # Normalize column names in case of casing differences
    df.columns = [c.lower() for c in df.columns]
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on a raw dataframe:
    missing value handling, deduplication, text cleaning, and
    label encoding (positive=1, negative=0).
    """
    ensure_nltk_resources()
    stop_words = get_stopwords()

    df = df.copy()

    # Handle missing values
    before = len(df)
    df = df.dropna(subset=["review", "sentiment"])
    print(f"Dropped {before - len(df)} rows with missing values.")

    # Remove duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset=["review"])
    print(f"Dropped {before - len(df)} duplicate reviews.")

    # Clean review text
    print("Cleaning review text (this may take a few minutes for 50k reviews)...")
    df["cleaned_review"] = df["review"].apply(lambda t: clean_text(t, stop_words))

    # Drop any reviews that became empty after cleaning
    before = len(df)
    df = df[df["cleaned_review"].str.strip().str.len() > 0]
    print(f"Dropped {before - len(df)} reviews that were empty after cleaning.")

    # Encode sentiment label
    df["label"] = df["sentiment"].str.lower().map({"positive": 1, "negative": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Keep useful derived columns for EDA
    df["review_length"] = df["review"].str.len()
    df["word_count"] = df["cleaned_review"].str.split().apply(len)

    return df[["review", "cleaned_review", "sentiment", "label", "review_length", "word_count"]]


def main() -> None:
    """Load raw data, preprocess it, and save the cleaned dataset."""
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = load_raw_data()
    print(f"Loaded {len(df)} raw reviews.")

    df_clean = preprocess_dataframe(df)
    print(f"Final cleaned dataset size: {len(df_clean)} reviews.")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
