"""
eda.py

Exploratory Data Analysis on the cleaned IMDb reviews dataset. Generates
and saves visualizations to reports/figures/:

    - sentiment_distribution.png
    - review_length_distribution.png
    - most_common_words.png
    - positive_words.png
    - negative_words.png
    - wordcloud_positive.png
    - wordcloud_negative.png

Run directly:

    python src/eda.py
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid")
PALETTE = {"Positive": "#2E7D32", "Negative": "#C62828"}


def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found. Run src/data_preprocessing.py first."
        )
    return pd.read_csv(PROCESSED_DATA_PATH)


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["sentiment"].str.capitalize().value_counts()
    plt.figure(figsize=(6, 5))
    bars = plt.bar(counts.index, counts.values,
                    color=[PALETTE.get(c, "#607D8B") for c in counts.index])
    plt.title("Positive vs Negative Review Distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Reviews")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 200, f"{height:,}",
                  ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()


def plot_review_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for sentiment, color in [("positive", PALETTE["Positive"]), ("negative", PALETTE["Negative"])]:
        subset = df[df["sentiment"] == sentiment]["review_length"]
        sns.kdeplot(subset, label=sentiment.capitalize(), fill=True, alpha=0.3, color=color)
    plt.title("Review Length Distribution (characters)", fontsize=14, fontweight="bold")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_length_distribution.png", dpi=150)
    plt.close()


def _top_words(text_series: pd.Series, n: int = 20) -> list:
    words = " ".join(text_series.dropna()).split()
    return Counter(words).most_common(n)


def plot_most_common_words(df: pd.DataFrame) -> None:
    top_words = _top_words(df["cleaned_review"], n=20)
    words, counts = zip(*top_words)
    plt.figure(figsize=(8, 8))
    sns.barplot(x=list(counts), y=list(words), color="#455A64")
    plt.title("Most Common Words (All Reviews)", fontsize=14, fontweight="bold")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "most_common_words.png", dpi=150)
    plt.close()


def plot_class_words(df: pd.DataFrame, sentiment: str, color: str, filename: str) -> None:
    subset = df[df["sentiment"] == sentiment]["cleaned_review"]
    top_words = _top_words(subset, n=20)
    words, counts = zip(*top_words)
    plt.figure(figsize=(8, 8))
    sns.barplot(x=list(counts), y=list(words), color=color)
    plt.title(f"Most Common Words — {sentiment.capitalize()} Reviews",
              fontsize=14, fontweight="bold")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def plot_wordcloud(df: pd.DataFrame, sentiment: str, filename: str) -> None:
    text = " ".join(df[df["sentiment"] == sentiment]["cleaned_review"].dropna())
    wc = WordCloud(
        width=1000, height=600, background_color="white",
        colormap="Greens" if sentiment == "positive" else "Reds",
        max_words=150,
    ).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {sentiment.capitalize()} Reviews", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    print(f"Loaded {len(df)} cleaned reviews for EDA.")

    print("Plotting sentiment distribution...")
    plot_sentiment_distribution(df)

    print("Plotting review length distribution...")
    plot_review_length_distribution(df)

    print("Plotting most common words...")
    plot_most_common_words(df)

    print("Plotting positive review words...")
    plot_class_words(df, "positive", "#2E7D32", "positive_words.png")

    print("Plotting negative review words...")
    plot_class_words(df, "negative", "#C62828", "negative_words.png")

    print("Generating positive word cloud...")
    plot_wordcloud(df, "positive", "wordcloud_positive.png")

    print("Generating negative word cloud...")
    plot_wordcloud(df, "negative", "wordcloud_negative.png")

    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
