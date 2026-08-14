"""
evaluate.py

Shared evaluation utilities used by train.py (and reusable standalone for
re-evaluating a saved model): metric computation, confusion matrix
plotting, and model comparison plotting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute accuracy, precision, recall, and F1-score for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_path: Path) -> None:
    """Plot and save a confusion matrix heatmap for the given model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(comparison_df, save_path: Path) -> None:
    """Plot a grouped bar chart comparing accuracy/precision/recall/F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(comparison_df))
    width = 0.2

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric.replace("_", " ").title())

    plt.xticks(x + width * 1.5, comparison_df["model_name"], rotation=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
