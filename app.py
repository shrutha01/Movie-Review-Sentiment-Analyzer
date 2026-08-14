"""
app.py

Streamlit web application: Movie Review Sentiment Analyzer.

Loads the pre-trained model and TF-IDF vectorizer (never retrains at
startup) and provides:
    - Single review prediction with confidence score
    - CSV batch prediction with downloadable results
    - Model performance dashboard
    - Dataset statistics / EDA gallery

Run with:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"

sys.path.insert(0, str(SRC_DIR))

from predict import predict_batch, predict_sentiment  # noqa: E402

# --------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eee;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a1a2e;
    }
    .result-box-positive {
        background: linear-gradient(135deg, #e8f8ef 0%, #d4f4e2 100%);
        border-left: 6px solid #2E7D32;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-box-negative {
        background: linear-gradient(135deg, #fdecea 0%, #fbdada 100%);
        border-left: 6px solid #C62828;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-label-positive { color: #2E7D32; font-size: 1.8rem; font-weight: 800; }
    .result-label-negative { color: #C62828; font-size: 1.8rem; font-weight: 800; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_best_model_info() -> dict | None:
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison_table() -> pd.DataFrame | None:
    path = REPORTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_dataset_stats() -> pd.DataFrame | None:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return None


def artifacts_ready() -> bool:
    return (MODELS_DIR / "sentiment_model.pkl").exists() and \
        (MODELS_DIR / "tfidf_vectorizer.pkl").exists()


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🎬 Movie Review\n### Sentiment Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home / Prediction", "📂 Batch Prediction", "📊 Model Performance", "📈 Dataset Statistics"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

best_info = load_best_model_info()
if best_info:
    st.sidebar.markdown("**Active model**")
    st.sidebar.info(f"{best_info['model_name']}\n\nF1-score: {best_info['f1_score']:.3f}")
else:
    st.sidebar.warning("No trained model found yet. Run `python src/train.py` first.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, Scikit-learn & Streamlit")


# --------------------------------------------------------------------------
# PAGE: Home / Prediction
# --------------------------------------------------------------------------
if page == "🏠 Home / Prediction":
    st.markdown('<div class="main-header">🎬 Movie Review Sentiment Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter a movie review below and let the model '
                'predict whether it\'s Positive or Negative.</div>', unsafe_allow_html=True)

    if not artifacts_ready():
        st.error(
            "⚠️ No trained model found. Please run `python src/train.py` from the "
            "project root first, then restart this app."
        )
    else:
        example = "The movie was amazing and the acting was fantastic."
        review_text = st.text_area(
            "Your movie review",
            placeholder=f'e.g. "{example}"',
            height=160,
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

        if analyze_clicked:
            if not review_text or not review_text.strip():
                st.warning("Please enter a review before analyzing.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_sentiment(review_text)

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                box_class = "result-box-positive" if sentiment == "Positive" else "result-box-negative"
                label_class = "result-label-positive" if sentiment == "Positive" else "result-label-negative"
                emoji = "😊" if sentiment == "Positive" else "😞"

                confidence_str = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"

                st.markdown(
                    f"""
                    <div class="{box_class}">
                        <div class="{label_class}">{emoji} {sentiment.upper()}</div>
                        <p style="margin-top:0.5rem; font-size:1.1rem;">
                            Confidence: <strong>{confidence_str}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.markdown(
                        f'<div class="metric-card"><h3>Word Count</h3><p>{result["word_count"]}</p></div>',
                        unsafe_allow_html=True,
                    )
                with stat_col2:
                    st.markdown(
                        f'<div class="metric-card"><h3>Character Count</h3><p>{result["char_count"]}</p></div>',
                        unsafe_allow_html=True,
                    )
                with stat_col3:
                    st.markdown(
                        f'<div class="metric-card"><h3>Confidence</h3><p>{confidence_str}</p></div>',
                        unsafe_allow_html=True,
                    )


# --------------------------------------------------------------------------
# PAGE: Batch Prediction
# --------------------------------------------------------------------------
elif page == "📂 Batch Prediction":
    st.markdown('<div class="main-header">📂 CSV Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file with a <code>review</code> column '
                'to predict sentiment for every row.</div>', unsafe_allow_html=True)

    if not artifacts_ready():
        st.error(
            "⚠️ No trained model found. Please run `python src/train.py` from the "
            "project root first, then restart this app."
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")
                batch_df = None

            if batch_df is not None:
                if "review" not in batch_df.columns:
                    st.error("The uploaded CSV must contain a column named 'review'.")
                else:
                    st.success(f"Loaded {len(batch_df)} reviews.")
                    if st.button("🚀 Predict All", type="primary"):
                        with st.spinner(f"Predicting sentiment for {len(batch_df)} reviews..."):
                            results_df = predict_batch(batch_df["review"])

                        st.markdown("### Results")
                        st.dataframe(results_df, use_container_width=True)

                        pos_count = (results_df["predicted_sentiment"] == "Positive").sum()
                        neg_count = (results_df["predicted_sentiment"] == "Negative").sum()
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown(
                                f'<div class="metric-card"><h3>Positive</h3><p>{pos_count}</p></div>',
                                unsafe_allow_html=True,
                            )
                        with m2:
                            st.markdown(
                                f'<div class="metric-card"><h3>Negative</h3><p>{neg_count}</p></div>',
                                unsafe_allow_html=True,
                            )

                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Results as CSV",
                            data=csv_bytes,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                        )


# --------------------------------------------------------------------------
# PAGE: Model Performance
# --------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance</div>', unsafe_allow_html=True)

    comparison_df = load_comparison_table()
    if comparison_df is None or best_info is None:
        st.error(
            "⚠️ No evaluation results found. Please run `python src/train.py` first."
        )
    else:
        st.markdown('<div class="sub-header">Comparison of all trained models on the held-out '
                    'test set.</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, key) in zip(
            [c1, c2, c3, c4],
            [("Accuracy", "accuracy"), ("Precision", "precision"),
             ("Recall", "recall"), ("F1-Score", "f1_score")],
        ):
            with col:
                st.markdown(
                    f'<div class="metric-card"><h3>{label}</h3>'
                    f'<p>{best_info[key] * 100:.1f}%</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f"**Best model:** {best_info['model_name']} "
                    f"(trained on {best_info['train_size']} reviews, "
                    f"tested on {best_info['test_size']} reviews)")

        st.markdown("### Comparison Table")
        st.dataframe(comparison_df.style.format({
            "accuracy": "{:.4f}", "precision": "{:.4f}",
            "recall": "{:.4f}", "f1_score": "{:.4f}",
        }), use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            comparison_fig = FIGURES_DIR / "model_comparison.png"
            if comparison_fig.exists():
                st.image(str(comparison_fig), caption="Model Performance Comparison")
        with chart_col2:
            cm_fig = FIGURES_DIR / "confusion_matrix.png"
            if cm_fig.exists():
                st.image(str(cm_fig), caption=f"Confusion Matrix — {best_info['model_name']}")


# --------------------------------------------------------------------------
# PAGE: Dataset Statistics
# --------------------------------------------------------------------------
elif page == "📈 Dataset Statistics":
    st.markdown('<div class="main-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)

    df = load_dataset_stats()
    if df is None:
        st.error(
            "⚠️ No processed dataset found. Please run `python src/data_preprocessing.py` first."
        )
    else:
        st.markdown('<div class="sub-header">Exploratory analysis of the IMDb movie '
                    'reviews dataset.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Total Reviews</h3><p>{len(df):,}</p></div>',
                        unsafe_allow_html=True)
        with c2:
            pos = (df["label"] == 1).sum()
            st.markdown(f'<div class="metric-card"><h3>Positive</h3><p>{pos:,}</p></div>',
                        unsafe_allow_html=True)
        with c3:
            neg = (df["label"] == 0).sum()
            st.markdown(f'<div class="metric-card"><h3>Negative</h3><p>{neg:,}</p></div>',
                        unsafe_allow_html=True)

        st.markdown("### Visualizations")
        figures = [
            ("sentiment_distribution.png", "Positive vs Negative Distribution"),
            ("review_length_distribution.png", "Review Length Distribution"),
            ("most_common_words.png", "Most Common Words"),
            ("positive_words.png", "Top Positive Review Words"),
            ("negative_words.png", "Top Negative Review Words"),
            ("wordcloud_positive.png", "Positive Word Cloud"),
            ("wordcloud_negative.png", "Negative Word Cloud"),
        ]

        cols = st.columns(2)
        for i, (filename, caption) in enumerate(figures):
            fig_path = FIGURES_DIR / filename
            with cols[i % 2]:
                if fig_path.exists():
                    st.image(str(fig_path), caption=caption, use_container_width=True)
                else:
                    st.info(f"{caption}: run `python src/eda.py` to generate this figure.")
