"""
app.py
------
Streamlit Web Interface for the Clickbait Headline Detector.

This app provides two pages:
    Page 1: Predict — Enter a headline and get an instant prediction
    Page 2: Model Insights — Compare model performance and view confusion matrices

HOW TO RUN:
    streamlit run app.py

REQUIREMENTS:
    Run main.py first to train models and generate models/ folder.
"""

import streamlit as st
import joblib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_text

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clickbait Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean, modern look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Prediction result boxes */
    .clickbait-box {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ff4b4b;
        margin: 16px 0;
    }
    .notclickbait-box {
        background: linear-gradient(135deg, #00cc8822, #00cc8844);
        border: 2px solid #00cc88;
        border-radius: 12px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #00cc88;
        margin: 16px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2e3250;
    }

    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #aab2d0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #161823;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED MODELS
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    """
    Load saved model artifacts from disk.
    @st.cache_resource ensures they are loaded only once and cached.
    """
    required_files = [
        os.path.join(MODELS_DIR, "best_model.pkl"),
        os.path.join(MODELS_DIR, "vectorizer.pkl"),
    ]
    for f in required_files:
        if not os.path.exists(f):
            return None, None, None, None

    best_model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    vectorizer  = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

    all_models  = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))  if os.path.exists(os.path.join(MODELS_DIR, "all_models.pkl"))  else None
    all_metrics = joblib.load(os.path.join(MODELS_DIR, "all_metrics.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "all_metrics.pkl")) else None

    return best_model, vectorizer, all_models, all_metrics


best_model, vectorizer, all_models, all_metrics = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Clickbait Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Predict Headline", "📊 Model Insights"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this App**

This tool uses Natural Language Processing (NLP) and Machine Learning to detect
clickbait news headlines.

**Model:** Logistic Regression + TF-IDF

**Dataset:** Kaggle Clickbait Dataset
""")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK IF MODELS ARE LOADED
# ─────────────────────────────────────────────────────────────────────────────
if best_model is None or vectorizer is None:
    st.error("""
    ⚠️ **Models not found!**

    Please run `main.py` first to train the models:
    ```
    python main.py
    ```
    This will create the `models/` folder with all required files.
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PREDICT HEADLINE
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔍 Predict Headline":
    st.title("🎯 Clickbait Headline Detector")
    st.markdown("Enter any news headline below to instantly check if it's **clickbait** or **legitimate news**.")
    st.markdown("---")

    # Input area
    headline_input = st.text_area(
        "📰 Enter a News Headline:",
        placeholder="e.g. You Won't Believe What Happened Next...",
        height=100,
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    predict_btn = col1.button("🔍 Predict", use_container_width=True, type="primary")
    clear_btn   = col2.button("🗑️ Clear",   use_container_width=True)

    if predict_btn and headline_input.strip():
        with st.spinner("Analyzing headline..."):
            cleaned   = clean_text(headline_input.strip())
            vector    = vectorizer.transform([cleaned])
            pred      = best_model.predict(vector)[0]
            label     = "CLICKBAIT" if pred == 1 else "NOT CLICKBAIT"

            # Try to get probability (Logistic Regression supports this)
            try:
                prob = best_model.predict_proba(vector)[0]
                confidence = max(prob) * 100
                has_proba  = True
            except AttributeError:
                confidence = None
                has_proba  = False

        # Display result
        st.markdown("### Prediction Result")

        if label == "CLICKBAIT":
            st.markdown(
                f'<div class="clickbait-box">🔴 CLICKBAIT</div>',
                unsafe_allow_html=True
            )
            st.warning("This headline appears to use typical clickbait patterns designed to generate curiosity gaps or emotional reactions.")
        else:
            st.markdown(
                f'<div class="notclickbait-box">🟢 NOT CLICKBAIT</div>',
                unsafe_allow_html=True
            )
            st.success("This headline appears to be from legitimate news reporting.")

        if has_proba and confidence:
            st.markdown(f"**Confidence:** `{confidence:.1f}%`")

        # Show cleaned text
        with st.expander("🔧 Preprocessed Text (what the model sees)"):
            st.code(cleaned)

    elif predict_btn and not headline_input.strip():
        st.warning("⚠️ Please enter a headline before clicking Predict.")

    # ── Example Headlines ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Try These Examples")

    examples = {
        "🔴 Clickbait Examples": [
            "You Won't Believe What This Celebrity Did Next!",
            "10 Shocking Secrets Doctors Don't Want You to Know",
            "This One Weird Trick Will Change Your Life Forever",
            "She Posted This Photo And The Internet Is Losing Its Mind",
        ],
        "🟢 Non-Clickbait Examples": [
            "Government Announces New Budget Plan for 2025",
            "Scientists Discover New Species in the Amazon Rainforest",
            "Federal Reserve Holds Interest Rates Steady",
            "Local Hospital Expands Emergency Services Capacity",
        ],
    }

    col_a, col_b = st.columns(2)

    for col, (category, headlines) in zip([col_a, col_b], examples.items()):
        with col:
            st.markdown(f"**{category}**")
            for h in headlines:
                if st.button(h, key=h, use_container_width=True):
                    cleaned   = clean_text(h)
                    vector    = vectorizer.transform([cleaned])
                    pred      = best_model.predict(vector)[0]
                    label_str = "🔴 CLICKBAIT" if pred == 1 else "🟢 NOT CLICKBAIT"
                    st.info(f"**{h}**\n\n→ **{label_str}**")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.title("📊 Model Performance Insights")
    st.markdown("Compare how each model performed on the test dataset.")
    st.markdown("---")

    if all_metrics is None:
        st.warning("Detailed metrics not found. Run `main.py` to generate them.")
        st.stop()

    # ── Metric Cards ─────────────────────────────────────────────────────────
    st.markdown("### 🏆 Best Model (Naive Bayes) Metrics")

    lr_metrics = all_metrics.get("Naive Bayes", {})
    col1, col2, col3, col4 = st.columns(4)

    for col, (metric, val) in zip([col1, col2, col3, col4], lr_metrics.items()):
        with col:
            st.metric(label=metric, value=f"{val:.4f}")

    st.markdown("---")

    # ── Model Comparison Bar Chart ────────────────────────────────────────────
    st.markdown("### 📈 All Models — Performance Comparison")

    model_names  = list(all_metrics.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x     = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1e2130")

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        vals   = [all_metrics[m][metric] for m in model_names]
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, vals, width, label=metric, color=color, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color="white"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color="white", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=10, facecolor="#1e2130", labelcolor="white")
    ax.grid(axis="y", alpha=0.2, color="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3250")

    st.pyplot(fig)
    plt.close(fig)

    # ── Full Metrics Table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Metrics Table")

    import pandas as pd
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metrics_df, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔲 Confusion Matrices")
    st.markdown("*Loaded from saved plot images generated by `main.py`*")

    plots_dir = "plots"
    if os.path.exists(plots_dir):
        cm_images = [
            f for f in os.listdir(plots_dir)
            if f.startswith("confusion_matrix") and f.endswith(".png")
        ]
        if cm_images:
            cols = st.columns(len(cm_images))
            for col, img_file in zip(cols, sorted(cm_images)):
                img_path = os.path.join(plots_dir, img_file)
                model_label = img_file.replace("confusion_matrix_", "").replace(".png", "").replace("_", " ")
                with col:
                    st.image(img_path, caption=model_label, use_container_width=True)
        else:
            st.info("No confusion matrix images found. Run `main.py` first.")
    else:
        st.info("Plots directory not found. Run `main.py` to generate plots.")

    # ── TF-IDF Explainer ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 How TF-IDF Works")
    st.info("""
**TF-IDF = Term Frequency × Inverse Document Frequency**

| Term | Meaning |
|------|---------|
| **TF** (Term Frequency) | How often a word appears in ONE headline |
| **IDF** (Inverse Document Frequency) | How rare the word is across ALL headlines |
| **TF-IDF** | Words common in one doc but rare overall → get highest scores |

**Why it's useful:**
- Words like "the", "is", "a" appear in every headline → low IDF → low score → model ignores them
- Words like "shocking", "won't believe" are rare but informative → high score → model pays attention
- With `ngram_range=(1,2)`, two-word phrases like "you won't" are also captured as features
    """)
