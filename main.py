"""
main.py
-------
Entry point for the Clickbait Headline Detection project.

This script orchestrates the full NLP pipeline:
    1. Load dataset
    2. Preprocess text
    3. Extract TF-IDF features
    4. Train models (Logistic Regression, Naive Bayes, SVM)
    5. Evaluate models and visualize results
    6. Save best model + vectorizer for use in the Streamlit app
    7. Run interactive custom headline prediction

HOW TO RUN:
    python main.py

REQUIREMENTS:
    - Dataset must be placed in the data/ folder
    - Supported file names: clickbait_data.csv, data.csv, clickbait.json, etc.
"""

import os
import sys
import glob

# Add src/ to the Python path so we can import our modules cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_dataset, preprocess_dataframe, clean_text
from features import build_tfidf_vectorizer, fit_and_transform, transform_only, save_vectorizer
from train import get_models, split_data, train_all_models, save_model
from evaluate import evaluate_all_models, plot_confusion_matrix, plot_model_comparison


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR     = "data"              # Folder containing your downloaded dataset
MODELS_DIR   = "models"           # Where trained models will be saved
PLOTS_DIR    = "plots"            # Where charts/plots will be saved
PRIMARY_MODEL = "Logistic Regression"  # Model to save for Streamlit app


def find_dataset(data_dir: str) -> str:
    """
    Automatically find the dataset file inside data/ folder.
    Supports CSV, JSON, and JSONL formats.
    """
    supported_patterns = ["*.csv", "*.json", "*.jsonl"]
    found_files = []

    for pattern in supported_patterns:
        found_files.extend(glob.glob(os.path.join(data_dir, pattern)))

    if not found_files:
        raise FileNotFoundError(
            f"\n[ERROR] No dataset found in '{data_dir}/' folder.\n"
            "Please download the Clickbait Dataset from Kaggle and place it inside data/\n"
            "Supported formats: .csv, .json, .jsonl\n"
            "Example: data/clickbait_data.csv"
        )

    # Use the first found file
    chosen = found_files[0]
    print(f"[INFO] Dataset found: {chosen}")
    return chosen


def predict_headline(headline: str, model, vectorizer) -> str:
    """
    Predict whether a single headline is clickbait or not.

    HOW IT WORKS:
        1. Clean the raw headline text
        2. Transform it using the fitted TF-IDF vectorizer
        3. Pass the feature vector to the trained model
        4. Interpret the output: 1 = Clickbait, 0 = Not Clickbait

    Parameters:
        headline (str): Raw headline text entered by the user
        model: Trained sklearn classifier
        vectorizer: Fitted TfidfVectorizer

    Returns:
        str: "CLICKBAIT" or "NOT CLICKBAIT"
    """
    cleaned = clean_text(headline)
    vector  = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "CLICKBAIT" if prediction == 1 else "NOT CLICKBAIT"


def interactive_demo(model, vectorizer):
    """
    Let the user enter custom headlines and get predictions interactively.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "="*60)
    print("  CUSTOM HEADLINE PREDICTOR")
    print("  Enter any news headline to check if it's clickbait.")
    print("  Type 'quit' to exit.")
    print("="*60 + "\n")

    # Example headlines shown automatically on startup
    example_headlines = [
        "You Won't Believe What This Celebrity Did Next",
        "Government Announces New Budget Plan for 2025",
        "10 Shocking Secrets Doctors Don't Want You to Know",
        "Scientists Discover New Species in the Amazon",
        "This One Weird Trick Will Change Your Life Forever",
    ]

    print("[DEMO] Predicting example headlines:\n")
    for h in example_headlines:
        result = predict_headline(h, model, vectorizer)
        label = "🔴 CLICKBAIT" if result == "CLICKBAIT" else "🟢 NOT CLICKBAIT"
        print(f"  Headline : {h}")
        print(f"  Result   : {label}\n")

    # Interactive loop
    print("-" * 60)
    while True:
        user_input = input("\nEnter a headline (or 'quit'): ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("[INFO] Exiting. Goodbye!")
            break

        if not user_input:
            print("[WARNING] Please enter a headline.")
            continue

        result = predict_headline(user_input, model, vectorizer)
        label = "🔴 CLICKBAIT" if result == "CLICKBAIT" else "🟢 NOT CLICKBAIT"
        print(f"  Result: {label}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  CLICKBAIT HEADLINE DETECTION — NLP PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Find and load the dataset ──────────────────────────────────
    dataset_path = find_dataset(DATA_DIR)
    df = load_dataset(dataset_path)

    # ── Step 2: Preprocess text ────────────────────────────────────────────
    df = preprocess_dataframe(df)

    # ── Step 3: TF-IDF Feature Extraction ─────────────────────────────────
    vectorizer = build_tfidf_vectorizer()

    # IMPORTANT: Fit the vectorizer ONLY on training data (no data leakage)
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df["clean_headline"], df["label"]
    )

    X_train = fit_and_transform(vectorizer, X_train_raw)
    X_test  = transform_only(vectorizer, X_test_raw)

    # ── Step 4: Train Models ───────────────────────────────────────────────
    models = get_models()
    trained_models = train_all_models(X_train, y_train, models)

    # ── Step 5: Evaluate Models ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)

    all_metrics = evaluate_all_models(trained_models, X_test, y_test)

    # ── Step 6: Visualize Results ──────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot confusion matrix for each model
    for name, model in trained_models.items():
        plot_confusion_matrix(model, X_test, y_test, model_name=name, save_dir=PLOTS_DIR)

    # Plot side-by-side model comparison
    plot_model_comparison(all_metrics, save_dir=PLOTS_DIR)

    # ── Step 7: Save Best Model + Vectorizer ───────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(trained_models[PRIMARY_MODEL], path=os.path.join(MODELS_DIR, "best_model.pkl"))
    save_vectorizer(vectorizer, path=os.path.join(MODELS_DIR, "vectorizer.pkl"))

    # Also save all trained models for the Streamlit comparison page
    import joblib
    joblib.dump(trained_models, os.path.join(MODELS_DIR, "all_models.pkl"))
    joblib.dump(all_metrics,    os.path.join(MODELS_DIR, "all_metrics.pkl"))
    print(f"\n[INFO] All models and metrics saved to: {MODELS_DIR}/\n")

    # ── Step 8: Interactive Demo ───────────────────────────────────────────
    best_model = trained_models[PRIMARY_MODEL]
    interactive_demo(best_model, vectorizer)


if __name__ == "__main__":
    main()
