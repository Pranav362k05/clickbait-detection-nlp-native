"""
train.py
--------
This module trains multiple machine learning classifiers and saves the best one.

WHY MULTIPLE MODELS?
Comparing models helps us understand which algorithm is best suited for our
specific problem. Each model has different strengths:

- Logistic Regression: Simple, fast, interpretable. Works well with TF-IDF
  because it learns a weight for each word/phrase → easy to explain.

- Naive Bayes (MultinomialNB): Probabilistic model, great for text.
  Assumes features are independent (naive assumption, but works well in practice).

- SVM (Support Vector Machine): Finds the best decision boundary between classes.
  Effective in high-dimensional spaces like TF-IDF vectors.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import joblib
import os


def get_models() -> dict:
    """
    Return a dictionary of all models to be trained.

    WHY THESE HYPERPARAMETERS?
    - LogisticRegression max_iter=1000: More iterations help convergence on text data
    - C=1.0: Regularization strength (default, balanced bias-variance tradeoff)
    - LinearSVC: Faster than kernel SVM for large text datasets
    - MultinomialNB: Works with non-negative features (TF-IDF values are always >= 0)

    Returns:
        dict: { model_name: model_object }
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,   # Allow enough iterations to converge
            C=1.0,           # Regularization (lower = stronger regularization)
            random_state=42  # For reproducibility
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.1        # Laplace smoothing (handles unseen words gracefully)
        ),
        "SVM": LinearSVC(
            max_iter=2000,   # SVM needs more iterations on text data
            C=1.0,
            random_state=42
        ),
    }
    return models


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.

    WHY 80/20 SPLIT?
    - 80% training: Enough data for the model to learn patterns
    - 20% testing: Enough to evaluate performance reliably
    - stratify=y: Ensures both sets have the same class ratio (important for
      imbalanced datasets where one class may dominate)

    Parameters:
        X: Feature matrix (TF-IDF vectors)
        y: Labels (0 = not clickbait, 1 = clickbait)
        test_size (float): Proportion of data for testing
        random_state (int): Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"[INFO] Splitting data: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # Maintain class distribution in both splits
    )

    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Testing samples:  {X_test.shape[0]}\n")

    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train, models: dict) -> dict:
    """
    Train all models on the training data.

    Parameters:
        X_train: Training feature matrix
        y_train: Training labels
        models (dict): Dictionary of model name → model object

    Returns:
        dict: { model_name: trained_model }
    """
    trained_models = {}

    for name, model in models.items():
        print(f"[INFO] Training: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"[INFO] {name} training complete.")

    print()
    return trained_models


def save_model(model, path: str = "model.pkl"):
    """
    Save a trained model to disk using joblib.

    WHY JOBLIB?
    Joblib is more efficient than pickle for large NumPy arrays (like model weights).

    Parameters:
        model: Trained sklearn model
        path (str): File path to save the model
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")


def load_model(path: str = "model.pkl"):
    """
    Load a previously saved model from disk.

    Parameters:
        path (str): File path to the saved model

    Returns:
        Loaded sklearn model
    """
    return joblib.load(path)
