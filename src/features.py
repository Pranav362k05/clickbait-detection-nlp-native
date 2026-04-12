"""
features.py
-----------
This module handles feature extraction using TF-IDF Vectorization.

WHAT IS TF-IDF?
TF-IDF stands for Term Frequency - Inverse Document Frequency.
It converts raw text into numerical vectors that machine learning models can understand.

- TF (Term Frequency): How often a word appears in one headline.
  e.g., if 'shocking' appears 2 times in a 10-word headline → TF = 0.2

- IDF (Inverse Document Frequency): How rare a word is across ALL headlines.
  Common words like 'the', 'is' get a LOW score (they appear everywhere).
  Rare but meaningful words like 'shocking' get a HIGH score.

- TF-IDF = TF × IDF → Words that appear often in ONE headline but rarely
  across others get the highest scores — these are the most distinctive words.

WHY TF-IDF OVER BAG-OF-WORDS?
Bag-of-Words just counts words. TF-IDF weighs them by importance.
This helps the model ignore filler words and focus on meaningful ones.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with carefully chosen settings.

    Settings explained:
        max_features=10000  → Only keep the top 10,000 most important words
                              (avoids memory issues and overfitting)

        ngram_range=(1, 2)  → Use single words AND two-word phrases
                              e.g., "won't believe" is captured as a phrase
                              This is crucial for clickbait which uses
                              specific phrase patterns

        sublinear_tf=True   → Apply log scaling to term frequency
                              Prevents very common words from dominating

    Returns:
        TfidfVectorizer: Configured (unfitted) vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,   # Vocabulary size limit
        ngram_range=(1, 2),   # Unigrams + Bigrams
        sublinear_tf=True,    # Log normalization
        min_df=2,             # Ignore words appearing in fewer than 2 headlines
    )
    return vectorizer


def fit_and_transform(vectorizer: TfidfVectorizer, texts):
    """
    Fit the vectorizer on training data and transform it.

    WHY FIT ONLY ON TRAINING DATA?
    If we fit on the entire dataset (including test data), the model would
    "see" test data during training — this is called data leakage and gives
    artificially high accuracy scores.

    Parameters:
        vectorizer: An unfitted TfidfVectorizer
        texts: Training headlines (list or Series)

    Returns:
        Sparse matrix of TF-IDF features for training data
    """
    print("[INFO] Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(texts)
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"[INFO] Feature matrix shape: {X_train.shape}\n")
    return X_train


def transform_only(vectorizer: TfidfVectorizer, texts):
    """
    Transform text using an already-fitted vectorizer.
    Used for test data and new user inputs.

    Parameters:
        vectorizer: An already-fitted TfidfVectorizer
        texts: Headlines to transform

    Returns:
        Sparse matrix of TF-IDF features
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = "vectorizer.pkl"):
    """
    Save the fitted vectorizer to disk so we can reuse it in the Streamlit app.

    Parameters:
        vectorizer: Fitted TfidfVectorizer
        path (str): File path to save to
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[INFO] Vectorizer saved to: {path}")


def load_vectorizer(path: str = "vectorizer.pkl") -> TfidfVectorizer:
    """
    Load a previously saved vectorizer from disk.

    Parameters:
        path (str): File path to load from

    Returns:
        TfidfVectorizer: Loaded fitted vectorizer
    """
    return joblib.load(path)
