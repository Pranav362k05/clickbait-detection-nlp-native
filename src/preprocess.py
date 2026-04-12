"""
preprocess.py
-------------
This module handles all text preprocessing steps.

WHY PREPROCESS?
Raw text contains noise (punctuation, capital letters, extra spaces) that can
confuse machine learning models. By standardizing the text, we help the model
focus on the actual words and their patterns rather than formatting differences.
"""

import re
import string
import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the clickbait dataset from a CSV or JSON file.

    WHY HANDLE BOTH FORMATS?
    The Kaggle clickbait dataset is available in different formats depending on
    which version you download. Supporting both makes the code more robust.

    Parameters:
        filepath (str): Path to the dataset file (CSV or JSON)

    Returns:
        pd.DataFrame: A dataframe with at least 'headline' and 'label' columns
    """
    print(f"[INFO] Loading dataset from: {filepath}")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".json") or filepath.endswith(".jsonl"):
        df = pd.read_json(filepath, lines=True)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON/JSONL.")

    # -----------------------------------------------------------------------
    # Automatically detect column names regardless of dataset version
    # The Kaggle clickbait dataset may use different column names:
    #   - 'headline' or 'text' for the article title
    #   - 'clickbait' or 'label' for the binary target (1 = clickbait, 0 = not)
    # -----------------------------------------------------------------------
    column_map = {}

    # Detect headline column
    for col in df.columns:
        if col.lower() in ["headline", "text", "title", "sentence"]:
            column_map[col] = "headline"
            break

    # Detect label column
    for col in df.columns:
        if col.lower() in ["clickbait", "label", "class", "tag"]:
            column_map[col] = "label"
            break

    df = df.rename(columns=column_map)

    # Validate that we found both required columns
    if "headline" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Could not find expected columns. Found: {list(df.columns)}. "
            "Please ensure your dataset has a headline/text column and a label/clickbait column."
        )

    # Keep only the two important columns and drop rows with missing values
    df = df[["headline", "label"]].dropna().reset_index(drop=True)

    print(f"[INFO] Dataset loaded: {len(df)} rows")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")

    return df


def clean_text(text: str) -> str:
    """
    Clean a single headline string.

    Steps:
        1. Lowercase  — so 'Click' and 'click' are treated the same
        2. Remove punctuation  — symbols add noise without adding meaning
        3. Remove extra whitespace  — normalizes spacing

    WHY NOT REMOVE STOPWORDS?
    Clickbait often uses emotional stopword-like phrases ("you won't believe").
    Removing them might discard valuable signals, so we keep all words.

    Parameters:
        text (str): Raw headline text

    Returns:
        str: Cleaned headline text
    """
    # Step 1: Lowercase everything
    text = text.lower()

    # Step 2: Remove punctuation using Python's string.punctuation constant
    # This removes: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3: Remove extra whitespace (including tabs and newlines)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to every row in the 'headline' column.

    Parameters:
        df (pd.DataFrame): Raw dataframe with 'headline' column

    Returns:
        pd.DataFrame: Dataframe with an added 'clean_headline' column
    """
    print("[INFO] Preprocessing text...")

    # Apply the cleaning function to every headline
    df["clean_headline"] = df["headline"].astype(str).apply(clean_text)

    print("[INFO] Text preprocessing complete.\n")
    return df
