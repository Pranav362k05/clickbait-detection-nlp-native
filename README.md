# 🎯 Clickbait Headline Detection using NLP

A complete Machine Learning project that detects clickbait news headlines using
Natural Language Processing (NLP) techniques.

# 🔗👉 Check out my project poster & demo to understand it easier and better!

**📊 Poster:** https://pranav362k05.github.io/clickbait-detection-nlp-native/
**🚀 Demo:** https://clickbait-detection-pranav-krishna.streamlit.app/

---

## 📌 Project Overview

Clickbait headlines are designed to manipulate readers into clicking a link by
triggering curiosity, shocks, or emotion — rather than informing them. Examples:

- *"You Won't BELIEVE What She Did Next!"*
- *"10 Shocking Secrets Doctors Are Hiding From You"*

This project builds a text classification pipeline to **automatically detect**
whether a headline is clickbait or legitimate news using:

- **TF-IDF** for feature extraction
- **Logistic Regression**, **Naive Bayes**, and **SVM** for classification
- **Streamlit** for an interactive web interface

---

## 📂 Project Structure

```
clickbait_detector/
│
├── data/                    ← Place your dataset here
│   └── clickbait_data.csv   ← (downloaded from Kaggle)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        ← Text cleaning & data loading
│   ├── features.py          ← TF-IDF feature extraction
│   ├── train.py             ← Model training
│   └── evaluate.py          ← Metrics, charts, confusion matrix
│
├── models/                  ← Created after running main.py
│   ├── best_model.pkl       ← Saved Logistic Regression model
│   ├── vectorizer.pkl       ← Saved TF-IDF vectorizer
│   ├── all_models.pkl       ← All 3 trained models
│   └── all_metrics.pkl      ← Performance metrics dict
│
├── plots/                   ← Generated charts (confusion matrices, comparison)
│
├── main.py                  ← Run this first — trains & evaluates all models
├── app.py                   ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

**Dataset:** [Clickbait Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)

| Property      | Details                                |
|---------------|----------------------------------------|
| Size          | ~32,000 headlines                      |
| Classes       | `1` = Clickbait, `0` = Not Clickbait  |
| Format        | CSV (two columns: headline, clickbait) |
| Balance       | ~50/50 split between classes           |

**How to get the dataset:**
1. Go to: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset
2. Click **Download**
3. Extract the ZIP file
4. Place the CSV file inside the `data/` folder

The code will **automatically detect** the file — no renaming required.

---

## ⚙️ How to Run the Project

### Step 1: Clone or download this project

```bash
# Navigate to the project folder in VS Code terminal
cd clickbait_detector
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

- Download the CSV from Kaggle (see Dataset section above)
- Place it inside `data/` folder

### Step 5: Train the models

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Extract TF-IDF features
- Train all 3 models
- Print evaluation metrics
- Save plots to `plots/`
- Save model files to `models/`
- Launch an interactive demo in the terminal

### Step 6: Run the Streamlit web app

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🧠 How TF-IDF Works (Simple Explanation)

**TF-IDF = Term Frequency × Inverse Document Frequency**

Imagine you're reading thousands of news headlines:

**Term Frequency (TF):**
How often does a word appear in *one specific headline*?
→ If "shocking" appears 2 times in a 10-word headline: TF = 0.2

**Inverse Document Frequency (IDF):**
How rare is that word across *all* headlines?
→ "The" appears everywhere → very low IDF (not useful)
→ "Shocking" appears rarely → high IDF (very useful for prediction)

**TF-IDF Score = TF × IDF**
→ Words that appear often in one headline but rarely across all = highest scores
→ These are the most *informative* words for that headline

**Why we use `ngram_range=(1, 2)`:**
Instead of just single words, we also capture **two-word phrases**:
- "won't believe" = powerful clickbait signal
- "you won't" = another common clickbait pattern
These phrases can't be detected with single words alone.

---

## 🤖 Why Logistic Regression?

Logistic Regression is ideal for this task because:

| Reason | Explanation |
|--------|-------------|
| **Works with high dimensions** | TF-IDF creates thousands of features; LR handles this efficiently |
| **Interpretable** | Each word gets a positive/negative weight — easy to explain |
| **Fast** | Trains in seconds even on large datasets |
| **Probabilistic output** | Gives a confidence score (0–100%), not just a label |
| **Strong baseline** | Often competitive with complex models for text classification |

**Example interpretation:**
- Weight of "shocking" = +0.8 → strongly predicts clickbait
- Weight of "government" = -0.5 → strongly predicts legitimate news

## 🤖 Why Naive Bayes?

Naive Bayes is effective for this task because:

| Reason | Explanation |
|--------|-------------|
| Works well with text data | Uses word frequencies/probabilities, which naturally fit NLP problems |
| Handles high-dimensional data | Performs efficiently even with thousands of TF-IDF features |
| Fast and lightweight | Requires very little computation and trains quickly |
| Performs well on small datasets | Doesn’t need large amounts of data to generalize well |
| Robust to irrelevant features | Less affected by noisy or unnecessary words |
| Strong baseline model | Often gives surprisingly good performance in text classification |

---

### Example Interpretation

Naive Bayes calculates probabilities like:

P(clickbait | "shocking") > P(non-clickbait | "shocking")
→ predicts Clickbait

P(non-clickbait | "government") > P(clickbait | "government")
→ predicts Not Clickbait


---

### How it works (Simple Explanation)

Instead of learning weights like Logistic Regression, Naive Bayes calculates probabilities of words given a class.

Example:

Headline: "shocking secret revealed"

P(clickbait) ∝
P("shocking"|clickbait) ×
P("secret"|clickbait) ×
P("revealed"|clickbait)


---

### Key Idea

Naive Bayes assumes that words are independent and combines their probabilities to make a prediction.

---

### One-Line Summary

Naive Bayes works well because it directly models the probability of words belonging to each class, making it highly suitable for text classification tasks.

---

## 📈 Example Outputs

### Terminal Output (main.py)

```
==================================================
  Results: Naive Bayes (Best)
==================================================
  Accuracy    : 0.9711
  Precision   : 0.9643
  Recall      : 0.9784
  F1-Score    : 0.9713
==================================================

Detailed Classification Report:
                precision    recall  f1-score   support

 Not Clickbait       0.96      0.96      0.96      3180
      Clickbait       0.96      0.96      0.96      3220

      accuracy                           0.96      6400
```

### Custom Headline Predictions

```
Headline : You Won't Believe What This Celebrity Did Next
Result   : 🔴 CLICKBAIT

Headline : Government Announces New Budget Plan for 2025
Result   : 🟢 NOT CLICKBAIT

Headline : 10 Shocking Secrets Doctors Don't Want You to Know
Result   : 🔴 CLICKBAIT

Headline : Scientists Discover New Species in the Amazon
Result   : 🟢 NOT CLICKBAIT
```

---

## 🛠️ Technologies Used

| Library        | Purpose                              |
|----------------|--------------------------------------|
| `scikit-learn` | ML models, TF-IDF, evaluation        |
| `pandas`       | Data loading and manipulation        |
| `numpy`        | Numerical operations                 |
| `matplotlib`   | Plotting charts and confusion matrix |
| `seaborn`      | Heatmap for confusion matrix         |
| `joblib`       | Saving/loading models                |
| `streamlit`    | Web interface                        |

---