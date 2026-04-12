"""
evaluate.py
-----------
This module evaluates trained models and visualizes their performance.

METRICS EXPLAINED (important for viva):

- Accuracy:  What % of all predictions were correct?
             Good general metric but misleading for imbalanced classes.

- Precision: Of all headlines we predicted AS clickbait, how many actually were?
             High precision = few false alarms.

- Recall:    Of all actual clickbait headlines, how many did we catch?
             High recall = we miss very few clickbait articles.

- F1-Score:  Harmonic mean of Precision and Recall.
             Balances both metrics — best single metric for text classification.

- Confusion Matrix: A 2×2 table showing:
             - True Positives (TP):  Correctly identified clickbait
             - True Negatives (TN):  Correctly identified non-clickbait
             - False Positives (FP): Wrongly flagged as clickbait
             - False Negatives (FN): Missed actual clickbait
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import os


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a single model and print all metrics.

    Parameters:
        model: Trained sklearn model with .predict() method
        X_test: Test feature matrix (TF-IDF vectors)
        y_test: True labels for test set
        model_name (str): Name of the model (for display)

    Returns:
        dict: Dictionary of metric name → score
    """
    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate all metrics
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
    }

    # Print a formatted results table
    print(f"\n{'='*50}")
    print(f"  Results: {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value:.4f}")
    print(f"{'='*50}")

    # Print detailed classification report (per-class breakdown)
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Clickbait", "Clickbait"]
    ))

    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Evaluate all trained models and collect their metrics.

    Parameters:
        trained_models (dict): { model_name: trained_model }
        X_test: Test feature matrix
        y_test: True test labels

    Returns:
        dict: { model_name: { metric: score } }
    """
    all_metrics = {}

    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics[name] = metrics

    return all_metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model", save_dir: str = "."):
    """
    Plot and save a confusion matrix heatmap.

    WHY A HEATMAP?
    A color-coded matrix is much easier to read than raw numbers.
    Dark colors show where most predictions fall — errors are immediately visible.

    Parameters:
        model: Trained model
        X_test: Test feature matrix
        y_test: True test labels
        model_name (str): Used for title and filename
        save_dir (str): Directory to save the plot image
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers inside cells
        fmt="d",              # Integer format
        cmap="Blues",         # Blue color palette
        xticklabels=["Not Clickbait", "Clickbait"],
        yticklabels=["Not Clickbait", "Clickbait"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Confusion matrix saved to: {filepath}")
    plt.close()


def plot_model_comparison(all_metrics: dict, save_dir: str = "."):
    """
    Plot a grouped bar chart comparing all models across all metrics.

    WHY THIS CHART?
    Side-by-side comparison makes it immediately clear which model
    performs best on which metric.

    Parameters:
        all_metrics (dict): { model_name: { metric: score } }
        save_dir (str): Directory to save the chart image
    """
    model_names = list(all_metrics.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Organize data for plotting
    values = {metric: [] for metric in metric_names}
    for model_name in model_names:
        for metric in metric_names:
            values[metric].append(all_metrics[model_name][metric])

    # Plotting setup
    x = np.arange(len(model_names))       # x positions for model groups
    width = 0.2                            # Width of each bar
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # Distinct colors per metric

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw one group of bars per metric
    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values[metric], width, label=metric, color=color, alpha=0.85)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_title("Model Performance Comparison", fontsize=14, pad=15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Model comparison chart saved to: {filepath}")
    plt.close()
