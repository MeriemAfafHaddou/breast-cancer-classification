"""
Helpers to :
- calculate: recall, precision and f1 score.
- summarize and visualize cross-validation metrics across folds.
"""

# pylint: disable=import-error,no-name-in-module
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    """
    Computes macro-averaged recall for a multi-class classification task.
    """
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=3)
    y_true = K.one_hot(K.cast(y_true, "int32"), num_classes=3)

    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    recall = tp / (tp + fn + K.epsilon())
    return K.mean(recall)


def precision_m(y_true, y_pred):
    """
    Computes macro-averaged precision for a multi-class classification task.
    """
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=3)
    y_true = K.one_hot(K.cast(y_true, "int32"), num_classes=3)

    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    return K.mean(precision)


def f1_m(y_true, y_pred):
    """
    Computes macro-averaged F1 score using precision and recall.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def summarize_cv_results(val_scores, metric_names):
    """
    Displays mean ± std for each metric across folds.
    """
    scores = np.array(val_scores)

    print("📊 Cross-Validation Performance Summary\n")
    for i, name in enumerate(metric_names):
        mean = scores[:, i].mean()
        std = scores[:, i].std()
        print(f"{name:20s}: {mean:.4f} ± {std:.4f}")


def plot_cv_metrics(val_scores, metric_names):
    """
    Plots metric values across folds.
    """
    scores = np.array(val_scores)

    plt.figure(figsize=(8, 4))
    for i, name in enumerate(metric_names):
        plt.plot(scores[:, i], marker="o", label=name)

    plt.xlabel("Fold")
    plt.ylabel("Metric Value")
    plt.title("Cross-Validation Metrics Across Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
