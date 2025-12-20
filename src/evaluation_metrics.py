from matplotlib import pyplot as plt
import numpy as np

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