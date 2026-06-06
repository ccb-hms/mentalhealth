"""Shared plotting helpers."""
import matplotlib
matplotlib.use("Agg")  # file-only backend; no interactive display needed
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion(y_true, y_pred, labels, title, path) -> None:
    """Save a row-normalised confusion matrix (diagonal = per-class recall)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true",
        xticks_rotation=45, values_format=".2f", cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
