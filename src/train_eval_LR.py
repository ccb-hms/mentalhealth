"""
Fit and evaluate the TF-IDF + Logistic Regression baseline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.data import get_splits, SEED, LABELS
from src.plots import plot_confusion

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
for _d in (MODEL_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def build_tfidf(ngram_range=(1, 2)) -> TfidfVectorizer:
    # TF-IDF feature extraction for the baseline model
    return TfidfVectorizer(
        ngram_range=ngram_range,  # Considers both unigrams and bigrams.
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        stop_words="english",
        strip_accents="unicode",
    )


def tfidf_logistic_regression(C: float = 1.0, ngram_range=(1, 2)) -> Pipeline:
    # Scikit-learn pipeline that chains the vectorizer and the classifier
    return Pipeline([
        ("tfidf", build_tfidf(ngram_range)),
        ("clf", LogisticRegression(
            C=C,  # inverse regularisation strength; smaller C = stronger regularisation
            class_weight="balanced",
            max_iter=2000,
        )),
    ])


def _top_features(model, k=12) -> dict:
    # Highest-weighted TF-IDF terms per class (interpretability).
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[i])[::-1][:k]
        out[cls] = vocab[top].tolist()
    return out


def main() -> None:
    train, val, test = get_splits(SEED)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}\n")

    # --- Fit the baseline (fixed C=1.0, sklearn default) ---
    model = tfidf_logistic_regression().fit(train["statement"], train["status"])
    joblib.dump(model, MODEL_DIR / "tfidf_logistic_regression.joblib")
    print("Fitted TF-IDF + LogReg (C=1.0)")

    # --- Evaluate on the test set ---
    pred = model.predict(test["statement"])
    acc = round(accuracy_score(test["status"], pred), 4)
    macro = round(f1_score(test["status"], pred, average="macro"), 4)
    per_class = classification_report(test["status"], pred,
                                      output_dict=True, zero_division=0)
    print(f"Test:  accuracy={acc:.3f}  macro-F1={macro:.3f}")

    plot_confusion(test["status"], pred, LABELS,
                   "Confusion matrix — TF-IDF + Logistic Regression",
                   FIG_DIR / "confusion_full.png")

    metrics = {
        "primary": "tfidf_logistic_regression",
        "comparison_full": [{"model": "tfidf_logistic_regression",
                             "accuracy": acc, "macro_f1": macro}],
        "per_class_full": per_class,
        "top_features": _top_features(model),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {OUT_DIR / 'metrics.json'} and {FIG_DIR / 'confusion_full.png'}")


if __name__ == "__main__":
    main()
