"""
classifier/train.py
===================
Train and evaluate the prompt complexity classifier.

Usage
-----
    python classifier/train.py

What it does
1. Loads ``data/train.csv`` and ``data/test.csv`` (create them first by
   running ``python data/split_dataset.py``).
2. Extracts features using ``classifier/features.py``.
3. Trains a ``RandomForestClassifier`` (interpretable, good baseline).
4. Evaluates on the held-out test set.
5. Saves the trained pipeline to ``classifier/model.pkl``.
6. Writes the evaluation report to ``classifier/eval_report.txt``.

Target: ≥ 80 % test accuracy.  If below, a WARNING is printed and the
confusion matrix is displayed to aid diagnosis.
"""
from __future__ import annotations

import pathlib
import pickle
import sys
import textwrap
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Ensure the project root is on the path when running from any directory.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from classifier.features import extract_features  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
CLASSIFIER_DIR = _PROJECT_ROOT / "classifier"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_PKL = CLASSIFIER_DIR / "model.pkl"
REPORT_TXT = CLASSIFIER_DIR / "eval_report.txt"

ACCURACY_THRESHOLD = 0.80
RANDOM_STATE = 42
LABEL_ORDER = ["simple", "moderate", "complex"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_features(csv_path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and return (feature_df, label_series)."""
    df = pd.read_csv(csv_path)
    if "prompt_text" not in df.columns or "complexity_label" not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'prompt_text' and 'complexity_label' columns.")

    feature_rows = [extract_features(p) for p in df["prompt_text"]]
    X = pd.DataFrame(feature_rows)
    y = df["complexity_label"]
    return X, y


def _check_splits_exist() -> None:
    """Verify train/test CSVs exist; instruct user to create them if not."""
    missing = [p for p in (TRAIN_CSV, TEST_CSV) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing dataset splits: {names}\n"
            "Run `python data/split_dataset.py` first to generate train/test CSVs."
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train() -> None:
    """Full training + evaluation pipeline.  Saves model and eval report."""
    _check_splits_exist()

    # ------------------------------------------------------------------ load
    print("Loading features …")
    X_train, y_train = _load_features(TRAIN_CSV)
    X_test, y_test = _load_features(TEST_CSV)
    print(f"  Train: {len(X_train)} rows   Test: {len(X_test)} rows")
    print(f"  Features: {list(X_train.columns)}")

    # ----------------------------------------------------------------- train
    print("\nTraining RandomForestClassifier …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --------------------------------------------------------------- evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)

    # Pretty-print results
    print(f"\n{'='*60}")
    print(f"Test accuracy: {acc:.2%}")
    print(f"{'='*60}")
    print(report_str)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {LABEL_ORDER}")
    print(f"  {cm}")

    if acc < ACCURACY_THRESHOLD:
        warnings.warn(
            f"\n⚠️  WARNING: Test accuracy {acc:.2%} is below the {ACCURACY_THRESHOLD:.0%} target.\n"
            "  Review the confusion matrix above to identify which tiers are being confused.\n"
            "  Possible fixes: add more training data, improve feature engineering, tune hyperparameters.",
            stacklevel=2,
        )

    # ----------------------------------------------------------------- feature importance
    importances = sorted(
        zip(X_train.columns, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    importance_str = "\n".join(f"  {name:<30} {imp:.4f}" for name, imp in importances)
    print(f"\nFeature importances (descending):\n{importance_str}")

    # ------------------------------------------------------------------- save model
    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PKL.open("wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved → {MODEL_PKL}")

    # ------------------------------------------------------------------ write report
    report_content = textwrap.dedent(f"""\
        LLM Cost Autopilot — Complexity Classifier Evaluation Report
        =============================================================

        Test accuracy : {acc:.4f} ({acc:.2%})
        Target        : {ACCURACY_THRESHOLD:.0%}
        Status        : {'✅ PASS' if acc >= ACCURACY_THRESHOLD else '❌ BELOW TARGET — see confusion matrix'}

        Dataset
        -------
        Train rows : {len(X_train)}
        Test rows  : {len(X_test)}
        Features   : {list(X_train.columns)}

        Classification Report
        ---------------------
        {report_str}

        Confusion Matrix (rows=actual, cols=predicted)
        -----------------------------------------------
        Labels : {LABEL_ORDER}
        {cm}

        Feature Importances
        -------------------
        {importance_str}
    """)

    REPORT_TXT.write_text(report_content, encoding="utf-8")
    print(f"Evaluation report saved → {REPORT_TXT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
