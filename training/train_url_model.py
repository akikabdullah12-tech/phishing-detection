"""
train_url_model.py

Trains five classifiers on the 18 URL features, compares them on a held-out
test set, and saves the best one for the web app.

Run from the project root (after build_url_data.py):
    python training/train_url_model.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

# Add the project root so that "from features..." works when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.url_features import FEATURE_COLS

# Hide harmless library warnings so the output stays readable.
warnings.filterwarnings("ignore")

# The table written by build_url_data.py (18 features + target, where 1 = phishing).
DATASET_PATH = Path("data/Live_Compatible_URL_Features.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RANDOM_SEED = 42  # one fixed seed everywhere, so a re-run gives the same numbers


def load_dataset(filepath):
    """Read the feature table and make sure every value is usable."""
    df = pd.read_csv(filepath)
    # Keep only the 18 features and the label.
    df = df[FEATURE_COLS + ["target"]].copy()

    # Force numbers; anything that cannot be converted becomes NaN and is dropped below.
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # Treat infinite values (for example a ratio divided by zero) as missing too.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    # Keep only valid labels and remove rows that are exact copies of another row.
    # Note: many legitimate URLs look alike, so this can shrink the legitimate class a lot.
    return df[df["target"].isin([0, 1])].drop_duplicates().reset_index(drop=True)


def balance_training_data(X_train, y_train):
    """Make both classes the same size in the TRAINING data by downsampling."""
    data = X_train.copy()
    data["target"] = y_train.values

    legit = data[data["target"] == 0]
    phish = data[data["target"] == 1]

    # Cut both classes down to the size of the smaller one.
    # Only training data is balanced; the test data keeps its natural class ratio.
    min_count = min(len(legit), len(phish))

    legit_sampled = resample(legit, replace=False, n_samples=min_count, random_state=RANDOM_SEED)
    phish_sampled = resample(phish, replace=False, n_samples=min_count, random_state=RANDOM_SEED)

    # Put them together and shuffle the rows.
    balanced = pd.concat([legit_sampled, phish_sampled]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return balanced[FEATURE_COLS], balanced["target"]


def get_model_suite():
    """Create the five models we want to compare."""
    return {
        # Linear models work better when features are on the same scale, hence the scaler.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_SEED, class_weight="balanced", solver="liblinear")),
        ]),
        # A single tree of yes/no questions.
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=8, min_samples_leaf=4,
            random_state=RANDOM_SEED, class_weight="balanced",
        ),
        # Many trees voting together.
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
        ),
        # Linear SVM, also with scaled features.
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=1.0, random_state=RANDOM_SEED, class_weight="balanced", max_iter=15000)),
        ]),
        # Boosted trees: each new tree tries to fix the mistakes of the earlier ones.
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=3,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
        ),
    }


def get_decision_scores(model, X):
    """Return a continuous score per URL (needed for the ROC and PR curves)."""
    if hasattr(model, "predict_proba"):
        # Pick the column that belongs to class 1 (phishing).
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index(1)]
    # LinearSVC has no probabilities, but it gives a distance from the decision line.
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    """Save the confusion matrix of one model as an image."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate URL", "Phishing URL"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, scores_map, output_path):
    """Draw the ROC curves of all models on one chart."""
    plt.figure(figsize=(10, 7))
    for name, scores in scores_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    # The dashed diagonal is what a random guess would look like.
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - URL Classifiers")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true, scores_map, output_path):
    """Draw the precision-recall curves of all models on one chart."""
    plt.figure(figsize=(10, 7))
    for name, scores in scores_map.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        plt.plot(rec, prec, label=f"{name} (AUC = {auc(rec, prec):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - URL Classifiers")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(rf_model, output_path):
    """Show which URL features the Random Forest relied on the most."""
    fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": rf_model.feature_importances_})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(RESULTS_DIR / "url_feature_importance.csv", index=False)

    # The chart shows only the top 15 to keep it readable.
    top_15 = fi_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top_15["Feature"][::-1], top_15["Importance"][::-1])
    plt.xlabel("Feature Importance Weight")
    plt.title("Top 15 URL Lexical Features (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}\n"
            "Run python training/build_url_data.py first."
        )

    print(f"Loading URL feature dataset: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df):,} valid records with {len(FEATURE_COLS)} features.")

    X = df[FEATURE_COLS]
    y = df["target"]

    # 80/20 split. stratify=y keeps the same legitimate/phishing ratio in both parts.
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Balance the training part only. The test part is left alone, so the evaluation
    # reflects the real imbalance (this is why precision for the legitimate class looks low).
    X_train, y_train = balance_training_data(X_train_raw, y_train_raw)

    print(f"Balanced Train split: {len(X_train):,} records | Test split: {len(X_test):,} records")

    models = get_model_suite()

    # 5-fold cross-validation, only used to see how stable each model is.
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    evaluation_results = []
    trained_models = {}
    score_history = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        # Cross-validate a fresh copy, so the model fitted next is not affected.
        cv_scores = cross_val_score(clone(model), X_train, y_train, cv=cv_splitter, scoring="f1", n_jobs=1)
        # Fit on the whole balanced training set, then score on the untouched test set.
        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        decision_scores = get_decision_scores(model, X_test)

        trained_models[name] = model
        score_history[name] = decision_scores

        bal_acc = balanced_accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, decision_scores)

        evaluation_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, test_preds),
            "Balanced Accuracy": bal_acc,
            "Precision": precision_score(y_test, test_preds, zero_division=0),
            "Recall": recall_score(y_test, test_preds, zero_division=0),
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV Mean F1": cv_scores.mean(),
            "CV F1 Std": cv_scores.std(),
            "Training Time Seconds": time.time() - start_time,
        })
        print(f"  F1: {f1:.4f} | Balanced Acc: {bal_acc:.4f} | ROC-AUC: {roc_auc:.4f}")

    # The test data is imbalanced, so models are ranked by balanced accuracy first.
    comparison_df = (
        pd.DataFrame(evaluation_results)
        .sort_values(["Balanced Accuracy", "F1 Score", "ROC AUC"], ascending=False)
        .reset_index(drop=True)
    )
    comparison_df.to_csv(RESULTS_DIR / "url_model_comparison.csv", index=False)

    print("\nModel Comparison Table:")
    print(comparison_df[["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].round(4).to_string(index=False))

    # The table is sorted, so the first row is the winner.
    best_name = str(comparison_df.iloc[0]["Model"])
    best_model = trained_models[best_name]

    print(f"\nBest Model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["Legitimate URL", "Phishing URL"]))

    # Full per-class report for every model, saved as text.
    with open(RESULTS_DIR / "url_classification_reports.txt", "w", encoding="utf-8") as f:
        for name, m in trained_models.items():
            f.write(f"{'=' * 60}\n{name}\n{'=' * 60}\n")
            f.write(classification_report(y_test, m.predict(X_test), target_names=["Legitimate URL", "Phishing URL"], zero_division=0))
            f.write("\n\n")

    # The charts used in the report.
    plot_confusion_matrix(y_test, best_model.predict(X_test), best_name, RESULTS_DIR / "url_best_model_confusion_matrix.png")
    plot_roc_curves(y_test, score_history, RESULTS_DIR / "url_models_roc_curve.png")
    plot_pr_curves(y_test, score_history, RESULTS_DIR / "url_models_precision_recall_curve.png")
    # Feature importance comes from the Random Forest, because it gives a simple importance chart.
    plot_feature_importance(trained_models["Random Forest"], RESULTS_DIR / "url_feature_importance.png")

    # Save the winning model and the feature list; the web app loads both at start-up.
    joblib.dump(best_model, MODELS_DIR / "best_url_model.pkl")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "url_feature_columns.pkl")

    # A small summary file with the model's scores and the data sizes.
    best_metrics = comparison_df.iloc[0]
    metadata = {
        "project": "Phishing Detection Using Machine Learning",
        "module": "URL Phishing Classification",
        "best_model": best_name,
        "selection_metric": "Balanced Accuracy",
        "accuracy": float(best_metrics["Accuracy"]),
        "balanced_accuracy": float(best_metrics["Balanced Accuracy"]),
        "precision": float(best_metrics["Precision"]),
        "recall": float(best_metrics["Recall"]),
        "f1_score": float(best_metrics["F1 Score"]),
        "roc_auc": float(best_metrics["ROC AUC"]),
        "cross_validation_mean_f1": float(best_metrics["CV Mean F1"]),
        "cross_validation_f1_std": float(best_metrics["CV F1 Std"]),
        "original_training_records": int(len(X_train_raw)),
        "balanced_training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "feature_count": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "legitimate_url_label": 0,
        "phishing_url_label": 1,
    }
    with open(MODELS_DIR / "url_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nTrained artifacts saved to {MODELS_DIR}/ and evaluation results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()