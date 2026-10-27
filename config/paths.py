"""
paths.py

All file and folder locations used by the web app are kept here, so that
a path only has to be changed in one place.
"""

from pathlib import Path

# The project root is one level above this file (config/paths.py -> project folder).
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# --- URL model files (created by train_url_model.py) ---
URL_MODEL = MODELS_DIR / "best_url_model.pkl"
URL_FEAT_COLS = MODELS_DIR / "url_feature_columns.pkl"
URL_META = MODELS_DIR / "url_model_metadata.json"

# --- Email model files (created by train_email_model.py) ---
EMAIL_MODEL = MODELS_DIR / "best_email_model.pkl"
EMAIL_VECTORIZER = MODELS_DIR / "email_tfidf_vectorizer.pkl"
EMAIL_META = MODELS_DIR / "email_model_metadata.json"

# --- Result tables shown on the analytics page ---
URL_COMPARISON = RESULTS_DIR / "url_model_comparison.csv"
EMAIL_COMPARISON = RESULTS_DIR / "email_model_comparison.csv"

# --- Result charts shown on the analytics page ---
URL_CONF_MATRIX = RESULTS_DIR / "url_best_model_confusion_matrix.png"
URL_ROC = RESULTS_DIR / "url_models_roc_curve.png"
URL_PR = RESULTS_DIR / "url_models_precision_recall_curve.png"
URL_FEAT_IMP = RESULTS_DIR / "url_feature_importance.png"

EMAIL_CONF_MATRIX = RESULTS_DIR / "email_best_model_confusion_matrix.png"
EMAIL_ROC = RESULTS_DIR / "email_models_roc_curve.png"
EMAIL_PR = RESULTS_DIR / "email_models_precision_recall_curve.png"
EMAIL_TOP_TERMS = RESULTS_DIR / "email_top_phishing_terms.png"