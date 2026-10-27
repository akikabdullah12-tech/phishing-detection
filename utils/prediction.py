"""
prediction.py

Small helper functions that turn a model's output into a phishing probability
and a risk level the user can understand.
"""

import math

import numpy as np


def sigmoid(x):
    """Squash any number into the range 0 to 1."""
    # Clip first so that very large scores do not cause an overflow error.
    clipped = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + math.exp(-clipped))


def get_phishing_prob(model, sample):
    """
    Return the probability (0 to 1) that the sample is phishing.

    XGBoost, Random Forest and Logistic Regression give probabilities directly.
    The linear SVM does not, so its score is passed through a sigmoid instead.
    That value is only an approximation and not a calibrated probability.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)
        classes = list(model.classes_)
        # Class 1 means phishing in both of our datasets.
        if 1 in classes:
            idx = classes.index(1)
            return float(probabilities[0][idx])

    if hasattr(model, "decision_function"):
        score = model.decision_function(sample)
        if isinstance(score, np.ndarray):
            score = score.ravel()[0]
        return sigmoid(score)

    # Last resort: just use the predicted label (0 or 1).
    return float(model.predict(sample)[0])


def risk_label(probability):
    """Turn a probability into a simple risk level."""
    # These bands are fixed values chosen for display. They were not tuned on data.
    if probability >= 0.85:
        return "Critical Risk"
    if probability >= 0.65:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"