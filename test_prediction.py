"""
test_prediction.py

A quick check that the saved models can be loaded and used on new input.
It does not need the web app. It tests the same steps the app uses:
extract features -> run the model -> get a probability and a risk level.

Run from the project root:
    python test_prediction.py
"""

import joblib

from config import paths
from features.email_features import clean_text
from features.url_features import get_url_features
from utils.prediction import get_phishing_prob, risk_label


def check_url(model, feature_cols, url):
    """Run one URL through the URL model and print the result."""
    # Keep the column order exactly as it was during training.
    features = get_url_features(url)[feature_cols]
    prob = get_phishing_prob(model, features)

    print(f"URL: {url}")
    print(f"  Phishing probability : {prob:.4f}")
    print(f"  Risk level           : {risk_label(prob)}")
    print(f"  Verdict              : {'PHISHING' if prob >= 0.5 else 'LEGITIMATE'}\n")


def check_email(model, vectorizer, text):
    """Run one email through the email model and print the result."""
    # Clean the text first, then turn it into TF-IDF numbers.
    sample = vectorizer.transform([clean_text(text)])
    prob = get_phishing_prob(model, sample)

    print(f"Email: {text[:70]}...")
    print(f"  Phishing probability : {prob:.4f}")
    print(f"  Risk level           : {risk_label(prob)}")
    print(f"  Verdict              : {'PHISHING' if prob >= 0.5 else 'SAFE'}\n")


def main():
    # Load the models saved by the two training scripts.
    url_model = joblib.load(paths.URL_MODEL)
    url_cols = joblib.load(paths.URL_FEAT_COLS)
    email_model = joblib.load(paths.EMAIL_MODEL)
    email_vectorizer = joblib.load(paths.EMAIL_VECTORIZER)

    print("=== URL model test ===\n")
    check_url(url_model, url_cols, "http://paypa1-secure.login.verify-account.com/signin?id=123&next=home")
    check_url(url_model, url_cols, "https://www.bbc.co.uk/news")

    print("=== Email model test ===\n")
    check_email(
        email_model, email_vectorizer,
        "Dear customer, your account has been suspended. Verify your password immediately "
        "by clicking the link below to avoid permanent closure.",
    )
    check_email(
        email_model, email_vectorizer,
        "Hi team, please find attached the minutes of yesterday's meeting. "
        "Let me know if you have any questions before Friday.",
    )


if __name__ == "__main__":
    main()