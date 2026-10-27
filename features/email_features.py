"""
email_features.py

Text cleaning for the email classifier. The cleaned text is later turned
into numbers by the TF-IDF step in the training script.
"""

import re


def clean_text(text):
    """Turn a raw email body into plain lower-case words only."""
    # Make sure we always work with a string (some cells can be empty) in lower case.
    text = str(text).lower()

    # Links are mostly random characters, so they add noise instead of meaning.
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Same for email addresses.
    text = re.sub(r"\S+@\S+", " ", text)

    # Some emails contain HTML; remove the tags like <br> or <div>.
    text = re.sub(r"<.*?>", " ", text)

    # Numbers rarely help the model, so remove them.
    text = re.sub(r"\d+", " ", text)

    # Keep letters only. Punctuation and symbols become spaces.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse repeated spaces into one and trim the ends.
    return re.sub(r"\s+", " ", text).strip()