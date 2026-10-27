"""
clean_emails.py

Reads the raw email dataset, removes duplicates, converts the labels to numbers
and cleans the text of every email. The result is saved as a new CSV file.

Run from the project root:
    python training/clean_emails.py
"""

import sys
from pathlib import Path

import pandas as pd

# Running "python training/clean_emails.py" only searches the training folder for imports.
# This line adds the project root so that "from features..." works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.email_features import clean_text

RAW_PATH = Path("data/Phishing_Email.csv")
OUTPUT_PATH = Path("data/Cleaned_Phishing_Email.csv")


def preprocess_emails(raw_file, output_file):
    """Load the raw emails, clean them and save the result."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_file}")

    print(f"Loading raw emails from {raw_file}...")
    df = pd.read_csv(raw_file)

    # The file has an extra index column that we do not need.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Initial dataset shape: {df.shape}")

    # Remove rows that are exact copies of another row.
    initial_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {initial_count - len(df)} duplicate rows. ({len(df):,} remaining)")

    # Turn the text labels into numbers: 0 = safe email, 1 = phishing email.
    label_map = {"Safe Email": 0, "Phishing Email": 1}
    df["Email Type"] = df["Email Type"].map(label_map).fillna(df["Email Type"])
    print("Class label counts:", df["Email Type"].value_counts().to_dict())

    # Clean every email body with the function from features/email_features.py.
    print("Cleaning email text content...")
    df["Clean_Text"] = df["Email Text"].apply(clean_text)

    # Save the result for the training script.
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset to {output_file} (Shape: {df.shape})")


def main():
    preprocess_emails(RAW_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()