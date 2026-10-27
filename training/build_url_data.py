"""
build_url_data.py

Reads the raw PhiUSIIL dataset (only the URL text and its label), extracts the
18 features for every URL using features/url_features.py, and saves the result
as a table that the training script can read.

Run from the project root:
    python training/build_url_data.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Running "python training/build_url_data.py" only searches the training folder for imports.
# This line adds the project root so that "from features..." works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.url_features import FEATURE_COLS, get_url_features

SOURCE_PATH = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUT_PATH = Path("data/Live_Compatible_URL_Features.csv")


def load_raw_dataset(filepath):
    """Load just the URL and label columns and tidy them up."""
    # The raw file has many columns, but we only need these two.
    df = pd.read_csv(filepath, usecols=["URL", "label"])

    # Throw away rows where the URL or the label is missing.
    df = df.dropna(subset=["URL", "label"]).copy()

    # Remove stray spaces, then drop any URL that is empty after that.
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""]

    # If the same URL appears twice with the same label, keep only one copy.
    df = df.drop_duplicates(subset=["URL", "label"]).reset_index(drop=True)
    return df


def build_feature_dataset(df):
    """Run the feature extractor on every URL. Returns the table and any failures."""
    records = []
    failed = []

    for idx, (url, label) in enumerate(zip(df["URL"], df["label"])):
        try:
            # The same function the web app uses, so training and live
            # predictions are built the same way.
            row = get_url_features(url).iloc[0].to_dict()

            # In PhiUSIIL, label 1 means legitimate and 0 means phishing.
            # We flip it so that target 1 = phishing, which is easier to read.
            row["target"] = 1 - int(label)
            records.append(row)
        except Exception as err:
            # One bad URL should not stop the whole run, so note it and move on.
            failed.append({"index": idx, "url": url, "error": str(err)})

        # Progress message, because this takes a couple of minutes.
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1:,} / {len(df):,} URLs")

    table = pd.DataFrame(records, columns=FEATURE_COLS + ["target"])

    # Drop broken rows, then drop rows whose 18 feature values AND target are
    # exactly the same as another row.
    table = table.dropna().drop_duplicates().reset_index(drop=True)
    return table, failed


def main():
    print(f"Loading raw dataset from {SOURCE_PATH}...")
    df = load_raw_dataset(SOURCE_PATH)
    print(f"Found {len(df):,} valid URL records.")

    start = time.time()
    table, failed = build_feature_dataset(df)
    minutes = (time.time() - start) / 60

    table.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFeature extraction finished in {minutes:.2f} minutes.")
    print(f"Saved dataset to {OUTPUT_PATH} (Shape: {table.shape})")
    print("Class distribution (target 1 = phishing):")
    print(table["target"].value_counts().sort_index())

    # Save the list of URLs that failed, if there were any.
    if failed:
        error_file = Path("results/failed_urls.csv")
        error_file.parent.mkdir(exist_ok=True)
        pd.DataFrame(failed).to_csv(error_file, index=False)
        print(f"Logged {len(failed)} failed URLs to {error_file}")


if __name__ == "__main__":
    main()
    