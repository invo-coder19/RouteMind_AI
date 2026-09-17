"""
Split labeled_prompts.csv into train (80%) and test (20%) splits.

Run once after modifying the labeled dataset:
    python data/split_dataset.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(__file__).parent
SRC = DATA_DIR / "labeled_prompts.csv"
TRAIN_OUT = DATA_DIR / "train.csv"
TEST_OUT = DATA_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SRC)
    required = {"prompt_text", "complexity_label"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    counts = df["complexity_label"].value_counts()
    print("Full dataset label distribution:")
    for label, n in counts.items():
        print(f"  {label}: {n} ({n / len(df):.0%})")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["complexity_label"],
    )

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"\nSaved {len(train_df)} rows -> {TRAIN_OUT.name}")
    print(f"Saved {len(test_df)} rows  -> {TEST_OUT.name}")


if __name__ == "__main__":
    main()
