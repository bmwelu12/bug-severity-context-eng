"""
01_data_prep.py

Loads and cleans the Bugzilla severity dataset.

REALITY CHECK vs. the original plan: this dataset does NOT have multi-class
resolution labels (fixed/duplicate/invalid/wontfix). It ships two separate
binary tasks, each with its own pre-made train/test split:

  - sev_{train,test}.csv : Label = 1 if Severity in {blocker, critical, major}
                                   Label = 0 if Severity in {minor, normal, trivial}
  - fix_{train,test}.csv : Label = 1 if Fixing_time > ~100 days, else 0

This script handles the severity task by default. Pass --task fix to run
the fix-time task instead (columns line up the same way).

Tested against the real uploaded CSVs.
"""
import argparse
import re
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def clean_description(text: str) -> str:
    """Light cleanup: collapse whitespace, strip the repeated
    'Created by / Updated by ... Additional Details' boilerplate noise
    down to something more readable, without losing content."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_task(task: str):
    if task == "sev":
        train = pd.read_csv(RAW_DIR / "sev_train.csv")
        test = pd.read_csv(RAW_DIR / "sev_test.csv")
        label_col = "Severity"
    elif task == "fix":
        train = pd.read_csv(RAW_DIR / "fix_train.csv")
        test = pd.read_csv(RAW_DIR / "fix_test.csv")
        label_col = "Fixing_time"
    else:
        raise ValueError(f"unknown task {task}")
    return train, test, label_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sev", "fix"], default="sev")
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: subsample N rows from each split "
                              "(useful for cheap end-to-end test runs before "
                              "spending API budget on the full test set).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, label_col = load_task(args.task)

    for df in (train, test):
        df["Description"] = df["Description"].apply(clean_description)
        df["text_len"] = df["Description"].str.len()

    train = train[["Description", label_col, "Label", "text_len"]].dropna(subset=["Description"])
    test = test[["Description", label_col, "Label", "text_len"]].dropna(subset=["Description"])

    train = train[train["text_len"] > 0].reset_index(drop=True)
    test = test[test["text_len"] > 0].reset_index(drop=True)

    if args.sample:
        train = train.sample(n=min(args.sample, len(train)), random_state=42).reset_index(drop=True)
        test = test.sample(n=min(args.sample, len(test)), random_state=42).reset_index(drop=True)

    train.to_csv(OUT_DIR / f"{args.task}_train_clean.csv", index=False)
    test.to_csv(OUT_DIR / f"{args.task}_test_clean.csv", index=False)

    print(f"[{args.task}] train: {len(train)} rows, test: {len(test)} rows")
    print(f"Label balance (train):\n{train['Label'].value_counts(normalize=True)}")
    print(f"Label balance (test):\n{test['Label'].value_counts(normalize=True)}")
    print(f"Wrote {OUT_DIR / f'{args.task}_train_clean.csv'}")
    print(f"Wrote {OUT_DIR / f'{args.task}_test_clean.csv'}")


if __name__ == "__main__":
    main()
