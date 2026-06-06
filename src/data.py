"""Data loading, cleaning, removing duplicate, and splitting.

- Drop index 
- Drop nulls
- normalise whitespace
- remove conflicting-label texts 
- deduplicate
- split dataset into train/val/test
"""

import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "mental_health_data.csv"
SEED = 42
VAL_SIZE = TEST_SIZE = 0.10

LABELS = ["Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality disorder"]

def normalize(text: str) -> str:
    """Collapse internal whitespace and strip. Keeps original casing."""
    return re.compile(r"\s+").sub(" ", str(text)).strip()

def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]], dict]:
    """Clean the raw frame and return (clean_df, accounting, stats).
    """
    accounting: list[tuple[str, int]] = [("raw", len(df))]
    # Drop the redundant index column.
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Drop null statements.
    df = df.dropna(subset=["statement"]).copy()
    accounting.append(("drop null", len(df)))
    
    # Normalise whitespace, then drop anything empty after normalisation.
    df["statement"] = df["statement"].map(normalize)
    df = df[df["statement"].str.len() > 0].copy()
    accounting.append(("drop empty after normalise", len(df)))

    # Dedup/conflict key
    df["_key"] = df["statement"].str.lower()

    # Remove conflicting-label texts before de-duplication. 
    n_status = df.groupby("_key")["status"].transform("nunique")
    n_conflict_texts = int(df.loc[n_status > 1, "_key"].nunique())
    df = df[n_status == 1].copy()
    accounting.append(("drop conflicting-label texts", len(df)))

    # De-duplicate
    n_before = len(df)
    df = df.drop_duplicates(subset="_key", keep="first").copy()
    accounting.append(("dedup duplicate texts", len(df)))
    n_dups = n_before - len(df)

    df = df.drop(columns="_key").reset_index(drop=True)
    stats = {"n_conflict": n_conflict_texts, "n_duplicate": n_dups}
    return df, accounting, stats

def _train_val_test_split(df: pd.DataFrame, seed: int) -> tuple:
    train, temp = train_test_split(df, test_size=VAL_SIZE + TEST_SIZE, stratify=df["status"], random_state=seed)
    val, test = train_test_split(temp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), stratify=temp["status"],random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))

def get_splits(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, _, _ = clean(load_raw())
    return _train_val_test_split(df, seed)

def main() -> None:
    raw = load_raw()
    df, accounting, stats = clean(raw)

    print("Cleaning steps and row counts:")
    prev = None
    for step, n in accounting:
        delta = "" if prev is None else f"  ({n - prev:+d})"
        print(f"  {step} {n,}{delta}")
        prev = n
    print(f"\nConflicts dropped: {stats['n_conflict']}")
    print(f"Duplicates dropped: {stats['n_duplicate']}")

    train, val, test = _train_val_test_split(df, SEED)
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"  {name} {len(d)}")


if __name__ == "__main__":
    main()
