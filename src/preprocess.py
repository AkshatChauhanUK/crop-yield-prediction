"""
preprocess.py
--------------
Cleans datafile (1).csv -> crop economics & yield data
Source columns (as seen in raw file):
    Crop
    State
    Cost of Cultivation ('/Hectare) A2+FL
    Cost of Cultivation ('/Hectare) C2
    Cost of Production ('/Quintal) C2
    Yield (Quintal/ Hectare)

Run:
    python src/preprocess.py --input data/datafile_1.csv --output data/cleaned_crop_data.csv
"""

import argparse
import re
import pandas as pd
import numpy as np


def clean_column_name(col: str) -> str:
    col = col.strip()
    col = col.replace("'", "")
    col = re.sub(r"[()/]", " ", col)
    col = re.sub(r"\s+", " ", col)
    col = col.strip().lower().replace(" ", "_")
    return col


def load_raw(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path} with common encodings")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [clean_column_name(c) for c in df.columns]
    print("Cleaned columns:", list(df.columns))

    for col in ["crop", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.title()

    numeric_cols = [c for c in df.columns if c not in ("crop", "state")]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    yield_col = next((c for c in df.columns if "yield" in c), None)
    if yield_col:
        before = len(df)
        df = df.dropna(subset=[yield_col])
        print(f"Dropped {before - len(df)} rows with missing {yield_col}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nRemaining missing values:")
        print(missing)
    else:
        print("\nNo missing values remaining.")

    for col in numeric_cols:
        if col in df.columns:
            n_bad = (df[col] <= 0).sum()
            if n_bad > 0:
                print(f"Warning: {n_bad} rows have non-positive values in '{col}'")

    df = df.reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw datafile (1).csv")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    raw = load_raw(args.input)
    print(f"Loaded raw data: {raw.shape[0]} rows, {raw.shape[1]} columns")

    cleaned = clean(raw)
    print(f"\nFinal cleaned data: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")

    cleaned.to_csv(args.output, index=False)
    print(f"Saved cleaned data to {args.output}")


if __name__ == "__main__":
    main()