"""
Create a stratified 8k sample from the fraud training CSV (read from zip in data/raw/).
Preserves time order and approximate fraud ratio (~1.7%).
"""

import os
import zipfile
from pathlib import Path

import pandas as pd


def main():
    raw_dir = Path("data/raw")
    zip_paths = list(raw_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No .zip file found in {raw_dir}")

    with zipfile.ZipFile(zip_paths[0], "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No .csv file found inside {zip_paths[0]}")
        csv_name = csv_names[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    # Detect fraud column (e.g. is_fraud)
    fraud_col = None
    for c in df.columns:
        if "fraud" in c.lower():
            fraud_col = c
            break
    if fraud_col is None:
        raise ValueError("No column with 'fraud' in name found. Columns: " + ", ".join(df.columns))

    # Original stats (support 0/1 or bool)
    n_orig = len(df)
    fraud_count_orig = int((df[fraud_col].astype(int) == 1).sum())
    rate_orig = 100 * fraud_count_orig / n_orig if n_orig else 0

    print("Original dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Fraud count: {fraud_count_orig}")
    print(f"  Fraud rate: {rate_orig:.2f}%")

    # Sort by time to preserve order
    time_col = "trans_date_trans_time"
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Columns: " + ", ".join(df.columns))
    df = df.sort_values(time_col).reset_index(drop=True)

    # Stratified sample: exactly 8000 rows, ~1.7% fraud -> 136 fraud, 7864 non-fraud
    target_fraud = 136
    target_non_fraud = 7864
    target_total = 8000

    fraud_mask = (df[fraud_col].astype(int) == 1)
    fraud_df = df[fraud_mask]
    non_fraud_df = df[~fraud_mask]

    n_fraud_available = len(fraud_df)
    n_non_fraud_available = len(non_fraud_df)

    n_fraud_sample = min(target_fraud, n_fraud_available)
    n_non_fraud_sample = min(target_non_fraud, n_non_fraud_available)
    # If one class is short, take more from the other to reach exactly 8000
    if n_fraud_sample + n_non_fraud_sample < target_total:
        short = target_total - (n_fraud_sample + n_non_fraud_sample)
        if n_non_fraud_available > n_non_fraud_sample:
            n_non_fraud_sample = min(n_non_fraud_sample + short, n_non_fraud_available)
        short = target_total - (n_fraud_sample + n_non_fraud_sample)
        if short > 0 and n_fraud_available > n_fraud_sample:
            n_fraud_sample = min(n_fraud_sample + short, n_fraud_available)

    fraud_sampled = fraud_df.sample(n=n_fraud_sample, random_state=42)
    non_fraud_sampled = non_fraud_df.sample(n=n_non_fraud_sample, random_state=42)
    sample_df = pd.concat([fraud_sampled, non_fraud_sampled], ignore_index=True)
    sample_df = sample_df.sort_values(time_col).reset_index(drop=True)

    # Exactly 8000 rows
    sample_df = sample_df.head(target_total)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/transactions_8k.csv"
    sample_df.to_csv(out_path, index=False)

    n_final = len(sample_df)
    fraud_count_final = int((sample_df[fraud_col].astype(int) == 1).sum())
    rate_final = 100 * fraud_count_final / n_final if n_final else 0

    print("\nSample (saved to data/processed/transactions_8k.csv):")
    print(f"  Shape: {sample_df.shape}")
    print(f"  Fraud count: {fraud_count_final}")
    print(f"  Fraud rate: {rate_final:.2f}%")
    print("\nFirst 5 rows:")
    print(sample_df.head().to_string())


if __name__ == "__main__":
    main()
