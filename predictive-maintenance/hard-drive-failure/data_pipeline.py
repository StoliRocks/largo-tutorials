"""
Data Pipeline for Backblaze Hard Drive Dataset

Downloads, preprocesses, and prepares data for failure prediction models.
Handles the massive scale of the dataset efficiently using chunked processing.

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import os
import zipfile
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from config import (
    RAW_DIR, PROCESSED_DIR, BACKBLAZE_BASE_URL, DEFAULT_DATASET,
    KEY_SMART_ATTRS, SEQUENCE_LENGTH, PREDICTION_HORIZON,
    UNDERSAMPLE_RATIO, TEST_SIZE, VAL_SIZE
)


def download_dataset(dataset_name: str = DEFAULT_DATASET, force: bool = False) -> Path:
    """
    Download Backblaze dataset if not already present.

    Args:
        dataset_name: Name of the dataset zip file (e.g., 'data_Q4_2023.zip')
        force: If True, re-download even if file exists

    Returns:
        Path to the downloaded zip file
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / dataset_name

    if zip_path.exists() and not force:
        print(f"Dataset already exists: {zip_path}")
        return zip_path

    url = f"{BACKBLAZE_BASE_URL}/{dataset_name}"
    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded to {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path) -> Path:
    """
    Extract the dataset zip file.

    Returns:
        Path to the extracted directory
    """
    extract_dir = RAW_DIR / zip_path.stem

    if extract_dir.exists():
        print(f"Already extracted: {extract_dir}")
        return extract_dir

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def load_daily_files(data_dir: Path, max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load and concatenate daily CSV files.

    The Backblaze dataset contains one CSV per day (YYYY-MM-DD.csv).
    Each row is a snapshot of one drive's SMART attributes.

    Args:
        data_dir: Directory containing the CSV files
        max_files: Limit number of files for quick experiments

    Returns:
        Combined DataFrame with all daily snapshots
    """
    csv_files = sorted(data_dir.glob("*.csv"))

    if max_files:
        csv_files = csv_files[:max_files]

    print(f"Loading {len(csv_files)} daily files...")

    dfs = []
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} drive-day records")

    return combined


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw Backblaze data.

    Steps:
    1. Parse dates
    2. Select relevant SMART columns
    3. Handle missing values
    4. Filter to drives with sufficient history
    """
    print("Preprocessing data...")

    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    # Core columns we always need
    core_cols = ['date', 'serial_number', 'model', 'failure']

    # Find which SMART columns exist in this dataset
    available_smart = [col for col in KEY_SMART_ATTRS if col in df.columns]
    print(f"Found {len(available_smart)}/{len(KEY_SMART_ATTRS)} key SMART attributes")

    # Select columns
    keep_cols = core_cols + available_smart
    df = df[keep_cols].copy()

    # Fill missing SMART values with 0 (common convention)
    for col in available_smart:
        df[col] = df[col].fillna(0)

    # Sort by drive and date
    df = df.sort_values(['serial_number', 'date']).reset_index(drop=True)

    print(f"Preprocessed: {len(df):,} records, {df['serial_number'].nunique():,} drives")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for point-in-time prediction (XGBoost).

    Features:
    - Raw SMART values
    - Rolling statistics (mean, std, delta over past N days)
    - Time-based features (power-on hours growth rate)
    """
    print("Engineering features...")

    # Find SMART columns
    smart_cols = [col for col in df.columns if col.startswith('smart_')]

    # Group by drive for rolling calculations
    grouped = df.groupby('serial_number')

    # Rolling statistics over past 7 days
    for col in tqdm(smart_cols, desc="Rolling features"):
        # 7-day rolling mean
        df[f'{col}_roll7_mean'] = grouped[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        # 7-day rolling std (volatility)
        df[f'{col}_roll7_std'] = grouped[col].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
        # Day-over-day change
        df[f'{col}_delta'] = grouped[col].diff().fillna(0)

    # Drive age (days since first appearance)
    df['drive_age_days'] = grouped.cumcount()

    # Days until failure (for training target)
    df['days_to_failure'] = np.nan

    # For drives that failed, calculate days remaining
    failed_drives = df[df['failure'] == 1]['serial_number'].unique()
    for serial in tqdm(failed_drives, desc="Computing days to failure"):
        mask = df['serial_number'] == serial
        drive_data = df.loc[mask].copy()
        failure_date = drive_data[drive_data['failure'] == 1]['date'].iloc[0]
        df.loc[mask, 'days_to_failure'] = (failure_date - drive_data['date']).dt.days

    # Binary target: will fail within PREDICTION_HORIZON days
    df['will_fail'] = (df['days_to_failure'] <= PREDICTION_HORIZON).astype(int)
    # Non-failed drives get label 0
    df['will_fail'] = df['will_fail'].fillna(0).astype(int)

    print(f"Engineered features: {len(df.columns)} total columns")

    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH) -> tuple:
    """
    Create sequences for LSTM/Transformer models.

    Each sample is a (sequence_length, num_features) tensor representing
    the past N days of SMART readings for a drive.

    Returns:
        X: (num_samples, sequence_length, num_features) array
        y: (num_samples,) binary labels
        serial_numbers: List of drive serial numbers for each sample
    """
    print(f"Creating sequences of length {sequence_length}...")

    smart_cols = [col for col in df.columns if col.startswith('smart_') and '_roll' not in col and '_delta' not in col]

    sequences = []
    labels = []
    serials = []

    for serial, group in tqdm(df.groupby('serial_number')):
        group = group.sort_values('date')

        if len(group) < sequence_length:
            continue

        values = group[smart_cols].values
        targets = group['will_fail'].values

        # Sliding window
        for i in range(sequence_length, len(group)):
            seq = values[i-sequence_length:i]
            label = targets[i]

            sequences.append(seq)
            labels.append(label)
            serials.append(serial)

    X = np.array(sequences)
    y = np.array(labels)

    print(f"Created {len(X):,} sequences")
    print(f"Positive samples (will fail): {y.sum():,} ({100*y.mean():.2f}%)")

    return X, y, serials


def time_based_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                     val_size: float = VAL_SIZE) -> tuple:
    """
    Split data by time for realistic evaluation.

    Training on past, testing on future prevents data leakage.

    Returns:
        train_df, val_df, test_df
    """
    dates = df['date'].sort_values().unique()
    n_dates = len(dates)

    test_start_idx = int(n_dates * (1 - test_size))
    val_start_idx = int(n_dates * (1 - test_size - val_size))

    test_start = dates[test_start_idx]
    val_start = dates[val_start_idx]

    train_df = df[df['date'] < val_start].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test_df = df[df['date'] >= test_start].copy()

    print(f"Time-based split:")
    print(f"  Train: {len(train_df):,} records (< {val_start.date()})")
    print(f"  Val:   {len(val_df):,} records ({val_start.date()} - {test_start.date()})")
    print(f"  Test:  {len(test_df):,} records (>= {test_start.date()})")

    return train_df, val_df, test_df


def undersample_healthy(df: pd.DataFrame, ratio: float = UNDERSAMPLE_RATIO) -> pd.DataFrame:
    """
    Undersample healthy drives to reduce class imbalance.

    Keeps all failed drives but only a fraction of healthy ones.
    Use for quick experiments; full data for final model.
    """
    failed = df[df['will_fail'] == 1]
    healthy = df[df['will_fail'] == 0]

    n_healthy_keep = int(len(healthy) * ratio)
    healthy_sampled = healthy.sample(n=n_healthy_keep, random_state=42)

    result = pd.concat([failed, healthy_sampled]).sample(frac=1, random_state=42)

    print(f"Undersampled: {len(result):,} records ({len(failed):,} failed, {n_healthy_keep:,} healthy)")

    return result


def main():
    """Run the full data pipeline."""
    # Download and extract
    zip_path = download_dataset()
    data_dir = extract_dataset(zip_path)

    # Find the actual CSV directory (may be nested)
    csv_dirs = list(data_dir.rglob("*.csv"))
    if csv_dirs:
        data_dir = csv_dirs[0].parent

    # Load data (limit files for demo)
    df = load_daily_files(data_dir, max_files=30)  # ~1 month

    # Preprocess
    df = preprocess_data(df)

    # Engineer features
    df = engineer_features(df)

    # Split by time
    train_df, val_df, test_df = time_based_split(df)

    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(PROCESSED_DIR / "train.parquet")
    val_df.to_parquet(PROCESSED_DIR / "val.parquet")
    test_df.to_parquet(PROCESSED_DIR / "test.parquet")

    print(f"\nSaved processed data to {PROCESSED_DIR}")

    # Create sequences for deep learning
    print("\nCreating sequences for LSTM/Transformer...")
    X_train, y_train, _ = create_sequences(train_df)
    X_val, y_val, _ = create_sequences(val_df)
    X_test, y_test, _ = create_sequences(test_df)

    np.savez(
        PROCESSED_DIR / "sequences.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    print(f"Saved sequences to {PROCESSED_DIR / 'sequences.npz'}")

    # Summary statistics
    print("\n" + "="*60)
    print("Data Pipeline Complete")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Unique drives: {df['serial_number'].nunique():,}")
    print(f"Failed drives: {df[df['failure']==1]['serial_number'].nunique():,}")
    print(f"Failure rate: {100*df['failure'].mean():.3f}%")
    print(f"Features: {len([c for c in df.columns if 'smart' in c])}")


if __name__ == "__main__":
    main()
