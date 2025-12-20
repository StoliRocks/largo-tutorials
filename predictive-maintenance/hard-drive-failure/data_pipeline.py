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

# Try to use cuDF for GPU acceleration
try:
    import cudf
    USE_CUDF = True
    print("cuDF available - using GPU acceleration")
except ImportError:
    USE_CUDF = False
    print("cuDF not available - using pandas (slower)")

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
    read_fn = cudf.read_csv if USE_CUDF else pd.read_csv
    concat_fn = cudf.concat if USE_CUDF else pd.concat

    for csv_file in tqdm(csv_files):
        # Some Backblaze files have encoding issues, try utf-8 first then latin-1
        try:
            df = read_fn(csv_file)
        except (UnicodeDecodeError, Exception):
            df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')
            if USE_CUDF:
                df = cudf.from_pandas(df)
        dfs.append(df)

    combined = concat_fn(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} drive-day records")

    return combined


def preprocess_data(df):
    """
    Clean and preprocess the raw Backblaze data.
    Works with both cuDF and pandas DataFrames.

    Steps:
    1. Parse dates
    2. Select relevant SMART columns
    3. Handle missing values
    4. Filter to drives with sufficient history
    """
    print("Preprocessing data...")

    # Parse date - use cuDF or pandas as appropriate
    if USE_CUDF:
        df['date'] = cudf.to_datetime(df['date'])
    else:
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


def engineer_features(df) -> pd.DataFrame:
    """
    Create features for point-in-time prediction (XGBoost).
    Uses cuDF for GPU acceleration when available.

    Features:
    - Raw SMART values
    - Rolling statistics (mean, std, delta over past N days)
    - Time-based features (power-on hours growth rate)
    """
    print("Engineering features" + (" (GPU accelerated)" if USE_CUDF else "") + "...")

    # Find SMART columns
    smart_cols = [col for col in df.columns if col.startswith('smart_')]

    if USE_CUDF:
        # GPU-accelerated feature engineering
        print(f"Computing rolling features for {len(smart_cols)} columns on GPU...")

        # Sort by drive and date for correct rolling windows
        df = df.sort_values(['serial_number', 'date']).reset_index(drop=True)

        # Compute all rolling features at once using cuDF
        for col in tqdm(smart_cols, desc="Rolling features (GPU)"):
            # Group by serial_number and compute rolling stats
            grouped = df.groupby('serial_number')[col]

            # Rolling mean (using shift to make it look-back only)
            df[f'{col}_roll7_mean'] = grouped.rolling(7, min_periods=1).mean().reset_index(drop=True)

            # Rolling std
            df[f'{col}_roll7_std'] = grouped.rolling(7, min_periods=1).std().reset_index(drop=True)
            df[f'{col}_roll7_std'] = df[f'{col}_roll7_std'].fillna(0)

            # Day-over-day change
            df[f'{col}_delta'] = grouped.diff().reset_index(drop=True).fillna(0)

        # Drive age
        df['drive_age_days'] = df.groupby('serial_number').cumcount()

    else:
        # CPU fallback with pandas
        grouped = df.groupby('serial_number')

        for col in tqdm(smart_cols, desc="Rolling features"):
            df[f'{col}_roll7_mean'] = grouped[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{col}_roll7_std'] = grouped[col].transform(
                lambda x: x.rolling(7, min_periods=1).std().fillna(0)
            )
            df[f'{col}_delta'] = grouped[col].diff().fillna(0)

        df['drive_age_days'] = grouped.cumcount()

    # Days until failure - vectorized approach
    print("Computing days to failure...")
    df['days_to_failure'] = -1  # Use -1 for non-failed

    # Get failure dates for each drive that failed
    if USE_CUDF:
        # Convert to pandas for this part (complex indexing)
        df_pd = df.to_pandas()
    else:
        df_pd = df

    failed_mask = df_pd['failure'] == 1
    if failed_mask.any():
        # Get failure date for each failed drive
        failure_dates = df_pd.loc[failed_mask, ['serial_number', 'date']].copy()
        failure_dates = failure_dates.rename(columns={'date': 'failure_date'})

        # Merge back to get failure date for all records of failed drives
        df_pd = df_pd.merge(failure_dates, on='serial_number', how='left')

        # Compute days to failure where we have a failure date
        has_failure_date = df_pd['failure_date'].notna()
        df_pd.loc[has_failure_date, 'days_to_failure'] = (
            df_pd.loc[has_failure_date, 'failure_date'] - df_pd.loc[has_failure_date, 'date']
        ).dt.days

        # Drop the temporary column
        df_pd = df_pd.drop(columns=['failure_date'])

    # Binary target: will fail within PREDICTION_HORIZON days
    df_pd['will_fail'] = ((df_pd['days_to_failure'] >= 0) &
                          (df_pd['days_to_failure'] <= PREDICTION_HORIZON)).astype(int)

    print(f"Engineered features: {len(df_pd.columns)} total columns")

    # Return as pandas DataFrame for compatibility
    return df_pd


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


def normalize_sequences(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Normalize sequences using StandardScaler fitted on training data.

    SMART attributes can have wildly different scales (0 to trillions),
    which causes numerical issues in neural networks. Normalization is critical.

    Returns:
        Normalized (X_train, X_val, X_test) with values clipped to [-10, 10]
    """
    from sklearn.preprocessing import StandardScaler

    n_samples, seq_len, n_features = X_train.shape

    # Flatten for normalization
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features) if len(X_val) > 0 else X_val
    X_test_flat = X_test.reshape(-1, n_features) if len(X_test) > 0 else X_test

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

    # Transform val/test
    if len(X_val) > 0:
        X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape)
    else:
        X_val_norm = X_val

    if len(X_test) > 0:
        X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)
    else:
        X_test_norm = X_test

    # Clip extreme values to prevent numerical issues
    X_train_norm = np.clip(X_train_norm, -10, 10)
    X_val_norm = np.clip(X_val_norm, -10, 10) if len(X_val_norm) > 0 else X_val_norm
    X_test_norm = np.clip(X_test_norm, -10, 10) if len(X_test_norm) > 0 else X_test_norm

    print(f"Normalized sequences: range [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")

    return X_train_norm.astype(np.float32), X_val_norm.astype(np.float32), X_test_norm.astype(np.float32)


def undersample_sequences(X: np.ndarray, y: np.ndarray,
                          ratio: float = UNDERSAMPLE_RATIO) -> tuple:
    """
    Undersample negative sequences to reduce class imbalance.

    Keeps all positive samples but only a fraction of negatives.
    This speeds up training and improves class balance.

    Args:
        X: Sequence array (n_samples, seq_len, features)
        y: Labels array (n_samples,)
        ratio: Fraction of negatives to keep

    Returns:
        Undersampled (X, y)
    """
    pos_mask = y == 1
    neg_mask = y == 0

    np.random.seed(42)
    neg_idx = np.where(neg_mask)[0]
    keep_neg = np.random.choice(neg_idx, size=int(len(neg_idx) * ratio), replace=False)
    keep_idx = np.concatenate([np.where(pos_mask)[0], keep_neg])
    np.random.shuffle(keep_idx)

    X_sub = X[keep_idx]
    y_sub = y[keep_idx]

    print(f"Undersampled: {len(X_sub):,} samples ({pos_mask.sum():,} positive, {len(keep_neg):,} negative)")

    return X_sub, y_sub


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

    # Find the actual CSV directory (may be nested, skip __MACOSX)
    csv_files = [f for f in data_dir.rglob("*.csv") if "__MACOSX" not in str(f)]
    if csv_files:
        data_dir = csv_files[0].parent
        print(f"Found CSV directory: {data_dir}")

    # Load data (10 days - balances sequence needs with GPU memory)
    df = load_daily_files(data_dir, max_files=7)

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

    # Handle case where val/test may be empty with limited data
    if len(X_val) == 0 or len(X_test) == 0:
        print("Warning: Val/Test sequences empty. Using train split.")
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42,
            stratify=y_train if y_train.sum() > 1 else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=y_temp if y_temp.sum() > 1 else None
        )

    # Normalize sequences (critical for neural networks)
    print("\nNormalizing sequences...")
    X_train, X_val, X_test = normalize_sequences(X_train, X_val, X_test)

    # Undersample for faster training (optional)
    print("\nUndersampling for training efficiency...")
    X_train, y_train = undersample_sequences(X_train, y_train)

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
