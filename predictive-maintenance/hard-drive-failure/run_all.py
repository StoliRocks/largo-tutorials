#!/usr/bin/env python3
"""
Hard Drive Failure Prediction - Full Pipeline Runner

Runs the complete pipeline:
1. Download and preprocess Backblaze data
2. Train XGBoost baseline
3. Train LSTM model
4. Train Transformer model
5. Compare all models

Usage:
    python run_all.py           # Run everything
    python run_all.py --quick   # Quick mode (subset of data)
    python run_all.py --skip-download  # Skip data download

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Hard Drive Failure Prediction Pipeline"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with subset of data')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download (use existing)')
    parser.add_argument('--model', choices=['xgboost', 'lstm', 'transformer', 'all'],
                        default='all', help='Which model to train')
    args = parser.parse_args()

    print("="*60)
    print("Hard Drive Failure Prediction Pipeline")
    print("="*60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Model: {args.model}")
    print("="*60)

    # Step 1: Data Pipeline
    if not args.skip_download:
        print("\n[1/5] Running Data Pipeline...")
        from data_pipeline import main as run_data_pipeline
        run_data_pipeline()
    else:
        print("\n[1/5] Skipping data download (--skip-download)")

    # Step 2: XGBoost
    if args.model in ['xgboost', 'all']:
        print("\n[2/5] Training XGBoost...")
        from model_xgboost import main as run_xgboost
        run_xgboost()
    else:
        print("\n[2/5] Skipping XGBoost")

    # Step 3: LSTM
    if args.model in ['lstm', 'all']:
        print("\n[3/5] Training LSTM...")
        from model_lstm import main as run_lstm
        run_lstm()
    else:
        print("\n[3/5] Skipping LSTM")

    # Step 4: Transformer
    if args.model in ['transformer', 'all']:
        print("\n[4/5] Training Transformer...")
        from model_transformer import main as run_transformer
        run_transformer()
    else:
        print("\n[4/5] Skipping Transformer")

    # Step 5: Compare
    if args.model == 'all':
        print("\n[5/5] Comparing Models...")
        from compare_models import main as run_comparison
        run_comparison()
    else:
        print("\n[5/5] Skipping comparison (single model mode)")

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nOutputs:")
    print("  - data/processed/    : Preprocessed datasets")
    print("  - models/            : Trained model weights")
    print("  - models/*.png       : Visualizations")
    print("  - models/comparison_report.md : Summary report")


if __name__ == "__main__":
    main()
