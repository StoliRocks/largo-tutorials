"""
XGBoost Baseline for Hard Drive Failure Prediction

Traditional ML approach using gradient boosting on engineered features.
This is often surprisingly competitive with deep learning approaches.

Key advantages:
- Fast training and inference
- Interpretable feature importance
- Handles missing values natively
- Works well with tabular data

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, MODELS_DIR, XGBOOST_PARAMS


def load_data():
    """Load preprocessed training data."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get columns to use as features."""
    exclude = ['date', 'serial_number', 'model', 'failure',
               'days_to_failure', 'will_fail']
    return [col for col in df.columns if col not in exclude]


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with early stopping.

    Uses class weights to handle severe imbalance (~1% failure rate).
    """
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df['will_fail'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['will_fail'].values

    print(f"Training XGBoost on {len(X_train):,} samples...")
    print(f"Features: {len(feature_cols)}")
    print(f"Positive rate: {100*y_train.mean():.2f}%")

    # Calculate class weight from data
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = pos_weight

    model = xgb.XGBClassifier(**params, early_stopping_rounds=10)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return model, feature_cols


def evaluate_model(model, df: pd.DataFrame, feature_cols: list,
                   threshold: float = 0.5, name: str = "Test") -> dict:
    """
    Evaluate model with multiple metrics.

    For failure prediction, we typically care more about recall
    (catching failures) than precision (false alarms are acceptable).
    """
    X = df[feature_cols].values
    y_true = df['will_fail'].values

    # Get probabilities
    y_prob = model.predict_proba(X)[:, 1]

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # F2 score (weights recall higher than precision)
    f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-9)

    # AUC (threshold-independent)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    print(f"\n{name} Set Results (threshold={threshold}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  F2:        {f2:.4f}")
    print(f"  AUC:       {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"    FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'auc': auc,
        'confusion_matrix': cm
    }


def find_optimal_threshold(model, val_df: pd.DataFrame,
                          feature_cols: list) -> float:
    """
    Find threshold that maximizes F2 score on validation set.

    F2 weights recall twice as much as precision, appropriate
    for failure prediction where missing a failure is costly.
    """
    X = val_df[feature_cols].values
    y_true = val_df['will_fail'].values

    y_prob = model.predict_proba(X)[:, 1]

    best_f2 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-9)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    print(f"\nOptimal threshold: {best_threshold:.2f} (F2={best_f2:.4f})")
    return best_threshold


def plot_feature_importance(model, feature_cols: list, top_n: int = 20):
    """Plot top N most important features."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.barh(range(top_n), importance[indices][::-1])
    plt.yticks(range(top_n), [feature_cols[i] for i in indices][::-1])
    plt.xlabel("Importance")
    plt.tight_layout()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(MODELS_DIR / "xgboost_feature_importance.png", dpi=150)
    plt.close()

    print(f"\nTop 10 Features:")
    for i in indices[:10]:
        print(f"  {feature_cols[i]}: {importance[i]:.4f}")


def plot_precision_recall_curve(model, test_df: pd.DataFrame, feature_cols: list):
    """Plot precision-recall curve."""
    X = test_df[feature_cols].values
    y_true = test_df['will_fail'].values

    y_prob = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (XGBoost)')
    plt.grid(True, alpha=0.3)

    # Mark a few threshold points
    for thresh in [0.1, 0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds - thresh))
        plt.scatter([recall[idx]], [precision[idx]], s=100, zorder=5)
        plt.annotate(f't={thresh}', (recall[idx], precision[idx]),
                    textcoords="offset points", xytext=(5, 5))

    plt.savefig(MODELS_DIR / "xgboost_pr_curve.png", dpi=150)
    plt.close()


def main():
    """Train and evaluate XGBoost model."""
    print("="*60)
    print("XGBoost Baseline - Hard Drive Failure Prediction")
    print("="*60)

    # Load data
    train_df, val_df, test_df = load_data()

    # Train model
    model, feature_cols = train_xgboost(train_df, val_df)

    # Find optimal threshold
    threshold = find_optimal_threshold(model, val_df, feature_cols)

    # Evaluate on test set
    results = evaluate_model(model, test_df, feature_cols, threshold, "Test")

    # Visualizations
    plot_feature_importance(model, feature_cols)
    plot_precision_recall_curve(model, test_df, feature_cols)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols}, f)

    print(f"\nModel saved to {model_path}")

    # Summary
    print("\n" + "="*60)
    print("XGBoost Training Complete")
    print("="*60)
    print(f"Best threshold: {threshold:.2f}")
    print(f"Test F2 Score: {results['f2']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")

    return model, results


if __name__ == "__main__":
    main()
