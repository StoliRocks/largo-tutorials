"""
Model Comparison for Hard Drive Failure Prediction

Compares XGBoost, LSTM, and Transformer approaches on the same test set.
Generates visualizations and a summary report.

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from config import PROCESSED_DIR, MODELS_DIR
from model_xgboost import load_data, get_feature_columns
from model_lstm import LSTMClassifier, load_sequences
from model_transformer import TransformerClassifier
from config import LSTM_PARAMS, TRANSFORMER_PARAMS


def load_all_models():
    """Load all trained models."""
    models = {}

    # XGBoost
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    if xgb_path.exists():
        with open(xgb_path, 'rb') as f:
            xgb_data = pickle.load(f)
        models['xgboost'] = {
            'model': xgb_data['model'],
            'feature_cols': xgb_data['feature_cols']
        }
        print("Loaded XGBoost model")

    # LSTM
    lstm_path = MODELS_DIR / "lstm_best.pt"
    if lstm_path.exists():
        data = np.load(PROCESSED_DIR / "sequences.npz")
        input_dim = data['X_test'].shape[2]
        lstm = LSTMClassifier(input_dim, **LSTM_PARAMS)
        lstm.load_state_dict(torch.load(lstm_path))
        lstm.eval()
        models['lstm'] = {'model': lstm}
        print("Loaded LSTM model")

    # Transformer
    tf_path = MODELS_DIR / "transformer_best.pt"
    if tf_path.exists():
        data = np.load(PROCESSED_DIR / "sequences.npz")
        input_dim = data['X_test'].shape[2]
        transformer = TransformerClassifier(input_dim, **TRANSFORMER_PARAMS)
        transformer.load_state_dict(torch.load(tf_path))
        transformer.eval()
        models['transformer'] = {'model': transformer}
        print("Loaded Transformer model")

    return models


def evaluate_xgboost(model_dict, test_df):
    """Get predictions from XGBoost."""
    model = model_dict['model']
    feature_cols = model_dict['feature_cols']

    X_test = test_df[feature_cols].values
    y_true = test_df['will_fail'].values

    start_time = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time

    return y_true, y_prob, inference_time


def evaluate_pytorch(model, X_test, y_test, device):
    """Get predictions from PyTorch model (LSTM or Transformer)."""
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    start_time = time.time()
    with torch.no_grad():
        y_prob = model(X_tensor).cpu().numpy()
    inference_time = time.time() - start_time

    return y_test, y_prob, inference_time


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute all metrics for a model."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-9)

    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_curve, prec_curve)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def plot_comparison(results, save_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(results.keys())
    colors = {'xgboost': '#2ecc71', 'lstm': '#3498db', 'transformer': '#e74c3c'}

    # 1. ROC Curves
    ax = axes[0, 0]
    for name in models:
        r = results[name]
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        ax.plot(fpr, tpr, color=colors.get(name, 'gray'),
                label=f"{name.upper()} (AUC={r['metrics']['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Precision-Recall Curves
    ax = axes[0, 1]
    for name in models:
        r = results[name]
        prec, rec, _ = precision_recall_curve(r['y_true'], r['y_prob'])
        ax.plot(rec, prec, color=colors.get(name, 'gray'),
                label=f"{name.upper()} (AUC={r['metrics']['pr_auc']:.3f})", linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Metric Comparison Bar Chart
    ax = axes[1, 0]
    metrics = ['precision', 'recall', 'f1', 'f2']
    x = np.arange(len(metrics))
    width = 0.25

    for i, name in enumerate(models):
        values = [results[name]['metrics'][m] for m in metrics]
        ax.bar(x + i*width, values, width, label=name.upper(),
               color=colors.get(name, 'gray'))

    ax.set_ylabel('Score')
    ax.set_title('Metric Comparison (threshold=0.5)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Inference Time Comparison
    ax = axes[1, 1]
    times = [results[name]['inference_time'] * 1000 for name in models]  # Convert to ms
    bars = ax.bar(models, times, color=[colors.get(n, 'gray') for n in models])
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Latency (Full Test Set)')
    ax.set_xticklabels([n.upper() for n in models])

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Comparison plot saved to {save_path}")


def generate_report(results, save_path):
    """Generate markdown comparison report."""
    lines = [
        "# Hard Drive Failure Prediction - Model Comparison\n",
        "## Summary\n",
        "| Model | Precision | Recall | F1 | F2 | ROC AUC | PR AUC | Inference (ms) |",
        "|-------|-----------|--------|----|----|---------|--------|----------------|"
    ]

    for name, r in results.items():
        m = r['metrics']
        t = r['inference_time'] * 1000
        lines.append(
            f"| {name.upper()} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['f2']:.3f} | {m['roc_auc']:.3f} | "
            f"{m['pr_auc']:.3f} | {t:.1f} |"
        )

    lines.extend([
        "\n## Key Findings\n",
        "### XGBoost",
        "- **Pros**: Fast training and inference, interpretable feature importance",
        "- **Cons**: Requires manual feature engineering, no temporal awareness",
        "- **Best for**: Quick baselines, limited compute resources\n",
        "### LSTM",
        "- **Pros**: Learns temporal patterns automatically, handles variable sequences",
        "- **Cons**: Sequential training (slow), vanishing gradient for long sequences",
        "- **Best for**: Medium-length sequences, moderate compute\n",
        "### Transformer",
        "- **Pros**: Parallelizable, attention shows what matters, long-range dependencies",
        "- **Cons**: More parameters, needs more data, slower inference",
        "- **Best for**: Large datasets, interpretability requirements\n",
        "## Recommendation\n",
        "For production hard drive failure prediction:\n",
        "1. **Start with XGBoost** as baseline (often competitive)",
        "2. **Add Transformer** if you need attention-based interpretability",
        "3. **Consider ensemble** of both for best performance\n",
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to {save_path}")


def main():
    """Run full model comparison."""
    print("="*60)
    print("Model Comparison - Hard Drive Failure Prediction")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    models = load_all_models()

    if len(models) == 0:
        print("No trained models found. Run training scripts first.")
        return

    # Load test data
    _, _, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences()

    results = {}

    # Evaluate each model
    for name, model_dict in models.items():
        print(f"\nEvaluating {name.upper()}...")

        if name == 'xgboost':
            y_true, y_prob, inference_time = evaluate_xgboost(model_dict, test_df)
        else:
            model = model_dict['model']
            y_true, y_prob, inference_time = evaluate_pytorch(
                model, X_test, y_test.numpy(), device
            )

        metrics = compute_metrics(y_true, y_prob)

        results[name] = {
            'y_true': y_true,
            'y_prob': y_prob,
            'metrics': metrics,
            'inference_time': inference_time
        }

        print(f"  F2: {metrics['f2']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Inference: {inference_time*1000:.1f}ms")

    # Generate visualizations and report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, MODELS_DIR / "model_comparison.png")
    generate_report(results, MODELS_DIR / "comparison_report.md")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print("\n{:<12} {:>10} {:>10} {:>10} {:>12}".format(
        "Model", "F2", "ROC AUC", "PR AUC", "Inference"
    ))
    print("-"*54)

    for name, r in results.items():
        m = r['metrics']
        print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.1f}ms".format(
            name.upper(), m['f2'], m['roc_auc'], m['pr_auc'],
            r['inference_time']*1000
        ))


if __name__ == "__main__":
    main()
