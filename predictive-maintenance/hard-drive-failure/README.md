# Hard Drive Failure Prediction

Predict hard drive failures using the Backblaze dataset. Compares three approaches: XGBoost (traditional ML), LSTM, and Transformers.

**Full Tutorial**: [largo.dev/tutorials/predictive-maintenance/hard-drive-failure/](https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/)

## The Problem

Hard drives fail. When they do, data is lost. If we can predict failure 7 days in advance, we can proactively migrate data to healthy drives.

The Backblaze dataset contains daily SMART (Self-Monitoring, Analysis, and Reporting Technology) attributes from hundreds of thousands of drives over many years—a perfect playground for predictive maintenance.

## Approaches Compared

| Model | Approach | Strengths |
|-------|----------|-----------|
| **XGBoost** | Point-in-time features | Fast, interpretable, strong baseline |
| **LSTM** | Sequence modeling | Learns temporal patterns automatically |
| **Transformer** | Self-attention | Attention shows what matters, parallel training |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_all.py

# Or run quick mode (subset of data)
python run_all.py --quick

# Train individual models
python run_all.py --model xgboost
python run_all.py --model lstm
python run_all.py --model transformer
```

## Project Structure

```
hard-drive-failure/
├── config.py           # Hyperparameters and paths
├── data_pipeline.py    # Download and preprocess Backblaze data
├── model_xgboost.py    # XGBoost baseline
├── model_lstm.py       # Bidirectional LSTM
├── model_transformer.py # Transformer with attention
├── compare_models.py   # Compare all approaches
├── run_all.py          # Full pipeline runner
└── requirements.txt    # Dependencies
```

## Key Concepts

### SMART Attributes

Hard drives report health metrics via SMART. The most predictive:

- `smart_5`: Reallocated Sectors Count (bad sectors)
- `smart_187`: Reported Uncorrectable Errors
- `smart_197`: Current Pending Sector Count
- `smart_198`: Offline Uncorrectable Sector Count
- `smart_9`: Power-On Hours

### Class Imbalance

Only ~1-2% of drives fail. We handle this via:
- Weighted sampling during training
- Class weights in loss functions
- F2 score (emphasizes recall over precision)
- Threshold optimization

### Time-Based Splits

We split data by time (train on past, test on future) to prevent data leakage and simulate real deployment conditions.

## Expected Results

From the Backblaze dataset:

| Model | F2 Score | ROC AUC | Inference |
|-------|----------|---------|-----------|
| XGBoost | 0.45-0.55 | 0.85-0.90 | ~10ms |
| LSTM | 0.50-0.60 | 0.88-0.92 | ~50ms |
| Transformer | 0.52-0.62 | 0.89-0.93 | ~100ms |

**Note**: Results vary by quarter and drive population. XGBoost is often surprisingly competitive—don't dismiss it!

## Key Takeaways

1. **XGBoost is a strong baseline** — Always start here
2. **Sequence models help** — But gains may not justify complexity
3. **Transformers add interpretability** — Attention shows which days matter
4. **Feature engineering matters** — Rolling stats boost all models
5. **Threshold tuning is critical** — Default 0.5 is rarely optimal

## Next Steps

After completing this tutorial:

- Try different prediction horizons (3 days vs 30 days)
- Experiment with different drive models (Seagate vs HGST vs WDC)
- Add uncertainty quantification (when is the model confident?)
- Deploy as an API with real-time SMART data ingestion

## Data Source

[Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

Released quarterly, this dataset contains daily SMART snapshots from Backblaze's production data center drives. It's one of the largest public datasets for predictive maintenance research.
