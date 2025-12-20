# Hard Drive Failure Prediction

Predict hard drive failures using Backblaze SMART data. Compares 5 model architectures with techniques for handling extreme class imbalance (0.01% failure rate).

**Full Tutorial**: [largo.dev/tutorials/predictive-maintenance/hard-drive-failure/](https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/)

## The Problem

Hard drives fail. When they do, data is lost. If we can predict failure 7 days in advance, we can proactively migrate data to healthy drives.

The challenge: **99.99% of drive-days are normal**. A naive model predicting "normal" for everything gets 99.99% accuracy but catches 0% of failures.

## Results

| Model | Precision | Recall | F1 | Best For |
|-------|-----------|--------|-----|----------|
| **XGBoost** | **97%** | 72% | 0.83 | Precision, speed |
| **CNN-LSTM** | 79% | **85%** | 0.82 | Maximum recall |
| **Conv-Transformer** | 82% | 81% | 0.81 | Balanced, interpretable |
| LSTM | 81% | 68% | 0.74 | Baseline sequence model |
| Transformer | 65% | 58% | 0.61 | Needs Conv1D preprocessing |

## Key Insight

**Conv1D is the secret ingredient.** Both CNN-LSTM and Conv-Transformer dramatically outperform their basic counterparts. The convolution extracts cross-attribute patterns (e.g., "smart_5 + smart_187 rising together = danger") that sequential models miss.

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
├── model_xgboost.py    # XGBoost with balanced sampling
├── model_lstm.py       # LSTM and CNN-LSTM implementations
├── model_transformer.py # Transformer and Conv-Transformer
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

Only ~0.01% of drive-days are failures. We handle this via:
- **Balanced sampling**: Downsample majority class to match minority
- Class weights in loss functions
- F2 score (emphasizes recall over precision)
- GPU acceleration with cuDF (4.4x faster than CPU pandas)

### Time-Based Splits

We split data by time (train on past, test on future) to prevent data leakage and simulate real deployment conditions.

## Key Takeaways

1. **Don't trust accuracy** — 99.99% accuracy means nothing with 0.01% failure rate
2. **Balanced sampling is essential** — Improves recall from 3% to 72%+
3. **Conv1D preprocessing helps all sequence models** — CNN-LSTM and Conv-Transformer both outperform basic versions by 20%+ F1
4. **XGBoost for precision, CNN-LSTM for recall** — Choose based on your cost model
5. **Simpler transformers work better** — d_model=32, 1 layer beats complex architectures on short sequences

## Next Steps

After completing this tutorial:

- Try different prediction horizons (3 days vs 30 days)
- Experiment with different drive models (Seagate vs HGST vs WDC)
- Add uncertainty quantification (when is the model confident?)
- Deploy as an API with real-time SMART data ingestion

## Data Source

[Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

Released quarterly, this dataset contains daily SMART snapshots from Backblaze's production data center drives. It's one of the largest public datasets for predictive maintenance research.
