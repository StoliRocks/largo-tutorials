# Largo Tutorials

Companion code for tutorials at [largo.dev](https://largo.dev).

Practical, code-heavy implementations of ML concepts—embeddings, transformers, time series, edge deployment, and more.

## Tutorials

### Embeddings
- **[Sentence Embeddings from Scratch](embeddings/sentence-embeddings-from-scratch/)** — Build a BiLSTM sentence encoder with PyTorch. Covers tokenization, embedding layers, and pooling strategies.

### Predictive Maintenance
- **[Hard Drive Failure Prediction](predictive-maintenance/hard-drive-failure/)** — Predict drive failures using Backblaze SMART data (24.8M records, Q4 2023). Compares XGBoost, LSTM, Transformer, and Mamba architectures.

  | Model | Test AUC | F2 | Recall | Notes |
  |-------|----------|-----|--------|-------|
  | **XGBoost** | **0.920** | **0.095** | 54% | Feature engineering wins |
  | Transformer | 0.916 | 0.006 | 80% | Conv1D preprocessing |
  | LSTM | 0.907 | 0.006 | 78% | Bidirectional |
  | Mamba (SSM) | 0.901 | 0.017 | 69% | Linear complexity O(n) |

  Key techniques: balanced sampling, GPU acceleration with cuDF, Conv1D preprocessing, threshold optimization for extreme class imbalance (0.01% positive rate).

  **New**: `model_mamba.py` implements Mamba SSM using the official `mamba-ssm` library. All neural models achieve ~0.90 AUC (good ranking), but XGBoost with engineered features achieves 10x better F2.

## Setup

```bash
# Clone the repo
git clone https://github.com/StoliRocks/largo-tutorials.git
cd largo-tutorials

# Install common dependencies
pip install -r requirements.txt

# Or install for a specific tutorial
pip install -r embeddings/sentence-embeddings-from-scratch/requirements.txt
```

## Author

**Steven W. White** — [largo.dev](https://largo.dev) · [LinkedIn](https://www.linkedin.com/in/stvwhite)

## License

MIT
