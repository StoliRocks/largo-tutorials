# Largo Tutorials

Companion code for tutorials at [largo.dev](https://largo.dev).

Practical, code-heavy implementations of ML concepts—embeddings, transformers, time series, edge deployment, and more.

## Tutorials

### Embeddings
- **[Sentence Embeddings from Scratch](embeddings/sentence-embeddings-from-scratch/)** — Build a BiLSTM sentence encoder with PyTorch. Covers tokenization, embedding layers, and pooling strategies.

### Predictive Maintenance
- **[Hard Drive Failure Prediction](predictive-maintenance/hard-drive-failure/)** — Predict drive failures using Backblaze SMART data. Compares 6 model architectures with balanced sampling for extreme class imbalance.

  | Model | Precision | Recall | F1 | Notes |
  |-------|-----------|--------|-----|-------|
  | XGBoost | **97%** | 72% | 0.83 | Best precision |
  | CNN-LSTM | 79% | **85%** | 0.82 | Best recall |
  | Conv-Transformer | 82% | 81% | 0.81 | Balanced |
  | **Mamba (SSM)** | 0.5% | 40% | 0.01 | Linear complexity O(n) |
  | LSTM | 81% | 68% | 0.74 | Baseline RNN |
  | Transformer | 65% | 58% | 0.61 | Baseline attention |

  Key techniques: balanced sampling, GPU acceleration with cuDF, Conv1D preprocessing for sequence models, Mamba state space models.

  **New**: `model_mamba.py` implements Mamba SSM using the official `mamba-ssm` library. Mamba achieves 10x better F2 than Transformer on imbalanced data with the same parameter count.

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
