# Sentence Embeddings from Scratch

Build a complete sentence encoder from the ground up with PyTorch.

**Full Tutorial**: [largo.dev/tutorials/embeddings/sentence-embeddings-from-scratch/](https://largo.dev/tutorials/embeddings/sentence-embeddings-from-scratch/)

## What You'll Learn

- Character-level tokenization (and why production uses subword tokenizers)
- Embedding layers and how they map tokens to dense vectors
- Bidirectional LSTMs for sequence encoding
- Pooling strategies: mean, max, and CLS token
- L2 normalization for cosine similarity

## Architecture

```
Token IDs → Embedding → BiLSTM → Mean Pooling → L2 Normalize
```

## Quick Start

```bash
pip install -r requirements.txt
python sentence_encoder.py
```

## Output

```
============================================================
Sentence Embeddings from Scratch
============================================================

Vocabulary size: 29
Embedding output dimension: 512

Test sentences:
  [0] the cat sat on the mat
  [1] a cat was sitting on a mat
  [2] the dog ran in the park
  [3] machine learning is fascinating

Cosine similarity matrix:
[[1.    0.xxx 0.xxx 0.xxx]
 [0.xxx 1.    0.xxx 0.xxx]
 [0.xxx 0.xxx 1.    0.xxx]
 [0.xxx 0.xxx 0.xxx 1.   ]]

Note: This is an UNTRAINED model with random weights.
```

## Files

- `sentence_encoder.py` — Complete implementation with demo

## Next Steps

After understanding the basics, explore:
- Training with contrastive learning (InfoNCE loss)
- Transformer-based encoders (BERT, Sentence-BERT)
- Production sentence embedding models (all-MiniLM, GTE, BGE)
