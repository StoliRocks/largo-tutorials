"""
Transformer Model for Hard Drive Failure Prediction

Attention-based approach that can capture long-range dependencies
in SMART attribute sequences and identify which time steps matter most.

Key advantages:
- Attention reveals which historical readings predict failure
- Parallelizable training (faster than LSTM on GPU)
- Captures long-range dependencies without vanishing gradients
- State-of-the-art for many sequence tasks

Key considerations:
- Requires positional encoding (no inherent sequence order)
- More parameters than LSTM for same hidden dim
- May need more data to train effectively

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, MODELS_DIR, TRANSFORMER_PARAMS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.

    Since transformers have no inherent notion of sequence order,
    we add positional information via sinusoidal functions at
    different frequencies.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for binary sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        → Linear projection to d_model
        → Positional encoding
        → Transformer encoder layers
        → Mean pooling over sequence
        → Classification head

    We use mean pooling rather than a CLS token for simplicity,
    though both approaches work well in practice.
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_attention: If True, also return attention weights

        Returns:
            (batch_size,) probability of failure
        """
        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        encoded = self.transformer(x)

        # Mean pooling over sequence dimension
        pooled = encoded.mean(dim=1)

        # Classify
        logits = self.classifier(pooled).squeeze(-1)

        return torch.sigmoid(logits)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability.

        Shows which time steps the model attends to when making predictions.
        """
        self.eval()
        with torch.no_grad():
            x = self.input_projection(x)
            x = self.pos_encoder(x)

            # Get attention from each layer
            attention_weights = []
            for layer in self.transformer.layers:
                # Forward through self-attention manually to get weights
                attn_output, attn_weights = layer.self_attn(
                    x, x, x, need_weights=True
                )
                attention_weights.append(attn_weights)
                x = layer(x)

        # Average across layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=0)
        return avg_attention


def load_sequences():
    """Load preprocessed sequence data."""
    data = np.load(PROCESSED_DIR / "sequences.npz")

    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.FloatTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'])

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                       batch_size: int = BATCH_SIZE):
    """Create PyTorch DataLoaders with class-balanced sampling."""
    pos_count = y_train.sum().item()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    sample_weights = torch.where(y_train == 1, pos_weight, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device, threshold: float = 0.5):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            probs = model(X_batch)
            loss = criterion(probs, y_batch)

            preds = (probs >= threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-9)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        'loss': total_loss / num_batches,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def plot_attention_heatmap(model, X_sample, feature_names, save_path):
    """
    Visualize attention weights to show which time steps matter.

    This is the key interpretability advantage of transformers.
    """
    device = next(model.parameters()).device
    X_sample = X_sample.to(device)

    attention = model.get_attention_weights(X_sample)

    # Take first sample, average over all positions
    attn_pattern = attention[0].cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(attn_pattern, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position (Time Step)')
    plt.ylabel('Query Position (Time Step)')
    plt.title('Self-Attention Heatmap\n(Which time steps attend to which)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Full training loop with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Model
    input_dim = X_train.shape[2]
    model = TransformerClassifier(input_dim, **TRANSFORMER_PARAMS).to(device)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = nn.BCELoss()

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_f2 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f2': []}

    print("\nTraining Transformer...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_f2'].append(val_results['f2'])

        scheduler.step()

        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_results['loss']:.4f}, "
              f"Val F2={val_results['f2']:.4f}, "
              f"Val AUC={val_results['auc']:.4f}")

        # Early stopping
        if val_results['f2'] > best_f2:
            best_f2 = val_results['f2']
            patience_counter = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "transformer_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "transformer_best.pt"))

    # Final evaluation
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    test_results = evaluate(model, test_loader, criterion, device)

    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1:        {test_results['f1']:.4f}")
    print(f"  F2:        {test_results['f2']:.4f}")
    print(f"  AUC:       {test_results['auc']:.4f}")

    # Plot attention for a sample
    sample_idx = np.where(y_test.numpy() == 1)[0]
    if len(sample_idx) > 0:
        X_sample = X_test[sample_idx[0]:sample_idx[0]+1]
        plot_attention_heatmap(
            model, X_sample, None,
            MODELS_DIR / "transformer_attention.png"
        )
        print(f"\nAttention heatmap saved to {MODELS_DIR / 'transformer_attention.png'}")

    return model, test_results, history


def main():
    """Train and evaluate Transformer model."""
    print("="*60)
    print("Transformer Model - Hard Drive Failure Prediction")
    print("="*60)

    # Load sequences
    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Positive rate (train): {100*y_train.mean():.2f}%")

    # Train
    model, results, history = train_transformer(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "="*60)
    print("Transformer Training Complete")
    print("="*60)
    print(f"Test F2 Score: {results['f2']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")

    return model, results


if __name__ == "__main__":
    main()
