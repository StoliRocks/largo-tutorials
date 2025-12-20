"""
LSTM Model for Hard Drive Failure Prediction

Sequence-based approach using bidirectional LSTM to capture
temporal patterns in SMART attribute degradation.

Key advantages over XGBoost:
- Learns temporal patterns automatically (no manual rolling features)
- Can capture long-term degradation trends
- End-to-end learning from raw sequences

Key challenges:
- Requires more data and compute
- Longer training time
- Hyperparameter sensitivity

Tutorial: https://largo.dev/tutorials/predictive-maintenance/hard-drive-failure/
Author: Steven W. White
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from config import (
    PROCESSED_DIR, MODELS_DIR, LSTM_PARAMS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
)


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        → Bidirectional LSTM (captures forward and backward patterns)
        → Take final hidden state
        → Dropout
        → Linear → Sigmoid

    The bidirectional architecture is important because degradation
    patterns may be clearer when viewed from either direction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        # Output layer takes concatenated forward and backward states
        self.classifier = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            (batch_size,) probability of failure
        """
        # LSTM output: (batch, seq, hidden * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the final hidden states from both directions
        if self.bidirectional:
            # Concatenate forward and backward final states
            forward_final = hidden[-2, :, :]
            backward_final = hidden[-1, :, :]
            final_hidden = torch.cat([forward_final, backward_final], dim=1)
        else:
            final_hidden = hidden[-1, :, :]

        # Dropout and classify
        dropped = self.dropout(final_hidden)
        logits = self.classifier(dropped).squeeze(-1)

        return torch.sigmoid(logits)


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
    # Compute class weights for balanced sampling
    pos_count = y_train.sum().item()
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    # Sample weights for WeightedRandomSampler
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

    return train_loader, val_loader, test_loader, pos_weight


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

        # Gradient clipping to prevent exploding gradients in LSTM
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


def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Full training loop with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Model
    input_dim = X_train.shape[2]  # Number of SMART features
    model = LSTMClassifier(input_dim, **LSTM_PARAMS).to(device)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class weight
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_f2 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f2': []}

    print("\nTraining LSTM...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_f2'].append(val_results['f2'])

        scheduler.step(val_results['f2'])

        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_results['loss']:.4f}, "
              f"Val F2={val_results['f2']:.4f}, "
              f"Val AUC={val_results['auc']:.4f}")

        # Early stopping
        if val_results['f2'] > best_f2:
            best_f2 = val_results['f2']
            patience_counter = 0
            # Save best model
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt"))

    # Final evaluation on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    test_results = evaluate(model, test_loader, criterion, device)

    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1:        {test_results['f1']:.4f}")
    print(f"  F2:        {test_results['f2']:.4f}")
    print(f"  AUC:       {test_results['auc']:.4f}")

    return model, test_results, history


def main():
    """Train and evaluate LSTM model."""
    print("="*60)
    print("LSTM Model - Hard Drive Failure Prediction")
    print("="*60)

    # Load sequences
    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Positive rate (train): {100*y_train.mean():.2f}%")

    # Train
    model, results, history = train_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "="*60)
    print("LSTM Training Complete")
    print("="*60)
    print(f"Test F2 Score: {results['f2']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")

    return model, results


if __name__ == "__main__":
    main()
