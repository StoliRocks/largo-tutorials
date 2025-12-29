"""
Mamba (State Space Model) for Hard Drive Failure Prediction

This module implements a Mamba-based architecture for predictive maintenance,
comparing it to LSTM and Transformer baselines. Mamba offers linear complexity
O(n) compared to Transformer's O(n²), while maintaining competitive accuracy.

Key features:
- Selective state space mechanism (learned input-dependent filtering)
- Linear complexity for long sequences
- Conv1D preprocessing for local pattern extraction (hybrid approach)

Tutorial: https://largo.dev/tutorials/production-ml/mamba-predictive-maintenance/
Author: Steven W. White
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from config import (
    PROCESSED_DIR, MODELS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE
)

# Mamba-specific parameters
MAMBA_PARAMS = {
    "d_model": 64,           # Model dimension
    "d_state": 16,           # SSM state dimension (N in paper)
    "d_conv": 4,             # Local convolution width
    "expand": 2,             # Expansion factor for inner dimension
    "num_layers": 2,         # Number of Mamba blocks
    "dropout": 0.1,
}


class MambaBlock(nn.Module):
    """
    A single Mamba block implementing selective state space mechanism.

    This is a simplified implementation that captures the key ideas:
    1. Input-dependent selection (selectivity)
    2. Hardware-efficient parallel scan
    3. Gated architecture similar to GLU

    For production, use the official mamba-ssm package.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection (expands dimension)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise conv
            bias=True
        )

        # SSM parameters - these are input-dependent (selective)
        # Δ, B, C are projected from input
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable parameters for SSM
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)

        # Input projection with gating
        xz = self.in_proj(x)  # (batch, seq, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, seq, d_inner)

        # 1D convolution for local context
        x = x.transpose(1, 2)  # (batch, d_inner, seq)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim to original length
        x = x.transpose(1, 2)  # (batch, seq, d_inner)
        x = torch.silu(x)

        # SSM computation (simplified - actual Mamba uses parallel scan)
        y = self.ssm(x)

        # Gating (similar to GLU)
        y = y * torch.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual

    def ssm(self, x):
        """
        Simplified SSM computation.

        The actual Mamba uses hardware-efficient parallel scan.
        This sequential version is for clarity.
        """
        batch_size, seq_len, d_inner = x.shape

        # Project to get Δ, B, C (input-dependent = selective)
        x_proj = self.x_proj(x)  # (batch, seq, d_state*2 + 1)

        delta = x_proj[:, :, :1]  # Discretization step
        B = x_proj[:, :, 1:1+self.d_state]
        C = x_proj[:, :, 1+self.d_state:]

        # Discretize Δ
        delta = torch.softplus(self.dt_proj(delta))  # (batch, seq, d_inner)

        # Get A from learnable log
        A = -torch.exp(self.A_log)  # (d_state,)

        # Discretize A: Ā = exp(Δ·A)
        dA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq, d_inner, d_state)

        # Discretize B: B̄ = Δ·B
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_inner, d_state)

        # Sequential scan (could be parallelized with associative scan)
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)
            y_t = (h * C[:, t:t+1, :].transpose(1, 2)).sum(dim=-1)  # (batch, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq, d_inner)

        # Add skip connection with D
        y = y + x * self.D

        return y


class ConvMambaClassifier(nn.Module):
    """
    Conv1D + Mamba hybrid for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        → Conv1D (extract local patterns across features)
        → Mamba blocks (capture long-range dependencies with linear complexity)
        → Global average pooling
        → Classification head

    This hybrid approach follows the same pattern as CNN-LSTM and
    Conv-Transformer, using Conv1D to provide local feature extraction
    before the sequence model.
    """

    def __init__(self, input_dim: int, d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # Conv1D feature extractor (like CNN-LSTM and Conv-Transformer)
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(d_model)

        # Stack of Mamba blocks
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch,) probabilities
        """
        # Conv1D expects (batch, channels, seq)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = x.transpose(1, 2)  # Back to (batch, seq, d_model)
        x = self.conv_norm(x)

        # Apply Mamba blocks
        for mamba in self.mamba_layers:
            x = mamba(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(x).squeeze(-1)

        return torch.sigmoid(logits)


class OfficialMambaClassifier(nn.Module):
    """
    Classifier using the official mamba-ssm package.

    Falls back to the simplified implementation if mamba-ssm is not available.
    """

    def __init__(self, input_dim: int, d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.use_official = False

        try:
            from mamba_ssm import Mamba
            self.use_official = True
            print("Using official mamba-ssm implementation (faster)")

            # Conv1D feature extractor
            self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
            self.conv_norm = nn.LayerNorm(d_model)

            # Official Mamba layers
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ])

            self.norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])

        except ImportError:
            print("mamba-ssm not available, using simplified implementation")
            # Fallback to our simplified version
            self.model = ConvMambaClassifier(
                input_dim, d_model, d_state, d_conv, expand, num_layers, dropout
            )
            return

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        if not self.use_official:
            return self.model(x)

        # Conv1D preprocessing
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.conv_norm(x)

        # Mamba layers with residual connections
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        # Pool and classify
        x = x.mean(dim=1)
        logits = self.classifier(x).squeeze(-1)

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def find_optimal_threshold(labels, probs):
    """Find threshold that maximizes F2 score."""
    best_f2 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (probs >= threshold).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-9)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
    return best_threshold, best_f2


def evaluate(model, data_loader, criterion, device, threshold: float = None):
    """Evaluate model on a dataset."""
    model.eval()
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

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(all_labels, all_probs)

    all_preds = (all_probs >= threshold).astype(int)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-9)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        'loss': total_loss / max(num_batches, 1),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'auc': auc,
        'threshold': threshold,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def train_mamba(X_train, y_train, X_val, y_val, X_test, y_test,
                use_official: bool = True):
    """Full training loop with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    input_dim = X_train.shape[2]

    if use_official:
        model = OfficialMambaClassifier(input_dim, **MAMBA_PARAMS).to(device)
    else:
        model = ConvMambaClassifier(input_dim, **MAMBA_PARAMS).to(device)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f2': [], 'val_auc': []}

    # Create models dir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTraining Mamba...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_f2'].append(val_results['f2'])
        history['val_auc'].append(val_results['auc'])

        scheduler.step(val_results['auc'])  # Use AUC for scheduler

        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_results['loss']:.4f}, "
              f"Val F2={val_results['f2']:.4f} (thr={val_results['threshold']:.2f}), "
              f"Val AUC={val_results['auc']:.4f}")

        # Use AUC for early stopping (more stable for imbalanced data)
        if val_results['auc'] > best_auc:
            best_auc = val_results['auc']
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "mamba_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(MODELS_DIR / "mamba_best.pt", weights_only=True))

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
    """Train and evaluate Mamba model."""
    print("="*60)
    print("Mamba Model - Hard Drive Failure Prediction")
    print("="*60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences()

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Positive rate (train): {100*y_train.mean():.2f}%")

    model, results, history = train_mamba(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "="*60)
    print("Mamba Training Complete")
    print("="*60)
    print(f"Test F2 Score: {results['f2']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")

    return model, results


if __name__ == "__main__":
    main()
