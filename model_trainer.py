import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_curve
import joblib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MusicGenreModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.4)

        self.attention = nn.Linear(2048, 2048)

        # Genre branches (metal, punk, blues)
        def branch():
            return nn.Sequential(
                nn.Linear(2048, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
        self.metal_branch = branch()
        self.punk_branch = branch()
        self.blues_branch = branch()

        self.final_dense = nn.Linear(2048 + 256*3, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.silu(self.dense1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        att = torch.sigmoid(self.attention(x))
        x_att = x * att

        metal_feat = self.metal_branch(x_att)
        punk_feat = self.punk_branch(x_att)
        blues_feat = self.blues_branch(x_att)

        x_concat = torch.cat([x_att, metal_feat, punk_feat, blues_feat], dim=1)

        x = F.silu(self.final_dense(x_concat))
        x = self.bn2(x)
        x = self.dropout2(x)

        out = torch.sigmoid(self.output_layer(x))
        return out


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights  # tensor shape [num_classes]

    def forward(self, y_pred, y_true):
        loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        if self.weights is not None:
            loss = loss * self.weights
        return loss.mean()


def compute_balanced_weights(class_freqs, smoothing=0.15):
    inv_freq = {k: 1 / (v + 1e-6) for k, v in class_freqs.items()}
    max_inv = max(inv_freq.values())
    weights = {k: (v / max_inv) for k, v in inv_freq.items()}
    weights = {k: smoothing + (1 - smoothing) * w for k, w in weights.items()}
    return weights


def print_label_distribution(y, mlb, name):
    print(f"\n{name} set label distribution:")
    label_counts = y.sum(axis=0)
    total_samples = y.shape[0]
    for i, count in enumerate(label_counts):
        print(f"{mlb.classes_[i]}: {count} samples ({count / total_samples:.1%})")


def find_optimal_thresholds(model, dataloader, y_true, mlb, label_freqs):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
    y_pred_val = np.vstack(all_preds)
    thresholds = {}
    median_freq = np.median(label_freqs)

    plt.figure(figsize=(12, 8))
    for i, genre in enumerate(mlb.classes_):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_pred_val[:, i])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1)
        freq_weight = np.log(median_freq / (label_freqs[i] + 1e-6))
        thresh_opt = np.clip(thresh[best_idx] * (1 - 0.1 * freq_weight), 0.2, 0.8)
        thresholds[genre] = thresh_opt
        plt.plot(thresh, f1[:-1], label=f'{genre} (best={thresh_opt:.2f})')

    plt.title('F1 Scores by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/threshold_analysis.png')
    plt.close()
    return thresholds


def train_model(X, y, mlb, model_path, mlb_path, scaler_path=None, scaler=None):
    X_np = X.numpy() if torch.is_tensor(X) else np.array(X)
    y_np = y.numpy() if torch.is_tensor(y) else np.array(y)
    input_dim = X_np.shape[1]
    num_classes = y_np.shape[1]

    # Train/val split (80/20) using sklearn stratified split or random
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in msss.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

    print_label_distribution(y_train, mlb, "Training")
    print_label_distribution(y_val, mlb, "Validation")

    batch_size = 512
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    label_freqs = y_np.sum(axis=0)
    class_freqs = {genre: freq / len(y_np) for genre, freq in zip(mlb.classes_, label_freqs)}
    class_weights_dict = compute_balanced_weights(class_freqs)

    class_weights = torch.tensor([class_weights_dict[genre] for genre in mlb.classes_], dtype=torch.float32).to(device)
    manual_boosts = {
        'punk': 4.0, 'country': 4.0, 'industrial': 3.5,
        'blues': 3.0, 'reggae': 2.8, 'metal': 3.0,
        'jazz': 2.5, 'lounge': 2.3, 'world': 2.3, 'rnb': 2.0,
        'folk': 2.0, 'hiphop': 2.0,
        'classical': 2.0, 'rock': 2.0,
        'electronic': 0.4, 'pop': 1.8, 'soundtrack': 2.5,
        'alternative': 3.5
    }
    for i, genre in enumerate(mlb.classes_):
        if genre in manual_boosts:
            class_weights[i] *= manual_boosts[genre]

    print("\nFinal Class Weights:")
    for genre, weight in zip(mlb.classes_, class_weights):
        print(f"{genre:15s}: {weight.item():.2f}")

    model = MusicGenreModel(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = WeightedBCELoss(class_weights)

    epochs = 150
    best_val_f1 = 0
    patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).cpu().numpy()
                val_preds.append(outputs)
                val_true.append(y_batch.numpy())
        val_preds = np.vstack(val_preds)
        val_true = np.vstack(val_true)

        # Threshold = 0.5 for macro F1 calculation
        val_pred_binary = (val_preds > 0.5).astype(int)
        f1_per_class = []
        for i in range(num_classes):
            f1_per_class.append(f1_score(val_true[:, i], val_pred_binary[:, i], zero_division=0))
        val_macro_f1 = np.mean(f1_per_class)

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Macro F1: {val_macro_f1:.4f}")

        # Early stopping
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print("  Saving best model...")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Optionally adjust LR on plateau here with scheduler

    print("\nLoading best model for threshold tuning...")
    model.load_state_dict(torch.load(model_path))

    thresholds = find_optimal_thresholds(model, val_loader, val_true, mlb, label_freqs)
    threshold_path = os.path.join(os.path.dirname(model_path), 'genre_thresholds.pkl')
    joblib.dump(thresholds, threshold_path)
    print(f"Saved thresholds to {threshold_path}")

    # Save label binarizer & scaler if provided
    joblib.dump(mlb, mlb_path)
    if scaler is not None and scaler_path is not None:
        joblib.dump(scaler, scaler_path)

    print("\n[COMPLETE] Training finished")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    pass
