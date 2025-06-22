import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MoodClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train_model(X, y, mlb, model_path, mlb_path, scaler_path=None, scaler=None):
    X_np = np.array(X)
    y_np = np.array(y)
    input_dim = X_np.shape[1]
    num_classes = y_np.shape[1]  # Will be 1-hot for single-label classification

    y_labels = np.argmax(y_np, axis=1)  # convert from 1-hot to class indices

    # Print mood class distribution
    print("Mood label distribution:")
    for i, count in enumerate(np.bincount(y_labels)):
        print(f"  {mlb.classes_[i]}: {count} samples")

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_labels, test_size=0.2, stratify=y_labels, random_state=42)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MoodClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    best_acc = 0
    patience = 20
    counter = 0
    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} - Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            counter = 0
            print("  Saving best model...")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("\n[COMPLETE] Training finished")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to {model_path}")

    joblib.dump(mlb, mlb_path)
    if scaler is not None and scaler_path is not None:
        joblib.dump(scaler, scaler_path)


if __name__ == "__main__":
    pass 
