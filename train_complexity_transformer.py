import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Models.GeneralModels.attention_test import classnames66

class EmbeddingComplexityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: (batch, seq_len)
        b, seq_len = x.size()
        x = x.unsqueeze(-1)              # (batch, seq_len, 1)
        x = self.proj(x)                 # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)           # (seq_len, batch, d_model)
        x = self.transformer(x)          # (seq_len, batch, d_model)
        x = x.permute(1, 2, 0)           # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)     # (batch, d_model)
        out = self.regressor(x).squeeze(-1)
        return out


def main():
    # paths
    base_dir = os.path.dirname(__file__)
    emb_csv = os.path.join(base_dir, 'Embeddings', 'CLIP-HBA', 'Savoias', 'Objects_output', 'static_embedding.csv')
    gt_csv = os.path.join(base_dir, 'Images', 'Savoias-Dataset', 'GroundTruth', 'csv', 'global_ranking_objects_labeled.csv')

    # load
    emb = pd.read_csv(emb_csv, index_col='image')
    gt = pd.read_csv(gt_csv, index_col='image')
    df = emb.join(gt).dropna()
    X = df.iloc[:, :66].values
    y = df['gt'].values

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # datasets and loaders
    train_ds = EmbeddingComplexityDataset(X_train, y_train)
    val_ds   = EmbeddingComplexityDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    model = TransformerRegressor(seq_len=66).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # train loop
    epochs = 50
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs}  Train MSE: {train_loss:.4f}  Val MSE: {val_loss:.4f}")

    # save model, scaler, and feature names for interpretability
    save_dict = {
        'model_state': model.state_dict(),
        'scaler': scaler,
        'feature_names': classnames66
    }
    torch.save(save_dict, os.path.join(base_dir, 'transformer_complexity_model.pt'))
    print("Transformer-based model, scaler, and feature names saved for interpretability.")

if __name__ == '__main__':
    main()
