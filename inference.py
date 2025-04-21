import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import joblib

# --- Paths ---
MODEL_PATH = "/home/wallacelab/investigating-complexity/transformer_predictor_state.pth"
SCALER_PATH = "scaler.joblib" s
OUTPUT_BASE_DIR = "/home/wallacelab/investigating-complexity/output/new"

# --- Model Definition ---
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, h):
        scores = self.attention_weights(h)
        alpha = F.softmax(scores, dim=1)
        pooled = torch.sum(h * alpha, dim=1)
        return pooled, alpha.squeeze(-1)

class TransformerPredictor(nn.Module):
    def __init__(self, num_features=66, d_model=256, nhead=16, num_encoder_layers=8, dim_feedforward=1024, dropout=0.01):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)
        self.dim_embed = nn.Embedding(num_features, d_model)
        self.register_buffer('dim_id', torch.arange(num_features))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        encoder_layer.self_attn.average_attn_weights = False
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.attn_pool = AttnPool(d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_raw):
        x = x_raw.unsqueeze(-1)
        x_tok = self.token_embed(x)
        dim_ids = self.dim_id.to(x_tok.device)
        h_in = x_tok + self.dim_embed(dim_ids)
        h = self.transformer_encoder(h_in)
        pooled, _ = self.attn_pool(h)
        return self.prediction_head(pooled)

# --- Utility Functions ---
def scale_ground_truth(gt_values):
    return gt_values / 100.0

def evaluate_spearman(preds, targets):
    return spearmanr(preds.flatten(), targets.flatten())[0]

def plot_scatter(targets, preds, spearman_rho, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, label='Predictions')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y = x)')
    plt.xlabel('True Scores (Scaled)')
    plt.ylabel('Predicted Scores')
    plt.title(f"Inference: Spearman ρ = {spearman_rho:.3f}")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)
    plt.close()

# --- Inference Function ---
def run_inference(embedding_csv, gt_csv):
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load embedding CSV
    df_embeddings = pd.read_csv(embedding_csv)
    images = df_embeddings['image'].tolist()
    features = df_embeddings.drop(columns=['image']).values.astype(np.float32)

    # Load GT CSV and scale values from 0-100 to 0-1
    df_gt = pd.read_csv(gt_csv)
    if 'gt' not in df_gt.columns:
        raise ValueError("Ground truth CSV must contain a 'gt' column.")
    targets = scale_ground_truth(df_gt['gt'].values.astype(np.float32))

    # Scale input features
    scaler = joblib.load(SCALER_PATH)
    features_scaled = scaler.transform(features)

    # Load model
    model = TransformerPredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Inference
    with torch.no_grad():
        inputs = torch.tensor(features_scaled, dtype=torch.float32)
        preds = model(inputs).squeeze().numpy()

    # Spearman correlation
    rho = evaluate_spearman(preds, targets)

    # Save results
    result_df = pd.DataFrame({
        'image_name': images,
        'ground_truth_complexity': targets,
        'predicted_complexity': preds
    })
    result_csv_path = os.path.join(output_dir, "inference_results.csv")
    result_df.to_csv(result_csv_path, index=False)

    # Save scatterplot
    scatter_path = os.path.join(output_dir, "inference_scatterplot.png")
    plot_scatter(targets, preds, rho, scatter_path)

    print(f"Inference complete. Spearman ρ = {rho:.3f}")
    print(f"Results saved to {output_dir}")


# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Transformer-based inference on image embeddings.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embedding CSV with 'image' column.")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth scores CSV.")
    args = parser.parse_args()

    run_inference(args.embeddings, args.ground_truth)
