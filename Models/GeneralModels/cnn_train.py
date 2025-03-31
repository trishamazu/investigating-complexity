import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from datetime import datetime
from tqdm import tqdm

# ---------------------------
# Define directories and paths
# ---------------------------
EMBEDDING_DIR = "/home/wallacelab/complexity-final/Embeddings/CLIP-HBA/66d/Savoias"
GT_DIR = "/home/wallacelab/complexity-final/Images/Savoias-Dataset/GroundTruth/csv"
OUTPUT_DIR = "/home/wallacelab/complexity-final/output/Savoias_output/cross_attention"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# CrossDimensionalAttention Model
# ---------------------------
class CrossDimensionalAttention(nn.Module):
    def __init__(self, embed_dim=66, num_heads=4, hidden_dim=128, ffn_dim=64, dropout=0.1):
        super(CrossDimensionalAttention, self).__init__()

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)

        attn_output, attn_weights = self.attention(x, x, x)

        x = self.layer_norm(attn_output + x)
        x = x.squeeze(1)

        return self.ffn(x).view(-1, 1), attn_weights.squeeze(1)

# ---------------------------
# Model Evaluation & Visualization
# ---------------------------
def evaluate_model(model, test_loader, category_output_dir):
    model.eval()
    actual, predictions = [], []
    example_embeddings, example_attn_weights = None, None

    with torch.no_grad():
        for batch in test_loader:
            embeddings, targets = batch
            outputs, attn_weights = model(embeddings)

            actual.extend(targets.cpu().numpy().flatten())
            predictions.extend(outputs.cpu().numpy().flatten())

            # Store first batch for heatmap visualization
            if example_embeddings is None:
                example_embeddings, example_attn_weights = embeddings.cpu().numpy(), attn_weights.cpu().numpy()

    # Compute Spearman Correlation
    spearman_corr, _ = spearmanr(actual, predictions)

    # Save results
    results = {
        "spearman_correlation": spearman_corr
    }
    with open(os.path.join(category_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Scatterplot with Line of Best Fit
    actual_reshaped = np.array(actual).reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(actual_reshaped, predictions)
    line_of_best_fit = regressor.predict(actual_reshaped)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(actual, predictions, alpha=0.6, label="Predictions")
    plt.plot(actual, line_of_best_fit, color="blue", linestyle="-", label="Line of Best Fit")
    plt.text(
        0.05, 0.95,
        f"Spearman Correlation: {spearman_corr:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    plt.xlabel("Ground Truth Complexity Score")
    plt.ylabel("Predicted Complexity Score")
    plt.title("Predicted vs Ground Truth Complexity Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(category_output_dir, "pred_vs_gt.png"))
    plt.close()

    # Heatmap Visualization of Attention Weights
    if example_embeddings is not None:
        plt.figure(figsize=(10, 6))
        sns.heatmap(example_attn_weights, cmap="coolwarm", annot=False)
        plt.xlabel("Embedding Dimensions")
        plt.ylabel("Attention Head")
        plt.title("Attention Weights for One Image")
        plt.savefig(os.path.join(category_output_dir, "attention_heatmap.png"))
        plt.close()

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, test_loader, device, config, category_output_dir):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        for batch in train_loader:
            embeddings, targets = batch
            embeddings, targets = embeddings.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(embeddings)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings, targets = batch
                embeddings, targets = embeddings.to(device), targets.to(device)

                outputs, _ = model(embeddings)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)

    # Evaluate the trained model
    evaluate_model(model, test_loader, category_output_dir)

    return model

# ---------------------------
# Training Pipeline
# ---------------------------
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 250,
        "early_stopping_patience": 20,
        "batch_size": 16,
        "num_heads": 4,
        "hidden_dim": 128,
        "ffn_dim": 64,
        "dropout": 0.1
    }

    categories = os.listdir(EMBEDDING_DIR)
    categories = [c for c in categories if os.path.isdir(os.path.join(EMBEDDING_DIR, c))]

    print(f"Found categories: {categories}")

    for category in categories:
        print(f"\nProcessing category: {category}")

        # Construct correct file paths
        embedding_file = os.path.join(EMBEDDING_DIR, category, "static_embedding.csv")
        gt_file = os.path.join(GT_DIR, f"global_ranking_{category.lower().replace('_output', '')}.csv")

        if not os.path.exists(embedding_file) or not os.path.exists(gt_file):
            print(f"Skipping {category}: Missing files.")
            continue

        # Create category-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_output_dir = os.path.join(OUTPUT_DIR, f"{category}_results_{timestamp}")
        os.makedirs(category_output_dir, exist_ok=True)

        # Load CSV files
        embeddings_df = pd.read_csv(embedding_file)
        gt_df = pd.read_csv(gt_file)
        embeddings_df["gt"] = gt_df["gt"]

        # Prepare features and targets
        feature_columns = [str(i) for i in range(66)]
        X = embeddings_df[feature_columns].values.astype(np.float32)
        y = torch.tensor(embeddings_df["gt"].values, dtype=torch.float32).view(-1, 1)

        # Split into datasets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=config["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=config["batch_size"], shuffle=False)
        test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=config["batch_size"], shuffle=False)

        # Train model
        model = CrossDimensionalAttention(embed_dim=66, num_heads=config["num_heads"], hidden_dim=config["hidden_dim"]).to(device)
        model = train_model(model, train_loader, val_loader, test_loader, device, config, category_output_dir)

if __name__ == "__main__":
    train_pipeline()
