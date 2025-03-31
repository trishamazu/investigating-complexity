import os
import re
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# ---------------------------
# Define directories and paths
# ---------------------------
EMBEDDING_DIR = "/home/wallacelab/complexity-final/Embeddings/CLIP-HBA/66d/Savoias"
GT_DIR = "/home/wallacelab/complexity-final/Images/Savoias-Dataset/GroundTruth/csv"
OUTPUT_DIR = "/home/wallacelab/complexity-final/output/Savoias_output/cross_attention/Run 5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Mapping category names to ground truth filenames
# ---------------------------
CATEGORY_GT_MAPPING = {
    "Scenes_output": "global_ranking_scenes.csv",
}

# ---------------------------
# CrossDimensionalAttention Model
# ---------------------------
class CrossDimensionalAttention(nn.Module):
    def __init__(self, embed_dim=66, num_heads=4, hidden_dim=128, ffn_dim=64, dropout=0.1):
        super(CrossDimensionalAttention, self).__init__()

        # Project input to hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Multi-head self-attention across dimensions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1)  # Output: Complexity Score
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension

        attn_output, attn_weights = self.attention(x, x, x)

        x = self.layer_norm(attn_output + x)  # Residual connection
        x = x.squeeze(1)  # Remove sequence dimension

        return self.ffn(x).view(-1, 1), attn_weights.squeeze(1)  # Ensure output is (batch_size, 1)

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, device, config):
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
            loss = criterion(outputs, targets.view(-1, 1))  # Ensure both are (batch_size, 1)
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
    return model, best_val_loss

# ---------------------------
# Function to generate attention heatmap
# ---------------------------
def generate_attention_heatmap(model, embedding, category, output_dir, device):
    model.eval()
    with torch.no_grad():
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        _, attention_weights = model(embedding_tensor)
        
        # Get attention weights matrix (reshaping as needed)
        attention_matrix = attention_weights.cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(attention_matrix, cmap="viridis", annot=False)
        plt.title(f"Attention Weights Across Dimensions - {category}")
        plt.xlabel("Embedding Dimensions")
        plt.ylabel("Attention Heads")
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, "attention_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        print(f"Attention heatmap saved to {heatmap_path}")

# ---------------------------
# Function to evaluate model performance
# ---------------------------
def evaluate_model(model, X_test, y_test, category, output_dir, device):
    model.eval()
    test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions, _ = model(test_data)
        predictions = predictions.cpu().numpy().flatten()
    
    # Get actual values
    actual = y_test.cpu().numpy().flatten()
    
    # Calculate Spearman correlation
    spearman_corr, p_value = spearmanr(actual, predictions)
    
    # Save evaluation metrics
    results = {
        "spearman_correlation": float(spearman_corr),
        "p_value": float(p_value)
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Spearman Correlation on Test Set: {spearman_corr:.4f}")
    
    # Calculate line of best fit
    actual_reshaped = actual.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(actual_reshaped, predictions)
    line_of_best_fit = regressor.predict(actual_reshaped)
    
    # Plot and save image
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
    plt.title(f"Predicted vs Ground Truth Complexity Scores ({category})")
    plt.legend()
    plt.tight_layout()
    
    # Save plot in output
    plot_path = os.path.join(output_dir, "pred_vs_gt.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Performance plot saved to {plot_path}")
    
    return spearman_corr

# ---------------------------
# Training Pipeline
# ---------------------------
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "learning_rate": 0.007030687051361586,
        "weight_decay": 1.2890015522930213e-05,
        "batch_size": 32,
        "num_heads": 2,
        "hidden_dim": 128,
        "ffn_dim": 128,
        "dropout": 0.00960993937876381
    }

    categories = os.listdir(EMBEDDING_DIR)
    categories = [c for c in categories if os.path.isdir(os.path.join(EMBEDDING_DIR, c))]

    print(f"Found categories: {categories}")
    
    # Store overall results
    overall_results = {}

    for category in categories:
        print(f"\nProcessing category: {category}")

        # Construct correct file paths
        embedding_file = os.path.join(EMBEDDING_DIR, category, "static_embedding.csv")
        gt_filename = CATEGORY_GT_MAPPING.get(category, None)

        if gt_filename:
            gt_file = os.path.join(GT_DIR, gt_filename)
        else:
            print(f"Skipping {category}: no ground truth mapping found.")
            continue

        if not os.path.exists(embedding_file):
            print(f"Skipping {category}: missing embeddings file at {embedding_file}")
            continue

        if not os.path.exists(gt_file):
            print(f"Skipping {category}: missing ground truth file at {gt_file}")
            continue

        # Create category-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        category_output_dir = os.path.join(OUTPUT_DIR, f"{category}_results_{timestamp}")
        os.makedirs(category_output_dir, exist_ok=True)

        # Load CSV files
        embeddings_df = pd.read_csv(embedding_file)
        gt_df = pd.read_csv(gt_file)

        # Merge ground truth into embeddings dataframe (assumes column name 'gt')
        embeddings_df["gt"] = gt_df["gt"]

        # Save merged CSV
        merged_csv_path = os.path.join(category_output_dir, "embeddings_with_gt.csv")
        embeddings_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to {merged_csv_path}")

        # Prepare features and targets
        feature_columns = [str(i) for i in range(66)]
        X = embeddings_df[feature_columns].values.astype(np.float32)
        y = torch.tensor(embeddings_df["gt"].values, dtype=torch.float32).view(-1, 1)  # Ensure shape (batch_size, 1)

        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(list(zip(torch.tensor(X_train), torch.tensor(y_train))), batch_size=config["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(list(zip(torch.tensor(X_val), torch.tensor(y_val))), batch_size=config["batch_size"], shuffle=False)

        # Initialize and train the model
        model = CrossDimensionalAttention(embed_dim=66, num_heads=config["num_heads"],
                                           hidden_dim=config["hidden_dim"], ffn_dim=config["ffn_dim"],
                                           dropout=config["dropout"]).to(device)

        print("Training model...")
        model, best_val_loss = train_model(model, train_loader, val_loader, device, config)

        # Save trained model
        torch.save(model.state_dict(), os.path.join(category_output_dir, "best_model.pth"))
        
        # Generate attention heatmap for one example image
        # Select an example from the test set (using the first one for simplicity)
        example_embedding = X_test[0]
        generate_attention_heatmap(model, example_embedding, category, category_output_dir, device)
        
        # Evaluate model on test set and generate performance plot
        spearman_corr = evaluate_model(model, X_test, y_test, category, category_output_dir, device)
        
        # Store results for this category
        overall_results[category] = {
            "spearman_correlation": float(spearman_corr),
            "best_val_loss": float(best_val_loss)
        }
        
        print(f"Model, visualizations, and results saved for category: {category}")

    # Save overall results
    with open(os.path.join(OUTPUT_DIR, "overall_results.json"), "w") as f:
        json.dump(overall_results, f, indent=4)
    
    print("\nTraining and evaluation completed for all categories.")

from sklearn.model_selection import KFold

def optuna_objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category = "Scenes_output"

    embedding_file = os.path.join(EMBEDDING_DIR, category, "static_embedding.csv")
    gt_file = os.path.join(GT_DIR, CATEGORY_GT_MAPPING[category])

    embeddings_df = pd.read_csv(embedding_file)
    gt_df = pd.read_csv(gt_file)
    embeddings_df["gt"] = gt_df["gt"]

    X = embeddings_df[[str(i) for i in range(66)]].values.astype(np.float32)
    y = embeddings_df["gt"].values.astype(np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    spearman_scores = []

    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
        "num_epochs": 150,
        "early_stopping_patience": 15,
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "ffn_dim": trial.suggest_categorical("ffn_dim", [32, 64, 128]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5)
    }

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader = torch.utils.data.DataLoader(
            list(zip(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            list(zip(torch.tensor(X_val), torch.tensor(y_val).view(-1, 1))),
            batch_size=64, shuffle=False
        )

        model = CrossDimensionalAttention(
            embed_dim=66,
            num_heads=config["num_heads"],
            hidden_dim=config["hidden_dim"],
            ffn_dim=config["ffn_dim"],
            dropout=config["dropout"]
        ).to(device)

        model, _ = train_model(model, train_loader, val_loader, device, config)

        model.eval()
        with torch.no_grad():
            val_preds, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
        val_preds = val_preds.cpu().numpy().flatten()
        val_actual = y_val.flatten()
        score, _ = spearmanr(val_actual, val_preds)

        if np.isnan(score):  # Sometimes Spearman can return NaN
            score = 0.0

        spearman_scores.append(score)

    avg_spearman = np.mean(spearman_scores)
    return -avg_spearman  # Optuna minimizes the objective

def cross_val_pipeline(config, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category = "Scenes_output"

    embedding_file = os.path.join(EMBEDDING_DIR, category, "static_embedding.csv")
    gt_file = os.path.join(GT_DIR, CATEGORY_GT_MAPPING[category])

    # Load data
    embeddings_df = pd.read_csv(embedding_file)
    gt_df = pd.read_csv(gt_file)
    embeddings_df["gt"] = gt_df["gt"]

    X = embeddings_df[[str(i) for i in range(66)]].values.astype(np.float32)
    y = torch.tensor(embeddings_df["gt"].values, dtype=torch.float32).view(-1, 1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {}

    all_preds = []
    all_actuals = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Further split train into train/val
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        train_loader = torch.utils.data.DataLoader(
            list(zip(torch.tensor(X_train_fold), torch.tensor(y_train_fold))),
            batch_size=config["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            list(zip(torch.tensor(X_val_fold), torch.tensor(y_val_fold))),
            batch_size=config["batch_size"], shuffle=False
        )

        model = CrossDimensionalAttention(
            embed_dim=66,
            num_heads=config["num_heads"],
            hidden_dim=config["hidden_dim"],
            ffn_dim=config["ffn_dim"],
            dropout=config["dropout"]
        ).to(device)

        model, best_val_loss = train_model(model, train_loader, val_loader, device, config)

        # Create fold-specific output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_dir = os.path.join(OUTPUT_DIR, f"{category}_fold{fold}_{timestamp}")
        os.makedirs(fold_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
        generate_attention_heatmap(model, X_test[0], category, fold_dir, device)
        spearman_corr = evaluate_model(model, X_test, y_test, category, fold_dir, device)

        # Store results for this fold
        fold_results[f"fold_{fold}"] = {
            "spearman_correlation": float(spearman_corr),
            "best_val_loss": float(best_val_loss)
        }

        # Store predictions for global scatter
        model.eval()
        with torch.no_grad():
            preds, _ = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        all_preds.extend(preds.cpu().numpy().flatten())
        all_actuals.extend(y_test.numpy().flatten())

    # Save all fold results
    with open(os.path.join(OUTPUT_DIR, f"{category}_crossval_results.json"), "w") as f:
        json.dump(fold_results, f, indent=4)

    # Final average correlation plot
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    spearman_corr, _ = spearmanr(all_actuals, all_preds)

    actual_reshaped = all_actuals.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(actual_reshaped, all_preds)
    line_of_best_fit = regressor.predict(actual_reshaped)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(all_actuals, all_preds, alpha=0.6, label="Predictions")
    plt.plot(all_actuals, line_of_best_fit, color="blue", linestyle="-", label="Line of Best Fit")
    plt.text(
        0.05, 0.95,
        f"Avg Spearman Correlation: {spearman_corr:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    plt.xlabel("Ground Truth Complexity Score")
    plt.ylabel("Predicted Complexity Score")
    plt.title("Cross-Validated Predictions vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "crossval_pred_vs_gt.png"))
    plt.close()
    print(f"Saved cross-validated average correlation plot.")

if __name__ == "__main__":
    best_config = {
        "learning_rate": 0.007030687051361586,
        "weight_decay": 1.2890015522930213e-05,
        "batch_size": 32,
        "num_heads": 2,
        "hidden_dim": 128,
        "ffn_dim": 128,
        "dropout": 0.00960993937876381,
        "num_epochs": 150,
        "early_stopping_patience": 15
    }

    cross_val_pipeline(best_config, n_splits=5)

#if __name__ == "__main__":
    #study = optuna.create_study(direction="minimize")
    #study.optimize(optuna_objective, n_trials=50)

    #print("Best trial:")
    #print(study.best_trial)

    # Save best params
    #with open(os.path.join(OUTPUT_DIR, "optuna_best_params_scenes.json"), "w") as f:
        #json.dump(study.best_trial.params, f, indent=4)

