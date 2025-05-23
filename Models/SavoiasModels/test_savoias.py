"""
test.py - Train Transformer Model for Human Rating Prediction
===========================================================

Purpose:
--------
This script trains a Transformer-based model to predict human-assigned complexity
or aesthetic scores based on pre-computed static image embeddings (e.g., CLIP).
It handles data loading, preprocessing (scaling), model definition, training
with validation, evaluation, and saving of results.

Key Features:
-------------
- Loads image embeddings and corresponding human ratings.
- Splits data into training, validation, and test sets.
- Standardizes features using StandardScaler.
- Defines a Transformer encoder model with learnable attention pooling.
- Trains the model using MSE loss and AdamW optimizer.
- Monitors training and validation loss, Spearman & Pearson correlation during training.
- Evaluates the final model on the test set, reporting MSE, MAE, RMSE, RMAE, SRCC, and PCC.
- Saves the trained model state, feature scaler, and a baseline vector (mean training feature).
- Generates plots for training progress and test set performance.
- Logs progress and results to both console and a 'training.log' file.

Usage:
------
1. Ensure your data directory (specified by DATA_DIR constant) contains:
   - `static_embedding.csv`: CSV with 'image' column (ID) and feature columns ('0' to 'N-1').
   - `train.txt`: Space-separated file with 'image_id target_value' for training.
   - `test.txt`: Space-separated file with 'image_id target_value' for testing.
2. Modify the constants (e.g., DATA_DIR, hyperparameters) in the script if needed.
3. Run the script from the command line:
   ```bash
   python test.py
   ```

Outputs:
--------
- `transformer_predictor_state.pth`:
    - **What:** Contains the learned weights and biases (the "state") of the Transformer model after training is complete. This file allows reloading the trained model later for inference or further training.
    - **How:** Generated by calling `torch.save(model.state_dict(), MODEL_SAVE_PATH)` after the training loop finishes.

- `scaler.joblib`:
    - **What:** A saved `StandardScaler` object from scikit-learn. It stores the mean and standard deviation calculated from the *training* data features. This is crucial for applying the exact same scaling to validation, test, or new data before feeding it to the model.
    - **How:** Generated by calling `joblib.dump(scaler, SCALER_SAVE_PATH)` within the `load_data` function after fitting the scaler (`scaler.fit_transform(X_train_raw)`).

- `train_feature_mean.pt`:
    - **What:** A PyTorch tensor containing the mean vector of the *scaled* training features (after applying the StandardScaler). This can serve as a simple baseline input for interpretability methods like Integrated Gradients.
    - **How:** Calculated by taking the mean (`X_train.mean(axis=0)`) of the scaled training data `X_train` and saving it using `torch.save()`.

- `pooling_weights_alpha.pt`:
    - **What:** A PyTorch tensor containing the learned parameters (weights and bias) of the `attention_weights` linear layer within the `AttnPool` module. These weights determine how the attention scores are calculated for pooling the Transformer's output sequence.
    - **How:** Saved by calling `torch.save(model.attn_pool.attention_weights.state_dict(), POOLING_WEIGHTS_SAVE_PATH)` *This part seems to be missing from the current main execution block but was likely intended. If needed, it would save the state dict of the attention layer.* (Correction: The current code does not save this specific file. The comment reflects the original intention based on the variable name.)

- `training_loss_curve.png`:
    - **What:** A plot showing the training loss (MSE) and validation loss (MSE) curves over the course of the training epochs. Helps visualize learning progress and identify potential overfitting (validation loss increasing while training loss decreases).
    - **How:** Generated by the `plot_loss_curve` function, which uses matplotlib to plot the `train_losses` and `val_losses` lists collected during the `train_model_with_validation` loop.

- `spearman_correlation_curve.png`:
    - **What:** A plot showing the Spearman rank correlation coefficient (SRCC or ρ) calculated on the *validation* set at the end of each epoch. Tracks how well the model's predicted ranking matches the true ranking on unseen validation data as training progresses.
    - **How:** Generated by the `plot_spearman_curve` function, using matplotlib to plot the `val_spearmans` list collected during the `train_model_with_validation` loop.

- `test_predictions_scatter.png`:
    - **What:** A scatter plot comparing the model's final predictions against the true target scores for the *test* set. Includes the overall Spearman correlation on the test set. Provides a visual assessment of the model's performance on completely unseen data.
    - **How:** Generated by the `plot_predictions` function. It takes the final predictions (`test_preds`) and true targets (`test_targets`) obtained from running `evaluate_model` on the test loader after training.

- `training.log`:
    - **What:** A text file containing detailed logs of the script's execution, including setup information (device used, seeds), data loading steps (skipped lines, sample counts), epoch-wise training/validation metrics (losses, correlations), final evaluation results, and any warnings or errors encountered.
    - **How:** Generated by the Python `logging` module configured at the beginning of the script to write messages to both the console and this file.

Dependencies:
-------------
- PyTorch
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Tqdm
- Joblib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import joblib
import math
import torch.nn.functional as F # Needed for softmax
import datetime
import optuna
from optuna.trial import TrialState
import argparse


# Create timestamped output folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("/home/wallacelab/investigating-complexity/output/new", f"run_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "training.log")
# --- Configure Logging ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Constants ---
DATA_DIR = "/home/wallacelab/investigating-complexity/Embeddings/CLIP-HBA/Savoias/Objects_output"
EMBEDDING_FILE = os.path.join(DATA_DIR, "static_embedding.csv")
TARGET_CSV = os.path.join(DATA_DIR, "/home/wallacelab/investigating-complexity/Images/Savoias-Dataset/GroundTruth/csv/global_ranking_objects.csv")  # Single-column CSV, no header, in order

NUM_FEATURES = 66
D_MODEL = 64
N_HEAD = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 4 * D_MODEL
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
SEED = 42

VALIDATION_SPLIT = 0.1
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "transformer_predictor_state.pth")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
BASELINE_SAVE_PATH = os.path.join(OUTPUT_DIR, "train_feature_mean.pt")
POOLING_WEIGHTS_SAVE_PATH = os.path.join(OUTPUT_DIR, "pooling_weights_alpha.pt")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss_curve.png")
PREDICTION_PLOT_PATH = os.path.join(OUTPUT_DIR, "test_predictions_scatter.png")
SPEARMAN_PLOT_PATH = os.path.join(OUTPUT_DIR, "spearman_correlation_curve.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
logging.info(f"Set random seeds to {SEED}")

# --- Data Loading and Preprocessing ---

def parse_id_target_file(filepath):
    """Parses the train/test file (image_id target_value)."""
    ids = []
    targets = []
    skipped_lines = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    image_id = parts[0]
                    target_value = float(parts[1])
                    ids.append(image_id)
                    targets.append(target_value)
                except ValueError:
                    logging.warning(f"Skipping line {i+1} in {filepath}: Invalid target value '{parts[1]}'")
                    skipped_lines += 1
            elif len(parts) == 1:
                logging.warning(f"Skipping line {i+1} in {filepath}: Missing target value for '{parts[0]}'")
                skipped_lines += 1
            elif len(parts) > 2:
                 logging.warning(f"Skipping line {i+1} in {filepath}: Unexpected format '{line.strip()}'")
                 skipped_lines += 1
            # Ignore empty lines silently
    if skipped_lines > 0:
        logging.warning(f"Skipped {skipped_lines} lines in {filepath} due to formatting or missing/invalid targets.")
    return ids, targets

def load_data(embedding_file, target_csv_file, val_split=0.1, test_split=0.2, random_state=SEED):
    """Loads embeddings and targets, splits train/val, scales, and saves scaler."""
    # Load embeddings
    df = pd.read_csv(embedding_file, dtype={'image': str})
    feature_cols = [str(i) for i in range(NUM_FEATURES)]
    X_all = df[feature_cols].values.astype(np.float32)

    # Load target scores (1D array, no header)
    y_all = pd.read_csv(target_csv_file).values.flatten().astype(np.float32)

    if len(X_all) != len(y_all):
        raise ValueError("Number of embeddings and number of targets must match.")

    # Normalize targets to [0, 1]
    y_min = y_all.min()
    y_max = y_all.max()
    y_all_normalized = (y_all - y_min) / (y_max - y_min)

    # Split into train/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all_normalized, test_size=test_split, random_state=random_state, shuffle=True
    )

    # Then split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split, random_state=random_state, shuffle=True
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, y_min, y_max


class HumanRatingDataset(Dataset):
    """PyTorch Dataset for human rating data."""
    def __init__(self, features, targets):
        # Ensure features and targets are float32 for PyTorch
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1) # Ensure target shape is (N, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- Model Definition ---

class AttnPool(nn.Module):
    """Learnable Attention Pooling layer."""
    def __init__(self, d_model):
        super().__init__()
        # Simple linear layer to compute attention scores
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, h):
        # h: (B, S, d_model) - Output from transformer encoder
        # Compute scores
        scores = self.attention_weights(h) # (B, S, 1)
        # Apply softmax over the sequence dimension (S)
        alpha = F.softmax(scores, dim=1) # (B, S, 1)
        # Weighted sum
        pooled = torch.sum(h * alpha, dim=1) # (B, d_model)
        # Return both pooled output and attention weights (alpha)
        # Squeeze alpha for easier handling: (B, S)
        return pooled, alpha.squeeze(-1)

class TransformerPredictor(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, return_attn=False):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        # Note: return_attn flag is primarily for the encoder's MHA, not pooling
        self.return_attn = return_attn

        # 1. Token embedding
        self.token_embed = nn.Linear(1, d_model)

        # 3. Dimension position embedding
        self.dim_embed = nn.Embedding(num_features, d_model)
        self.register_buffer('dim_id', torch.arange(num_features))

        # 4. Self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        encoder_layer.self_attn.average_attn_weights = False
        # No change needed here for pooling

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 5. Read-out head (REPLACED mean-pooling with AttnPool)
        # self.pool = lambda h: h.mean(dim=1) # Old mean pooling
        self.attn_pool = AttnPool(d_model) # New attention pooling layer
        logging.info("Using AttnPool instead of Mean Pooling.")

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _encode_with_attn(self, h):
        # First, run the transformer encoder normally
        encoded_h = self.transformer_encoder(h)
        
        # Now calculate attention weights separately based on the encoded features
        # This is a simplified approach that doesn't extract the exact internal attention
        # weights but gives a close approximation
        
        # Create attention map representing feature relationships
        batch_size, seq_len, d_model = encoded_h.shape
        
        # Method 1: Simple dot-product between features
        # Normalize features for better attention visualization
        normalized_h = F.normalize(encoded_h, p=2, dim=2)  # Normalize along feature dimension
        
        # Calculate dot product attention between all positions
        attn_map = torch.bmm(normalized_h, normalized_h.transpose(1, 2))
        
        # Apply softmax to get proper attention weights
        attn_map = F.softmax(attn_map, dim=2)
        
        logging.info(f"Successfully calculated attention map with shape {attn_map.shape}")
        
        return encoded_h, attn_map

    def forward(self, x_raw, require_maps=False):
        # x_raw: (B, num_features)

        # 1. Reshape and apply token embedding
        x = x_raw.unsqueeze(-1)         # (B, S, 1), S=num_features
        x_tok = self.token_embed(x)     # (B, S, d_model)

        # 3. Add dimension ID embedding
        # Ensure dim_id is on the same device as x_tok
        dim_ids = self.dim_id.to(x_tok.device)
        h_in = x_tok + self.dim_embed(dim_ids) # (B, S, d_model)

        # 4. Pass through Transformer Encoder (potentially extracting attention)
        if require_maps:
            # Use the new helper method
            h, A = self._encode_with_attn(h_in) # h:(B,S,d_model), A:(B,S,S)
        else:
            # Original path if maps not needed
            h = self.transformer_encoder(h_in)  # (B, S, d_model)
            A = None # No attention map calculated

        # 5. Pool using AttnPool and Predict
        pooled, alpha = self.attn_pool(h)    # pooled: (B, d_model), alpha: (B, S)
        pred = self.prediction_head(pooled)  # (B, 1)

        if require_maps:
            # Return prediction, pooling weights (alpha), and mean self-attention map (A)
            # Ensure A is not None if require_maps=True
            if A is None:
                 # This case shouldn't happen if _encode_with_attn works, but good practice
                 logging.error("Attention map requested but not generated.")
                 # Decide how to handle: raise error or return dummy
                 A = torch.zeros(h.size(0), self.num_features, self.num_features, device=h.device)
            return pred, alpha, A
        else:
            # Original return value if maps not needed
            return pred

# --- Training and Validation Loop ---
def run_epoch(model, loader, criterion, optimizer, device, is_training):
    """Runs a single epoch, returns loss, predictions, and targets."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    epoch_preds = []
    epoch_targets = []
    desc = "Training" if is_training else "Validating"
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)

    with torch.set_grad_enabled(is_training):
        for batch_features, batch_targets in pbar:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * batch_features.size(0)
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

            # Collect predictions and targets for correlation calculation
            epoch_preds.append(outputs.detach().cpu().numpy())
            epoch_targets.append(batch_targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    # Concatenate predictions and targets from all batches
    all_epoch_preds = np.concatenate(epoch_preds, axis=0).flatten()
    all_epoch_targets = np.concatenate(epoch_targets, axis=0).flatten()

    # Calculate metrics (MSE, MAE, SRCC, PCC, RMSE, RMAE)
    metrics = {}
    metrics['mse'] = avg_loss
    try:
        metrics['mae'] = mean_absolute_error(all_epoch_targets, all_epoch_preds)
        metrics['rmse'] = math.sqrt(metrics['mse'])
        metrics['rmae'] = math.sqrt(metrics['mae'])
        metrics['srcc'], _ = calculate_spearman(all_epoch_preds, all_epoch_targets)
        metrics['pcc'], _ = calculate_pearson(all_epoch_preds, all_epoch_targets)
    except Exception as e:
        logging.warning(f"Could not calculate one or more metrics: {e}")
        metrics['mae'] = np.nan
        metrics['rmse'] = np.nan
        metrics['rmae'] = np.nan
        metrics['srcc'] = np.nan
        metrics['pcc'] = np.nan

    return avg_loss, metrics

def calculate_spearman(predictions, targets):
    """Safely calculates Spearman correlation."""
    try:
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        if np.std(predictions_flat) < 1e-6 or np.std(targets_flat) < 1e-6:
             logging.warning("Cannot calculate Spearman correlation: Input has zero variance.")
             return np.nan, np.nan
        corr, p_value = spearmanr(predictions_flat, targets_flat)
        return corr, p_value
    except ValueError as e:
        logging.warning(f"Could not calculate Spearman correlation: {e}")
        return np.nan, np.nan

def calculate_pearson(predictions, targets):
    """Safely calculates Pearson correlation."""
    try:
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        if np.std(predictions_flat) < 1e-6 or np.std(targets_flat) < 1e-6:
             logging.warning("Cannot calculate Pearson correlation: Input has zero variance.")
             return np.nan, np.nan
        corr, p_value = pearsonr(predictions_flat, targets_flat)
        return corr, p_value
    except ValueError as e:
        logging.warning(f"Could not calculate Pearson correlation: {e}")
        return np.nan, np.nan

def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """Trains the model, validates, and tracks multiple metrics."""
    train_losses = []
    val_losses = []
    val_spearmans = [] # Store validation Spearman correlation per epoch
    val_pearsons = [] # Store validation Pearson correlation per epoch
    logging.info("Starting training with validation...")

    for epoch in range(num_epochs):
        # Training phase
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        train_losses.append(train_loss)

        # Validation phase
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, is_training=False)
        val_losses.append(val_loss)
        val_spearmans.append(val_metrics.get('srcc', np.nan))
        val_pearsons.append(val_metrics.get('pcc', np.nan))

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val SRCC: {val_metrics.get('srcc', float('nan')):.4f}, "
            f"Val PCC: {val_metrics.get('pcc', float('nan')):.4f}, "
            f"Val RMSE: {val_metrics.get('rmse', float('nan')):.4f}, "
            f"Val RMAE: {val_metrics.get('rmae', float('nan')):.4f}" # Added RMAE
        )

    # Return losses and SRCC for plotting compatibility, could return all metrics
    return train_losses, val_losses, val_spearmans

# --- Evaluation ---
def evaluate_model(model, test_loader, criterion, device, y_min, y_max):
    """Evaluates the model and returns metrics, predictions, targets, and Spearman corr."""
    model.eval()
    logging.info("Starting final evaluation on test set...")
    _, test_metrics = run_epoch(model, test_loader, criterion, None, device, is_training=False)

    logging.info(f"Test Loss (MSE): {test_metrics.get('mse', float('nan')):.4f}")
    logging.info(f"Test MAE: {test_metrics.get('mae', float('nan')):.4f}")
    logging.info(f"Test RMSE: {test_metrics.get('rmse', float('nan')):.4f}")
    logging.info(f"Test RMAE: {test_metrics.get('rmae', float('nan')):.4f}")
    logging.info(f"Test SRCC: {test_metrics.get('srcc', float('nan')):.4f}")
    logging.info(f"Test PCC: {test_metrics.get('pcc', float('nan')):.4f}")

    all_preds_list = []
    all_targets_list = []
    with torch.no_grad():
        for batch_features, batch_targets in tqdm(test_loader, desc="Collecting Test Preds", unit="batch", leave=False):
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_preds_list.append(outputs.cpu().numpy())
            all_targets_list.append(batch_targets.cpu().numpy())

    all_preds = np.concatenate(all_preds_list, axis=0).flatten()
    all_targets = np.concatenate(all_targets_list, axis=0).flatten()

    # Denormalize
    all_preds = all_preds * (y_max - y_min) + y_min
    all_targets = all_targets * (y_max - y_min) + y_min

    return test_metrics['mse'], test_metrics['mae'], test_metrics['srcc'], all_preds, all_targets

# --- Plotting Functions ---
def plot_loss_curve(train_losses, val_losses, save_path):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        logging.info(f"Loss curve saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save loss curve plot: {e}", exc_info=True)
    plt.close()

def plot_predictions(predictions, targets, spearman_rho, save_path):
    """Creates a scatter plot of predictions vs. targets with Spearman correlation."""
    plt.figure(figsize=(8, 8))
    # Check for empty arrays before plotting
    if len(predictions) == 0 or len(targets) == 0:
        logging.warning("Cannot plot predictions: Data is empty.")
        plt.title("Test Set: Predicted vs. True Scores (No data)")
        plt.xlabel('True Scores')
        plt.ylabel('Predicted Scores')
    else:
        min_val = min(np.min(predictions), np.min(targets)) - 0.05
        max_val = max(np.max(predictions), np.max(targets)) + 0.05
        plt.scatter(targets, predictions, alpha=0.5, label='Predictions')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        plt.xlabel('True Scores')
        plt.ylabel('Predicted Scores')
        # Add Spearman correlation text
        title_str = 'Test Set: Predicted vs. True Scores'
        if not np.isnan(spearman_rho):
            title_str += f'\nSpearman ρ = {spearman_rho:.3f}'
        else:
            title_str += f'\nSpearman ρ = N/A'
        plt.title(title_str)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

    try:
        plt.savefig(save_path)
        logging.info(f"Prediction scatter plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save prediction plot: {e}", exc_info=True)
    plt.close()

def plot_spearman_curve(val_spearmans, save_path):
    """Plots validation Spearman correlation over epochs."""
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(val_spearmans) + 1)
    # Filter out NaNs before plotting
    valid_epochs = [e for e, s in zip(epochs, val_spearmans) if not np.isnan(s)]
    valid_spearmans = [s for s in val_spearmans if not np.isnan(s)]

    if valid_epochs:
        plt.plot(valid_epochs, valid_spearmans, marker='o', linestyle='-', label='Validation Spearman ρ')
        plt.ylim(min(min(valid_spearmans)-0.1, 0), max(max(valid_spearmans)+0.1, 1)) # Adjust y-limits based on valid data
    else:
        plt.text(0.5, 0.5, "No valid Spearman data to plot", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('Epoch')
    plt.ylabel('Spearman Correlation (ρ)')
    plt.title('Validation Spearman Correlation Over Epochs')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(save_path)
        logging.info(f"Spearman correlation curve saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save Spearman curve plot: {e}", exc_info=True)
    plt.close()

# Define the objective function for Optuna tuning
def objective(trial):
    # Suggest hyperparameters
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 4)
    dim_feedforward = trial.suggest_int("dim_feedforward", 64, 512, step=64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    model = TransformerPredictor(
        num_features=NUM_FEATURES,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(HumanRatingDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HumanRatingDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    _, _, val_spearmans = train_model_with_validation(
        model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=25
    )

    best_srcc = max([s for s in val_spearmans if not np.isnan(s)], default=-1.0)
    return best_srcc

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or tune Transformer model.")
    parser.add_argument('--tune', action='store_true', help="Run Optuna hyperparameter search")
    parser.add_argument('--config', type=str, default=None, help="Path to JSON config file with hyperparameters")
    args = parser.parse_args()

    # 1. Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_min, y_max = load_data(
        EMBEDDING_FILE, TARGET_CSV, val_split=VALIDATION_SPLIT
    )

    # --- Save baseline vector (mean of scaled training features) ---
    try:
        baseline_vec = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        torch.save(baseline_vec, BASELINE_SAVE_PATH)
        logging.info(f"Saved training feature mean baseline to {BASELINE_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Failed to save baseline vector: {e}", exc_info=True)

    run_evaluation = len(X_test) > 0
    if not run_evaluation:
        logging.warning("Test set is empty. Skipping final evaluation and plotting.")

    if args.tune:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        # Save best config
        import json
        BEST_CONFIG_PATH = os.path.join(OUTPUT_DIR, "best_config.json")
        with open(BEST_CONFIG_PATH, "w") as f:
            json.dump(study.best_params, f, indent=4)
        logging.info(f"Best hyperparameters saved to {BEST_CONFIG_PATH}")
        exit()

    # Load config if provided
    import json
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        model = TransformerPredictor(
            num_features=NUM_FEATURES,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        model = TransformerPredictor(
            num_features=NUM_FEATURES,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT
        ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = nn.MSELoss()

    # Create datasets and DataLoaders
    train_dataset = HumanRatingDataset(X_train, y_train)
    val_dataset = HumanRatingDataset(X_val, y_val)
    test_dataset = HumanRatingDataset(X_test, y_test) if run_evaluation else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if run_evaluation else None

    # Train the Model with Validation & Metric Tracking
    train_losses, val_losses, val_spearmans = train_model_with_validation(
        model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS
    )
    logging.info("Training finished.")

    # Plot Training Curves
    plot_loss_curve(train_losses, val_losses, LOSS_PLOT_PATH)
    plot_spearman_curve(val_spearmans, SPEARMAN_PLOT_PATH)

    # Evaluate the Model on Test Set (Optional)
    test_spearman = np.nan
    if run_evaluation and test_loader:
        _, _, test_spearman, test_preds, test_targets = evaluate_model(
            model, test_loader, criterion, DEVICE, y_min, y_max
        )
        logging.info("Evaluation finished.")

        # Plot Test Predictions vs Targets (Optional)
        plot_predictions(test_preds, test_targets, test_spearman, PREDICTION_PLOT_PATH)

    elif not run_evaluation:
        logging.info("Skipping final evaluation and prediction plotting as test set was empty.")

    # Save the Final Model State
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.info(f"Model state saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Failed to save model state: {e}", exc_info=True)

    logging.info("Training script finished.")