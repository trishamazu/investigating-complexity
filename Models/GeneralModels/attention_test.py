import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import os
import datetime

classnames66 = [
    'metallic; artificial',
    'food-related',
    'animal-related',
    'textile',
    'plant-related',
    'house-related; furnishing-related',
    'valuable; precious',
    'transportation; movement-related',
    'body; people-related',
    'wood-related; brown',
    'electronics; technology',
    'colorful; playful',
    'outdoors',
    'circular; round',
    'paper-related; flat',
    'hobby-related; game-related; playing-related',
    'tools-related; handheld; elongated',
    'fluid-related; drink-related',
    'water-related',
    'oriented; many; plenty',
    'powdery; earth-related; waste-related',
    'white',
    'coarse-scale pattern; many things',
    'red',
    'long; thin',
    'weapon-related; war-related; dangerous',
    'black',
    'household-related',
    'feminine',
    'body-part-related',
    'tubular',
    'music-related; hearing-related; hobby-related; loud',
    'grid-related; grating-related',
    'repetitive; spiky',
    'construction-related; craftsmanship-related; housework-related',
    'spherical; voluminous',
    'string-related; stringy; curved',
    'seating; standing; lying-related',
    'flying-related; sky-related',
    'bug-related; non-mammalian; disgusting',
    'transparent; shiny; crystalline',
    'sand-colored',
    'green',
    'bathroom-related; wetness-related',
    'yellow',
    'heat-related; fire-related; light-related',
    'beams-related; mesh-related',
    'foot-related; walking-related',
    'box-related; container',
    'stick-shaped; container',
    'head-related',
    'upright; elongated; volumous',
    'pointed; spiky',
    'child-related; toy-related; cute',
    'farm-related; historical',
    'seeing-related',
    'medicine-related; health-related',
    'sweet; dessert-related',
    'orange',
    'thin; flat; wrapping',
    'cylindrical; conical; cushioning',
    'coldness-related; winter-related',
    'measurement-related; numbers-related',
    'fluffy; soft',
    'masculine',
    'fine-grained; pattern'
]


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GatedComplexityModel(nn.Module):
    def __init__(self, in_dim=66, hidden_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.gate(x)
        gated_input = x * attn_weights
        output = self.mlp(gated_input)
        return output, attn_weights

    def get_attention(self, x):
        return self.gate(x)
    

from transformers import BertTokenizer, BertModel

class BERTFeatureAttentionModel(nn.Module):
    def __init__(self, feature_names, hidden_dim=128, num_heads=2, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.feature_names = feature_names

        # Load BERT
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT

        self.embedding_dim = self.bert.config.hidden_size

        self.attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def encode_feature_texts(self, device):
        inputs = self.tokenizer(self.feature_names, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def forward(self, x):
        # x: [batch, num_features]
        batch_size, num_features = x.shape
        device = x.device

        text_embs = self.encode_feature_texts(device)  # [num_features, embedding_dim]
        text_embs = text_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_features, embedding_dim]

        scalars = x.unsqueeze(-1)  # [batch, num_features, 1]
        modulated = text_embs * scalars

        attn_output, attn_weights = self.attn(modulated, modulated, modulated)  # shape: [batch, num_features, embedding_dim]
        pooled = attn_output.mean(dim=1)
        output = self.mlp(pooled)

        return output, attn_weights
    
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPFeatureAttentionModel(nn.Module):
    def __init__(self, feature_names, hidden_dim=128, num_heads=2, clip_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.feature_names = feature_names

        # Load CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip = CLIPTextModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False  # freeze CLIP

        self.embedding_dim = self.clip.config.hidden_size

        self.attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # constrain predictions between 0 and 1 if IC9600
        )

    def encode_feature_texts(self, device):
        if hasattr(self, 'cached_text_embs'):
            return self.cached_text_embs.to(device)
        
        inputs = self.tokenizer(self.feature_names, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip(**inputs)
        self.cached_text_embs = outputs.last_hidden_state[:, 0, :].detach()
        return self.cached_text_embs.to(device)


    def forward(self, x):
        # x: [batch, num_features]
        batch_size, num_features = x.shape
        device = x.device

        text_embs = self.encode_feature_texts(device)  # [num_features, embedding_dim]
        text_embs = text_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_features, embedding_dim]

        scalars = x.unsqueeze(-1)  # [batch, num_features, 1]
        modulated = text_embs * scalars

        attn_output, attn_weights = self.attn(modulated, modulated, modulated)
        pooled = attn_output.mean(dim=1)
        output = self.mlp(pooled)

        return output, attn_weights



def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, patience, model_save_path):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0.0

        # tqdm loader with loss info
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        for x_batch, y_batch in train_iter:
            x_batch, y_batch = x_batch.to(model.device), y_batch.to(model.device)

            optimizer.zero_grad()
            outputs, _ = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)
            train_iter.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print("Validation improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break



def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0

    val_iter = tqdm(data_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for x_batch, y_batch in val_iter:
            x_batch, y_batch = x_batch.to(model.device), y_batch.to(model.device)
            outputs, _ = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            val_iter.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def plot_loss_curve(train_losses, val_losses, fold_num=None, save_path=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    title = f"Loss Curve - Fold {fold_num}" if fold_num is not None else "Loss Curve - Overall"
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_scatter(gt, pred, spearman_corr, fold_num=None, save_path=None):
    plt.figure()
    plt.scatter(gt, pred, alpha=0.6)
    m, b = np.polyfit(gt, pred, 1)
    plt.plot(gt, m * np.array(gt) + b, color='red')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    title = f"Scatter Plot - Fold {fold_num}" if fold_num is not None else "Scatter Plot - Overall"
    plt.title(f"{title}\nSpearman: {spearman_corr:.4f}")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def train_and_evaluate(model_class, model_kwargs, X_tensor, y_tensor, device, use_cv=True, n_splits=5, base_output_dir='cv_results'):
    if not use_cv:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds, all_gts = [], []
    all_train_losses, all_val_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = model_class(**model_kwargs).to(device)
        model.device = device
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(max_epochs):
            model.train()
            total_train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * x_batch.size(0)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            avg_val_loss = evaluate(model, val_loader, criterion)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold{fold+1}.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Store losses
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Predictions
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                out, _ = model(x_batch)
                preds.extend(out.cpu().numpy().flatten())
                gts.extend(y_batch.numpy().flatten())

        spearman_corr = spearmanr(gts, preds).correlation
        print(f"Fold {fold+1} Spearman: {spearman_corr:.4f}")
        plot_scatter(gts, preds, spearman_corr, fold_num=fold+1,
                     save_path=os.path.join(output_dir, f'scatter_fold{fold+1}.png'))
        plot_loss_curve(train_losses, val_losses, fold_num=fold+1,
                        save_path=os.path.join(output_dir, f'loss_fold{fold+1}.png'))

        all_preds.extend(preds)
        all_gts.extend(gts)

    # Overall results
    overall_spearman = spearmanr(all_gts, all_preds).correlation
    print(f"\n==== Overall Spearman: {overall_spearman:.4f} ====")
    plot_scatter(all_gts, all_preds, overall_spearman,
                 save_path=os.path.join(output_dir, f'scatter_overall.png'))

    # Average loss curve
    max_len = max(len(l) for l in all_train_losses)
    avg_train = np.mean([np.pad(l, (0, max_len - len(l)), 'edge') for l in all_train_losses], axis=0)
    avg_val = np.mean([np.pad(l, (0, max_len - len(l)), 'edge') for l in all_val_losses], axis=0)
    plot_loss_curve(avg_train, avg_val,
                    save_path=os.path.join(output_dir, f'loss_overall.png'))


if __name__ == "__main__":

    seed_everything(42)


    # data
    csv_path = "/home/wallacelab/investigating-complexity/Embeddings/CLIP-HBA/IC9600/Categories/scenes_embeddings.csv"
    target_score_path = "/home/wallacelab/investigating-complexity/Images/IC9600/GroundTruthNoNames/scenes.csv"
    model_save_path = '/home/wallacelab/investigating-complexity/output/CLIP-HBA/IC9600Output/model_weights/GatedMLP1Scenes_model.pth'
    criterion = nn.MSELoss()
    train_portion = 0.8
    batch_size = 40
    max_epochs = 1000
    patience = 25
    learning_rate = 3e-5
    hidden_dim = 128
    num_heads = 4
    cuda = 0 # {0: cuda0, 1: cuda1, -1: cpu}
    input_embeddings = pd.read_csv(csv_path, header=0, index_col=0).to_numpy()
    print("input_embeddings shape:")
    print(input_embeddings.shape) # (200, 66) (200 samples, 66 features)

    target_scores = pd.read_csv(target_score_path, header=0).to_numpy()
    print("target_scores shape:")
    print(target_scores.shape) # (200, 1) (200 samples, 1 target score)


    # Prepare data
    X = torch.tensor(input_embeddings, dtype=torch.float32)
    y = torch.tensor(target_scores, dtype=torch.float32)


    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * train_portion)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # try the 3 diferent model architecturesbelow 
    # ----------------------
    # model = GatedComplexityModel(in_dim=X.shape[1], hidden_dim=hidden_dim)
    # model = BERTFeatureAttentionModel(feature_names=classnames66, hidden_dim=hidden_dim, num_heads=num_heads)
    model = CLIPFeatureAttentionModel(feature_names=classnames66, hidden_dim=hidden_dim, num_heads=num_heads)
    # ----------------------


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    if cuda == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif cuda == 1:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # device = torch.device(f"cuda:{cuda}")

    model.to(device)
    model.device = device  # Attach device info to model
    criterion.to(device)

    # print all trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    # Train the model
    train(model, train_loader, val_loader, optimizer, criterion, max_epochs, patience, model_save_path)

    use_cross_validation = True  # << toggle CV mode

    if use_cross_validation:
        model_class = GatedComplexityModel  # Change this to your desired model
        # model_class = BERTFeatureAttentionModel
        # model_class = CLIPFeatureAttentionModel

        model_kwargs = {
            "in_dim": X.shape[1],
            "hidden_dim": hidden_dim
        } if model_class == GatedComplexityModel else {
            "feature_names": classnames66,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads
        }

        train_and_evaluate(
            model_class=model_class,
            model_kwargs=model_kwargs,
            X_tensor=X,
            y_tensor=y,
            device=device,
            use_cv=True,
            n_splits=5,
            base_output_dir='/home/wallacelab/investigating-complexity/output/CLIP-HBA/IC9600Output/cv_results'
        )
    else:
        # Existing single train call
        train(model, train_loader, val_loader, optimizer, criterion, max_epochs, patience, model_save_path)