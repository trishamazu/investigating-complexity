import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from datetime import datetime

# number of repeats
N_RUNS = 100

# load embeddings and ground truth complexity scores
embedding_file = '/home/wallacelab/complexity-final/Savoias/Advertisement_output/static_embedding.csv'
complexity_file = '/home/wallacelab/complexity-final/Images/Savoias-Dataset/Ground truth/csv/global_ranking_ad.csv'
df_embeddings = pd.read_csv(embedding_file)
df_complexity = pd.read_csv(complexity_file)

# extract predictors and ground truth scores
X = df_embeddings.iloc[:, 1:66].values  
y = df_complexity.iloc[:, 0].values  

# normalize embeddings from 0 to 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# initialize lists to store results
rmse_list = []
r2_list = []
spearman_list = []
selected_features_rfe_all = np.zeros((N_RUNS, X.shape[1]))
selected_features_lasso_all = np.zeros((N_RUNS, X.shape[1]))

# store predictions for averaging
all_y_test = []
all_y_pred = []

# perform 100 train/test splits and run regression
for i in range(N_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i)

    # train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Store results and predictions
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_list.append(r2_score(y_test, y_pred))
    spearman_corr, _ = spearmanr(y_test, y_pred)
    spearman_list.append(spearman_corr)
    all_y_test.append(y_test)
    all_y_pred.append(y_pred)

    # feature selection using RFE
    rfe = RFE(model, n_features_to_select=10)
    rfe.fit(X_train, y_train)
    selected_features_rfe_all[i, :] = rfe.support_

    # feature selection using LASSO
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    selected_features_lasso_all[i, :] = (lasso.coef_ != 0)

# compute averages
avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)
avg_spearman = np.mean(spearman_list)

# identify the most frequently selected features
top_features_rfe = np.argsort(np.sum(selected_features_rfe_all, axis=0))[-10:].tolist()
top_features_lasso = np.argsort(np.sum(selected_features_lasso_all, axis=0))[-10:].tolist()

# compute average predictions
y_test_avg = np.concatenate(all_y_test)
y_pred_avg = np.concatenate(all_y_pred)

# calculate best-fit line
regressor = LinearRegression()
regressor.fit(y_test_avg.reshape(-1, 1), y_pred_avg)
line_of_best_fit = regressor.predict(y_test_avg.reshape(-1, 1))

# create identifiable output folder name
category_name = re.search(r'/([^/]+)_output/', embedding_file)
category_name = category_name.group(1).lower() if category_name else "unknown"
output_folder = f"/home/wallacelab/complexity-final/Savoias_output/linear/{category_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_folder, exist_ok=True)

# save results
results_file = os.path.join(output_folder, "results.txt")
with open(results_file, "w") as f:
    f.write(f"Average RMSE: {avg_rmse:.4f}\n")
    f.write(f"Average R² Score: {avg_r2:.4f}\n")
    f.write(f"Average Spearman Correlation: {avg_spearman:.4f}\n")
    f.write(f"Top Features (RFE over {N_RUNS} runs): {top_features_rfe}\n")
    f.write(f"Top Features (LASSO over {N_RUNS} runs): {top_features_lasso}\n")

# plot and save image
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(y_test_avg, y_pred_avg, alpha=0.2, label='Predictions')
plt.plot(y_test_avg, line_of_best_fit, color='blue', linestyle='-', label='Average Line of Best Fit')
plt.text(
    0.05, 0.95,  
    f"Avg Spearman Correlation: {avg_spearman:.4f}",
    transform=plt.gca().transAxes,  
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)  
)          
plt.xlabel("Ground Truth Complexity Score")
plt.ylabel("Predicted Complexity Score")
plt.title(f"Predicted vs Ground Truth Complexity Scores ({category_name})")
plt.legend()
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_folder, "pred_vs_gt.png")
plt.savefig(plot_path)
plt.close()

print(f"Results saved in: {output_folder}")
