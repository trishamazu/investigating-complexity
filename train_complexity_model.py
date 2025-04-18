import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    # Paths to embeddings and ground truth
    base_dir = os.path.dirname(__file__)
    embedding_csv = os.path.join(base_dir, 'Embeddings', 'CLIP-HBA', 'Savoias', 'Objects_output', 'static_embedding.csv')
    gt_csv = os.path.join(base_dir, 'Images', 'Savoias-Dataset', 'GroundTruth', 'csv', 'global_ranking_objects_labeled.csv')

    # Load data
    emb_df = pd.read_csv(embedding_csv, index_col='image')
    gt_df = pd.read_csv(gt_csv, index_col='image')

    # Merge embeddings with ground truth
    df = emb_df.join(gt_df)
    if df.isnull().any().any():
        df = df.dropna()

    # Features and target
    X = df.iloc[:, :66].values
    y = df['gt'].values

    # Pipeline: polynomial features (including interactions), scaling, and ridge regression with CV
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    scaler = StandardScaler()
    # use ElasticNet to capture sparse and non-linear interactions with L1+L2
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=[0.1, 1.0, 10.0], cv=5)
    pipeline = Pipeline([('poly', poly), ('scaler', scaler), ('enet', enet)])

    # Train model
    print("Training model with polynomial interactions degree=2...")
    pipeline.fit(X, y)

    # Predictions and evaluation on training set
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"Training R^2: {r2:.4f}")
    print(f"Training MSE: {mse:.4f}")

    # Inspect top coefficients
    feature_names = poly.get_feature_names_out(input_features=emb_df.columns)
    coefs = pipeline.named_steps['enet'].coef_
    coef_series = pd.Series(coefs, index=feature_names)
    top_feats = coef_series.abs().sort_values(ascending=False).head(10)
    print("Top 10 features by absolute coefficient value:")
    for feat, val in top_feats.items():
        print(f"{feat}: {coef_series[feat]:.4f}")

    # Save trained pipeline
    model_path = os.path.join(base_dir, 'complexity_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
