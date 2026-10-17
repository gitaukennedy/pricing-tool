"""
pricing_tool/train.py
Trains LightGBM Tweedie pricing models on the hackathon portfolio dataset (CS2S26Q3.csv).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
from data_prep import load_pricing_data

def train_and_save_models(models_dir: str = "models", sub_dir: str = "submission"):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(sub_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n==============================================")
    print("Training Actuarial Pricing Model (CS2S26Q3)")
    print("==============================================")
    
    df = load_pricing_data(is_train=True)
    
    # Features, Target, and Exposure Offset
    X = df[["Premium", "Tenure"]]
    y = df["Claims"]
    exposure = np.maximum(df["Tenure"], 0.01)
    
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'tweedie',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'verbose': -1,
        'random_state': 42
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr, exp_tr = X.iloc[train_idx], y.iloc[train_idx], exposure.iloc[train_idx]
        X_va, y_va, exp_va = X.iloc[val_idx], y.iloc[val_idx], exposure.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=exp_tr)
        val_data = lgb.Dataset(X_va, label=y_va, weight=exp_va, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        score = model.best_score['valid_0']['tweedie']
        if score < best_score:
            best_score = score
            best_model = model
            
        print(f"Fold {fold+1} trained. Validation Tweedie Loss: {score:.4f}")
        
    # Save trained model binary
    model_path = Path(models_dir) / "pricing_tweedie_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nSaved model binary to {model_path}")
    
    # Generate predictions for dataset
    preds = np.maximum(0.0, best_model.predict(X))
    sub_df = pd.DataFrame({
        'row_id': df.index,
        'prediction': preds
    })
    
    sub_file = Path(sub_dir) / "final_submission.csv"
    sub_df.to_csv(sub_file, index=False)
    print(f"SUCCESS: Submission generated at {sub_file}")

if __name__ == "__main__":
    train_and_save_models()