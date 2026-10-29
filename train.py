"""
pricing_tool/train.py
Trains the LightGBM Tweedie pricing model on total_claim_amount from cached Parquet data.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
from data_prep import load_pricing_data

def train_and_save_models():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("submission").mkdir(parents=True, exist_ok=True)
    
    print("\n==============================================")
    print("Training Actuarial Tweedie Model (Parquet Mode)")
    print("==============================================")
    
    df = load_pricing_data(mode="train")
    
    y_col = 'total_claim_amount'
    if y_col not in df.columns:
        raise ValueError(f"Target column '{y_col}' not found in training cache! Check data_prep.py.")
        
    print(f"Using actuarial target column: '{y_col}' (Total Sum = {pd.to_numeric(df[y_col], errors='coerce').sum():,.2f})")
    y = pd.to_numeric(df[y_col], errors='coerce').fillna(0.0)
    
    # Exclude IDs, target, and non-feature columns strictly
    exclude_cols = ['row_id', 'id', 'customer_id', 'address_id', 'policy_id', 'renewal_decision', 'total_claim_amount', 'claim_count']
    feature_cols = [c for c in df.select_dtypes(include=[np.number, bool]).columns if c not in exclude_cols]
    
    X = df[feature_cols].astype(float)
    print(f"Training on {X.shape[1]} features across {len(X):,} rows.")
    
    # Exposure weighting
    exposure_col = next((c for c in ['exposure', 'Tenure', 'duration'] if c in df.columns), None)
    if exposure_col:
        exposure = pd.Series(np.maximum(pd.to_numeric(df[exposure_col], errors='coerce').fillna(1.0), 0.01))
    else:
        exposure = pd.Series(np.ones(len(df)))
    
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'tweedie',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'n_jobs': -1,
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
        
    model_path = Path("models") / "pricing_tweedie_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nSaved correct actuarial model binary to {model_path}")

if __name__ == "__main__":
    train_and_save_models()