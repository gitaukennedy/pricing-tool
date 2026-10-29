"""
train_p2.py
Upgraded training script for Part 2: Trains 3 Line-of-Business LightGBM Tweedie models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from pathlib import Path

def tweedie_deviance(y_true, y_pred, p=1.5, sample_weight=None):
    """Calculates official D2 Tweedie Deviance (p=1.5)."""
    eps = 1e-8
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, eps)
    
    dev = 2 * ( (np.power(y_true, 2 - p) / ((1 - p) * (2 - p))) 
                - (y_true * np.power(y_pred, 1 - p) / (1 - p)) 
                + (np.power(y_pred, 2 - p) / (2 - p)) )
    
    if sample_weight is not None:
        return np.average(dev, weights=sample_weight)
    return np.mean(dev)

def preprocess_features(df):
    df = df.copy()
    
    # Feature engineering & robustness
    if 'exposure' in df.columns:
        df['exposure'] = pd.to_numeric(df['exposure'], errors='coerce').fillna(1.0)
        df['exposure'] = np.maximum(df['exposure'], 0.001)
        
    if 'sum_insured' in df.columns:
        df['sum_insured'] = pd.to_numeric(df['sum_insured'], errors='coerce').fillna(0.0)
        
    if 'voluntary_excess' in df.columns and 'compulsory_excess' in df.columns:
        df['total_excess'] = pd.to_numeric(df['voluntary_excess'], errors='coerce').fillna(0) + \
                             pd.to_numeric(df['compulsory_excess'], errors='coerce').fillna(0)
                             
    if 'period_start' in df.columns:
        df['period_start'] = pd.to_datetime(df['period_start'], errors='coerce')
        df['start_month'] = df['period_start'].dt.month
        
    # Exclude IDs, dates, and leakage variables (like surname)
    exclude_cols = [
        # Identifiers & Metadata
        'row_id', 'id', 'customer_id', 'address_id', 'policy_id', 
        'surname', 'first_name', 'name', 'email', 'phone', 'customer_name',
        
        # Operational & Behavioral Leakage (Post-Inception / Admin Noise)
        'customer_service_contacts', 'previous_insurer', 'campaign_code',
        'document_preference', 'payment_method', 'channel',
        
        # Targets & Administrative Dates
        'renewal_decision', 'total_claim_amount', 'claim_count', 
        'period_start', 'period_end'
    ]
    
                    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Ensure proper typing for LightGBM
    for c in feature_cols:
        if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]) or df[c].dtype.name == 'category':
            df[c] = df[c].astype(str).fillna('MISSING').astype('category')
        elif pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = df[c].astype(str).fillna('MISSING').astype('category')
            
    return df, feature_cols

def train_line_model(line_name):
    print(f"\n==============================================")
    print(f" Training Part 2 Model for Line: {line_name.upper()}")
    print(f"==============================================")
    
    train_path = Path(f"data/processed_p2/emerging_{line_name}_train.parquet")
    df = pd.read_parquet(train_path)
    
    y = pd.to_numeric(df['total_claim_amount'], errors='coerce').fillna(0.0).values
    exposure = pd.to_numeric(df['exposure'], errors='coerce').fillna(1.0).values
    exposure = np.maximum(exposure, 0.001)
    
    df_clean, feature_cols = preprocess_features(df)
    X = df_clean[feature_cols]
    
    print(f"Target Total Net Incurred: {y.sum():,.2f} CU across {len(X):,} rows and {len(feature_cols)} features.")
    
    # Tuned hyperparameters for heavy-tailed, zero-inflated insurance loss modeling
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'tweedie',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr, exp_tr = X.iloc[train_idx], y[train_idx], exposure[train_idx]
        X_va, y_va, exp_va = X.iloc[val_idx], y[val_idx], exposure[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=exp_tr)
        val_data = lgb.Dataset(X_va, label=y_va, weight=exp_va, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=800,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds_va = model.predict(X_va)
        score = tweedie_deviance(y_va, preds_va, p=1.5, sample_weight=exp_va)
        val_scores.append(score)
        models.append(model)
        print(f"Fold {fold+1} Validation Tweedie Deviance: {score:.5f}")
        
    print(f"Mean {line_name.upper()} Validation Deviance: {np.mean(val_scores):.5f}")
    
    out_dir = Path("models_p2")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': models, 'feature_cols': feature_cols}, out_dir / f"model_{line_name}.pkl")
    print(f"Saved ensemble to models_p2/model_{line_name}.pkl")

def run_all_lines():
    for line in ['motor', 'buildings', 'contents']:
        train_line_model(line)

if __name__ == "__main__":
    run_all_lines()